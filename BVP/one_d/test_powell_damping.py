"""
test_powell_damping.py
======================

Ablation script for the staircase / piecewise-constant pattern observed in
the second-order phase of the 1D PINN BVP (same wavenumber-4 problem as
Chapter 4 of the thesis).

Background
----------
A first version of this script tested only Powell damping on the curvature
pair y_k. Empirically it changed nothing: when the line search rejects every
trial step, alpha_k is set to 0 and the update is skipped *before* y_k is
ever computed, so Powell damping has no surface on which to act.

This version sweeps several knobs that operate upstream of (or alongside)
Powell damping:

  * ``reset_on_ls_fail``: whether to reset H_k to the identity when the
    line search fails. Setting this to False preserves the curvature
    information from previous successful steps instead of falling back to
    plain steepest descent on every line-search failure.
  * ``alpha0_mode``: how the line search picks its initial step length.
        - ``unit``     alpha_0 = lr (current default; can be much too large
                       on PINN losses with O(1e3) gradient norms)
        - ``inv_norm`` alpha_0 = lr / max(1, ||p_k||) (scale-aware start)
        - ``carry``    alpha_0 = 2 * (last accepted alpha) capped at lr
                       (Nocedal-Wright p.59 heuristic)
  * Powell damping (kept available for the curvature pair, but treat it as
    secondary to the two knobs above).

Default ablation configurations
-------------------------------
  * standard          - reset_on_ls_fail=True, alpha0=unit (baseline)
  * no_reset_on_ls    - reset_on_ls_fail=False
  * alpha_inv_norm    - alpha0=inv_norm
  * alpha_carry       - alpha0=carry
  * combined          - no reset on LS fail + inv-norm initial step

Pass ``--include-powell`` to add a sixth ``combined_powell`` config.

Powell damping (kept for completeness)
--------------------------------------
Powell (1978) replaces y_k with

    y_bar_k = theta_k y_k + (1 - theta_k) B_k s_k

with theta_k = 1 if y_k^T s_k >= eta s_k^T B_k s_k, else
theta_k = (1 - eta) s_B_s / (s_B_s - y_T_s). In the inverse-Hessian
formulation here, B_k s_k = -alpha g_k so s_k^T B_k s_k = -alpha s_k^T g_k
is computed without inverting H.

Usage
-----
    python test_powell_damping.py --epochs 5000 --adam-epochs 2000
    python test_powell_damping.py --include-powell --max-ls 50
    python test_powell_damping.py --only standard combined --epochs 3000

Outputs are written to ``../results/ls_ablation_<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Make sibling helpers importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pinn_ssbroyden_1d import (  # noqa: E402
    A,
    B,
    DEFAULT_K,
    NeuralNetwork,
    PINN_BVP_SSBroyden,
    device,
)


# =============================================================================
# Powell-damped SSBroyden optimiser
# =============================================================================
class SSBroydenPowellOptimizer(optim.Optimizer):
    """SSBroyden update with optional Powell damping of the curvature pair.

    Mirrors ``SSBroydenOptimizer`` from ``BVP/optimizers/ssbroyden.py`` but
    inlines a Powell modification when ``powell_damping=True``. Per-iteration
    diagnostics (alpha, ys, sBs, theta, line-search failures, update skips)
    are recorded in ``self.diag`` for plotting.
    """

    _VALID_VARIANTS = ("bfgs", "ssbfgs", "ssbroyden")
    _VALID_ALPHA0 = ("unit", "inv_norm", "carry")

    def __init__(
        self,
        params,
        *,
        variant: str = "ssbroyden",
        powell_damping: bool = False,
        powell_eta: float = 0.2,
        lr: float = 1.0,
        line_search: bool = True,
        c1: float = 1e-4,
        backtrack: float = 0.5,
        max_ls: int = 20,
        damping: float = 1e-12,
        tau_min: float = 1e-6,
        tau_max: float = 1.0,
        reset_on_ls_fail: bool = True,
        reset_on_curv_fail: bool = True,
        alpha0_mode: str = "unit",
        H_on_cpu: bool = False,
    ) -> None:
        if variant not in self._VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {self._VALID_VARIANTS}, got {variant!r}"
            )
        if alpha0_mode not in self._VALID_ALPHA0:
            raise ValueError(
                f"alpha0_mode must be one of {self._VALID_ALPHA0}, got {alpha0_mode!r}"
            )
        if not 0.0 < powell_eta < 1.0:
            raise ValueError("powell_eta must lie in (0, 1).")

        defaults = dict(
            variant=variant,
            powell_damping=powell_damping,
            powell_eta=powell_eta,
            lr=lr,
            line_search=line_search,
            c1=c1,
            backtrack=backtrack,
            max_ls=max_ls,
            damping=damping,
            tau_min=tau_min,
            tau_max=tau_max,
            reset_on_ls_fail=reset_on_ls_fail,
            reset_on_curv_fail=reset_on_curv_fail,
            alpha0_mode=alpha0_mode,
            H_on_cpu=H_on_cpu,
        )
        super().__init__(params, defaults)
        self.H: torch.Tensor | None = None
        self._last_alpha: float = float(lr)

        # Per-iteration diagnostics. NaN means "not recorded this iteration".
        self.diag: dict[str, list[float]] = {
            "alpha": [],
            "alpha0": [],
            "ys_raw": [],
            "ys_eff": [],
            "sBs": [],
            "theta": [],
            "ls_failed": [],
            "skipped_update": [],
        }

    # ---------- vector helpers (same as base) ----------
    def _get_param_vector(self) -> torch.Tensor:
        return torch.cat(
            [p.data.view(-1) for g in self.param_groups for p in g["params"]]
        )

    def _set_param_vector(self, vec: torch.Tensor) -> None:
        offset = 0
        for g in self.param_groups:
            for p in g["params"]:
                n = p.numel()
                p.data.copy_(vec[offset : offset + n].view_as(p))
                offset += n

    def _get_grad_vector(self) -> torch.Tensor:
        grads = []
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p.data).view(-1))
                else:
                    grads.append(p.grad.data.view(-1))
        return torch.cat(grads)

    @torch.no_grad()
    def _init_H(self, n: int, ref_tensor: torch.Tensor, H_on_cpu: bool) -> None:
        if self.H is None or self.H.shape[0] != n:
            dev = torch.device("cpu") if H_on_cpu else ref_tensor.device
            self.H = torch.eye(n, device=dev, dtype=ref_tensor.dtype)

    def _record(self, **kwargs: float) -> None:
        for key in self.diag.keys():
            self.diag[key].append(float(kwargs.get(key, float("nan"))))

    # ---------- main step ----------
    def step(self, closure, loss_eval):  # type: ignore[override]
        group = self.param_groups[0]
        variant = group["variant"]
        powell_damping = group["powell_damping"]
        powell_eta = group["powell_eta"]
        lr = group["lr"]
        line_search = group["line_search"]
        c1 = group["c1"]
        backtrack = group["backtrack"]
        max_ls = group["max_ls"]
        damping = group["damping"]
        tau_min = group["tau_min"]
        tau_max = group["tau_max"]
        reset_on_ls_fail = group["reset_on_ls_fail"]
        reset_on_curv_fail = group["reset_on_curv_fail"]
        alpha0_mode = group["alpha0_mode"]
        H_on_cpu = group["H_on_cpu"]

        loss = closure()
        g = self._get_grad_vector().detach()
        x = self._get_param_vector().detach()
        n = g.numel()
        self._init_H(n, g, H_on_cpu)

        gH = g.detach().cpu() if self.H.device != g.device else g
        Hg = self.H.matmul(gH)
        p_dir = (-Hg).to(g.device) if self.H.device != g.device else -Hg

        gTp = torch.dot(g, p_dir).item()
        f0 = float(loss.item())

        # Initial step length according to alpha0_mode.
        if alpha0_mode == "unit":
            alpha0 = float(lr)
        elif alpha0_mode == "inv_norm":
            p_norm = float(torch.linalg.vector_norm(p_dir).item())
            alpha0 = float(lr) / max(1.0, p_norm)
        elif alpha0_mode == "carry":
            # Try roughly twice the previous successful step (Nocedal & Wright p.59),
            # capped at lr; fall back to a tiny floor on the first iteration.
            alpha0 = min(float(lr), max(self._last_alpha * 2.0, 1e-12))
        else:  # pragma: no cover - validated in __init__
            raise ValueError(f"Unknown alpha0_mode {alpha0_mode!r}")

        # Armijo backtracking line search.
        alpha = alpha0
        if line_search:
            for _ in range(max_ls):
                x_try = x + alpha * p_dir
                self._set_param_vector(x_try)
                f_try = float(loss_eval().item())
                if f_try <= f0 + c1 * alpha * gTp:
                    break
                alpha *= backtrack
            else:
                alpha = 0.0

        if alpha == 0.0 or not np.isfinite(alpha):
            self._set_param_vector(x)
            if reset_on_ls_fail:
                self._init_H(n, g, H_on_cpu)
            # Shrink the carry value so we try a smaller step next time.
            if alpha0_mode == "carry":
                self._last_alpha = max(self._last_alpha * backtrack, 1e-12)
            self._record(alpha=0.0, alpha0=alpha0, ls_failed=1.0, skipped_update=1.0)
            return loss

        # Line search succeeded: remember the accepted step for "carry" mode.
        self._last_alpha = float(alpha)

        s = alpha * p_dir
        x_new = x + s
        self._set_param_vector(x_new)

        new_loss = closure()
        g_new = self._get_grad_vector().detach()
        y = g_new - g

        # Cast to the device where H lives.
        if self.H.device != g.device:
            yH = y.detach().cpu()
            sH = s.detach().cpu()
            gH2 = g.detach().cpu()
            HgH2 = Hg.detach()
        else:
            yH = y
            sH = s
            gH2 = g
            HgH2 = Hg

        ys_raw = torch.dot(yH, sH)

        # s^T B s without inverting H: s = -alpha H g => B s = -alpha g.
        # Hence s^T B s = -alpha (s^T g) = alpha^2 g^T H g.
        sBs = -alpha * torch.dot(sH, gH2)
        if not torch.isfinite(sBs) or sBs.item() <= damping:
            # Pathological geometry; fall through to the standard guard below.
            sBs_val = max(damping, float(sBs.item()) if torch.isfinite(sBs) else damping)
        else:
            sBs_val = float(sBs.item())

        # ----- Powell damping of the curvature pair -----
        if powell_damping:
            ys_val = float(ys_raw.item())
            if ys_val >= powell_eta * sBs_val:
                theta = 1.0
                y_eff = yH
            else:
                denom = sBs_val - ys_val
                # denom >= (1 - eta) sBs > 0 whenever sBs_val > 0.
                denom_safe = denom if denom > damping else damping
                theta = (1.0 - powell_eta) * sBs_val / denom_safe
                Bs = -alpha * gH2  # B_k s_k in the inverse-Hessian formulation.
                y_eff = theta * yH + (1.0 - theta) * Bs
            ys_eff_t = torch.dot(y_eff, sH)
        else:
            theta = 1.0
            y_eff = yH
            ys_eff_t = ys_raw

        # Curvature guard on the effective pair.
        if (not torch.isfinite(ys_eff_t)) or (ys_eff_t.abs() <= damping):
            if reset_on_curv_fail:
                self._init_H(n, g_new, H_on_cpu)
            self._record(
                alpha=alpha,
                alpha0=alpha0,
                ys_raw=float(ys_raw.item()),
                ys_eff=float(ys_eff_t.item()),
                sBs=sBs_val,
                theta=theta,
                ls_failed=0.0,
                skipped_update=1.0,
            )
            return new_loss

        Hy = self.H.matmul(y_eff)
        yHy = torch.dot(y_eff, Hy)
        if (not torch.isfinite(yHy)) or (yHy.abs() <= damping):
            if reset_on_curv_fail:
                self._init_H(n, g_new, H_on_cpu)
            self._record(
                alpha=alpha,
                alpha0=alpha0,
                ys_raw=float(ys_raw.item()),
                ys_eff=float(ys_eff_t.item()),
                sBs=sBs_val,
                theta=theta,
                ls_failed=0.0,
                skipped_update=1.0,
            )
            return new_loss

        ys = ys_eff_t  # alias to keep formulas below readable

        # Scaling factor tau_k and weight phi_k -- branch by variant.
        if variant == "bfgs":
            tau_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
            phi_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
        elif variant == "ssbfgs":
            sTg = torch.dot(sH, gH2)
            denom = -sTg
            denom_safe = torch.clamp(denom, min=damping)
            tau_k = torch.minimum(
                torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype),
                ys / denom_safe,
            )
            tau_k = torch.clamp(tau_k, min=tau_min, max=tau_max)
            phi_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
        elif variant == "ssbroyden":
            sTg = torch.dot(sH, gH2)
            b_k = (-alpha * sTg) / ys
            h_k = yHy / ys
            a_k = h_k * b_k - 1.0

            if (
                (not torch.isfinite(a_k))
                or (a_k <= damping)
                or (not torch.isfinite(b_k))
                or (b_k <= damping)
            ):
                tau_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
                phi_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
            else:
                c_k = torch.sqrt(torch.clamp(a_k / (a_k + 1.0), min=0.0))
                rho_minus = torch.minimum(
                    torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype),
                    h_k * (1.0 - c_k),
                )
                rho_safe = torch.clamp(rho_minus, min=damping)

                theta_minus = rho_minus - (1.0 / a_k)
                theta_plus = 1.0 / rho_safe
                theta_hat = (1.0 - b_k) / torch.clamp(b_k, min=damping)

                theta_k_t = torch.maximum(
                    theta_minus, torch.minimum(theta_plus, theta_hat)
                )
                sigma_k = 1.0 + a_k * theta_k_t
                sigma_safe = torch.clamp(sigma_k, min=damping)

                gHg = torch.dot(gH2, HgH2)
                denom = (alpha * alpha) * torch.clamp(gHg, min=damping)
                tau1 = torch.minimum(
                    torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype),
                    ys / denom,
                )

                sigma_pow = torch.exp(-torch.log(sigma_safe) / (n - 1.0))

                if theta_k_t > 0:
                    tau2 = tau1 * torch.minimum(
                        sigma_pow, 1.0 / torch.clamp(theta_k_t, min=damping)
                    )
                else:
                    tau2 = torch.minimum(tau1 * sigma_pow, sigma_safe)

                tau_k = torch.clamp(tau2, min=tau_min, max=tau_max)
                phi_k = (1.0 - theta_k_t) / sigma_safe
        else:  # pragma: no cover
            raise ValueError(f"Unknown variant {variant!r}")

        # Inverse-Hessian update (eq. 10 in the paper) using the (possibly damped) y_eff.
        v = torch.sqrt(torch.clamp(yHy, min=damping)) * (sH / ys - Hy / yHy)
        term = self.H - torch.outer(Hy, Hy) / yHy + phi_k * torch.outer(v, v)
        H_new = (1.0 / tau_k) * term + torch.outer(sH, sH) / ys
        self.H = 0.5 * (H_new + H_new.t())

        self._record(
            alpha=alpha,
            alpha0=alpha0,
            ys_raw=float(ys_raw.item()),
            ys_eff=float(ys_eff_t.item()),
            sBs=sBs_val,
            theta=theta,
            ls_failed=0.0,
            skipped_update=0.0,
        )
        return new_loss


# =============================================================================
# Ablation configs
# =============================================================================
def default_ablation_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return the list of named configurations swept by the ablation runner."""
    base = dict(
        powell_damping=False,
        powell_eta=args.powell_eta,
        max_ls=args.max_ls,
        reset_on_ls_fail=True,
        reset_on_curv_fail=True,
        alpha0_mode="unit",
    )

    def cfg(label: str, **overrides: Any) -> dict[str, Any]:
        d = dict(base)
        d.update(overrides)
        d["label"] = label
        return d

    configs = [
        cfg("standard"),
        cfg("no_reset_on_ls", reset_on_ls_fail=False),
        cfg("alpha_inv_norm", alpha0_mode="inv_norm"),
        cfg("alpha_carry", alpha0_mode="carry"),
        cfg(
            "combined",
            reset_on_ls_fail=False,
            alpha0_mode="inv_norm",
        ),
    ]
    if args.include_powell:
        configs.append(
            cfg(
                "combined_powell",
                reset_on_ls_fail=False,
                alpha0_mode="inv_norm",
                powell_damping=True,
            )
        )
    return configs


# =============================================================================
# Runner
# =============================================================================
def run_one(cfg: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    """Run one configuration and return logs + diagnostics."""
    label = cfg["label"]
    print("\n" + "=" * 78)
    print(f"  Run: {label}")
    print(
        f"    powell={cfg['powell_damping']}  reset_on_ls_fail={cfg['reset_on_ls_fail']}  "
        f"reset_on_curv_fail={cfg['reset_on_curv_fail']}  "
        f"alpha0_mode={cfg['alpha0_mode']}  max_ls={cfg['max_ls']}"
    )
    print("=" * 78)

    # Identical seeding => identical Adam phase across all configs.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = NeuralNetwork(hidden_layers=tuple(args.hidden), activation=nn.Tanh())

    pinn = PINN_BVP_SSBroyden(
        model=model,
        k=args.wavenumber,
        lr=args.lr,
        lambda_pde=1.0,
        loss_transform="identity",
        loss_lambda=0.5,
        loss_eps=1e-12,
        qn_variant=args.qn_variant,
    )

    pinn.quasi_newton = SSBroydenPowellOptimizer(
        model.parameters(),
        variant=args.qn_variant,
        powell_damping=cfg["powell_damping"],
        powell_eta=cfg["powell_eta"],
        lr=1.0,
        line_search=True,
        c1=1e-4,
        backtrack=0.5,
        max_ls=cfg["max_ls"],
        damping=1e-12,
        tau_min=1e-6,
        tau_max=1.0,
        reset_on_ls_fail=cfg["reset_on_ls_fail"],
        reset_on_curv_fail=cfg["reset_on_curv_fail"],
        alpha0_mode=cfg["alpha0_mode"],
        H_on_cpu=False,
    )

    pinn.train(
        n_epochs=args.epochs,
        n_collocation=args.n_collocation,
        train_split=0.8,
        resample_every=args.resample_every,
        adam_epochs=args.adam_epochs,
        verbose_freq=max(1, args.epochs // 25),
        diag_grid_n=400,
        patience=args.epochs,
        min_delta=1e-12,
        moving_avg_window=20,
    )

    return {
        "label": label,
        "config": {k: v for k, v in cfg.items() if k != "label"},
        "obj_train": list(pinn.obj_train),
        "obj_val": list(pinn.obj_val),
        "J_train": list(pinn.J_train),
        "J_val": list(pinn.J_val),
        "pde_l2": list(pinn.pde_l2),
        "sol_l2": list(pinn.sol_l2),
        "sol_rel_l2": list(pinn.sol_rel_l2),
        "diag": dict(pinn.quasi_newton.diag),
        "adam_epochs": args.adam_epochs,
        "best_val_ma": float(pinn.best_val_ma),
    }


# =============================================================================
# Plotting
# =============================================================================
def _colour_for(idx: int) -> str:
    palette = ["C3", "C0", "C2", "C4", "C5", "C1", "C6", "C7", "C8", "C9"]
    return palette[idx % len(palette)]


def plot_comparison(
    runs: list[dict[str, Any]], save_path: str, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (0,0) Train + val objective on log scale.
    ax = axes[0, 0]
    for i, r in enumerate(runs):
        c = _colour_for(i)
        ax.semilogy(r["obj_train"], color=c, linewidth=1.0, label=f"{r['label']} (train)")
        ax.semilogy(
            r["obj_val"], color=c, linewidth=1.0, alpha=0.4, linestyle="--",
            label=f"{r['label']} (val)",
        )
    ax.axvline(args.adam_epochs, color="k", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.set_title("Objective J(theta) -- Adam->QN handoff at vertical line")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("J")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    # (0,1) Relative L2 solution error.
    ax = axes[0, 1]
    for i, r in enumerate(runs):
        c = _colour_for(i)
        ax.semilogy(r["sol_rel_l2"], color=c, linewidth=1.1, label=r["label"])
    ax.axvline(args.adam_epochs, color="k", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.set_title("Relative L^2 solution error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("rel-L2")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # (1,0) Per-QN-iteration accepted alpha (log scale, floored).
    ax = axes[1, 0]
    for i, r in enumerate(runs):
        c = _colour_for(i)
        alpha_arr = np.asarray(r["diag"]["alpha"], dtype=np.float64)
        if alpha_arr.size:
            iters = np.arange(args.adam_epochs + 1, args.adam_epochs + 1 + alpha_arr.size)
            ax.semilogy(
                iters, np.maximum(alpha_arr, 1e-30), color=c, linewidth=0.7, alpha=0.8,
                label=r["label"],
            )
    ax.set_title("Accepted line-search step length alpha_k (0 -> 1e-30 floor)")
    ax.set_xlabel("Epoch (QN phase)")
    ax.set_ylabel("alpha")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # (1,1) Bar chart of LS-failure and skipped-update rate per run.
    ax = axes[1, 1]
    labels = [r["label"] for r in runs]
    ls_rates: list[float] = []
    skip_rates: list[float] = []
    for r in runs:
        ls = np.asarray(r["diag"]["ls_failed"], dtype=np.float64)
        sk = np.asarray(r["diag"]["skipped_update"], dtype=np.float64)
        n_total = max(ls.size, 1)
        ls_rates.append(100.0 * np.nansum(ls) / n_total)
        skip_rates.append(100.0 * np.nansum(sk) / n_total)
    xpos = np.arange(len(labels))
    width = 0.38
    ax.bar(xpos - width / 2, ls_rates, width, color="C3", alpha=0.85,
           label="LS failure %")
    ax.bar(xpos + width / 2, skip_rates, width, color="C0", alpha=0.85,
           label="skipped update %")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("% of QN iterations")
    ax.set_ylim(0, 105)
    ax.set_title("Failure rates over the QN phase")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    for i, (lr, sr) in enumerate(zip(ls_rates, skip_rates)):
        ax.text(i - width / 2, lr + 1.5, f"{lr:.1f}", ha="center",
                fontsize=7, color="C3")
        ax.text(i + width / 2, sr + 1.5, f"{sr:.1f}", ha="center",
                fontsize=7, color="C0")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison figure to: {save_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Ablation: vary reset-on-LS-failure and alpha0 strategies on the 1D "
            "PINN BVP, with optional Powell damping. Each named config is run "
            "with identical seed/network/Adam warm-up so any difference comes "
            "from the QN update."
        )
    )
    p.add_argument("--wavenumber", type=float, default=DEFAULT_K)
    p.add_argument(
        "--qn-variant",
        type=str,
        default="ssbroyden",
        choices=["bfgs", "ssbfgs", "ssbroyden"],
    )
    p.add_argument("--powell-eta", type=float, default=0.2,
                   help="Powell threshold; standard choice is 0.2.")
    p.add_argument("--max-ls", type=int, default=20,
                   help="Backtracking budget per outer QN step.")
    p.add_argument("--include-powell", action="store_true",
                   help="Add a 'combined_powell' config to the sweep.")
    p.add_argument("--only", type=str, nargs="+", default=None,
                   help="Restrict the sweep to these config labels.")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--adam-epochs", type=int, default=2000)
    p.add_argument("--n-collocation", type=int, default=400)
    p.add_argument("--resample-every", type=int, default=500)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join("..", "results"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.results_dir,
        f"ls_ablation_k{args.wavenumber:g}_{args.qn_variant}_{run_tag}",
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nWriting outputs to: {os.path.abspath(save_dir)}")

    configs = default_ablation_configs(args)
    if args.only is not None:
        wanted = set(args.only)
        configs = [c for c in configs if c["label"] in wanted]
        if not configs:
            raise SystemExit(
                f"--only filter {args.only!r} matched no configs. Available: "
                f"{[c['label'] for c in default_ablation_configs(args)]}"
            )

    runs = [run_one(cfg, args=args) for cfg in configs]

    for r in runs:
        np.savez(
            os.path.join(save_dir, f"history_{r['label']}.npz"),
            obj_train=np.asarray(r["obj_train"], dtype=np.float64),
            obj_val=np.asarray(r["obj_val"], dtype=np.float64),
            J_train=np.asarray(r["J_train"], dtype=np.float64),
            J_val=np.asarray(r["J_val"], dtype=np.float64),
            pde_l2=np.asarray(r["pde_l2"], dtype=np.float64),
            sol_l2=np.asarray(r["sol_l2"], dtype=np.float64),
            sol_rel_l2=np.asarray(r["sol_rel_l2"], dtype=np.float64),
            alpha=np.asarray(r["diag"]["alpha"], dtype=np.float64),
            alpha0=np.asarray(r["diag"]["alpha0"], dtype=np.float64),
            ys_raw=np.asarray(r["diag"]["ys_raw"], dtype=np.float64),
            ys_eff=np.asarray(r["diag"]["ys_eff"], dtype=np.float64),
            sBs=np.asarray(r["diag"]["sBs"], dtype=np.float64),
            theta=np.asarray(r["diag"]["theta"], dtype=np.float64),
            ls_failed=np.asarray(r["diag"]["ls_failed"], dtype=np.float64),
            skipped_update=np.asarray(r["diag"]["skipped_update"], dtype=np.float64),
        )

    plot_comparison(runs, os.path.join(save_dir, "comparison.png"), args)

    summary = {
        "args": vars(args),
        "runs": [
            {
                "label": r["label"],
                "config": r["config"],
                "best_val_objective_ma": r["best_val_ma"],
                "final_obj_train": r["obj_train"][-1] if r["obj_train"] else None,
                "final_obj_val": r["obj_val"][-1] if r["obj_val"] else None,
                "final_pde_l2": r["pde_l2"][-1] if r["pde_l2"] else None,
                "final_sol_l2": r["sol_l2"][-1] if r["sol_l2"] else None,
                "final_sol_rel_l2": r["sol_rel_l2"][-1] if r["sol_rel_l2"] else None,
                "qn_iters_recorded": len(r["diag"]["alpha"]),
                "qn_skipped_updates": int(
                    np.nansum(np.asarray(r["diag"]["skipped_update"], dtype=np.float64))
                ),
                "qn_ls_failed": int(
                    np.nansum(np.asarray(r["diag"]["ls_failed"], dtype=np.float64))
                ),
            }
            for r in runs
        ],
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nSummary:")
    name_w = max((len(r["label"]) for r in summary["runs"]), default=10)
    for r in summary["runs"]:
        n_total = max(r["qn_iters_recorded"], 1)
        ls_pct = 100.0 * r["qn_ls_failed"] / n_total
        skip_pct = 100.0 * r["qn_skipped_updates"] / n_total
        print(
            f"  {r['label']:>{name_w}s}: "
            f"J(train)={r['final_obj_train']:.3e}, "
            f"rel-L2={r['final_sol_rel_l2']:.3e}, "
            f"LS_fail={r['qn_ls_failed']}/{n_total} ({ls_pct:.1f}%), "
            f"skipped={r['qn_skipped_updates']}/{n_total} ({skip_pct:.1f}%)"
        )


if __name__ == "__main__":
    main()
