"""
test_powell_damping.py
======================

Compare standard SSBroyden with a Powell-damped variant on the 1D PINN BVP
(same wavenumber-4 problem as Chapter 4 of the thesis). The motivation is
the staircase / piecewise-constant pattern observed in second-order loss
curves.

Mechanism the script tests
--------------------------
The base SSBroyden optimiser requires the curvature condition
`y_k^T s_k > damping`. When it fails (which happens on flat / sharp regions
of the PINN landscape) the optimiser skips its update and, with
`reset_on_fail=True`, resets H_k to the identity. The next iterations are
effectively steepest descent until a productive curvature pair appears,
which produces visible plateaus and sudden cliffs in the loss curve.

Powell (1978) damping replaces y_k with

    y_bar_k = theta_k y_k + (1 - theta_k) B_k s_k

    theta_k = 1                                          if r >= eta
            = (1 - eta) s_B_s / (s_B_s - y_T_s)          otherwise

with r = y_T_s / s_B_s and the standard choice eta = 0.2. The damped pair
satisfies y_bar_k^T s_k >= eta s_k^T B_k s_k > 0, so the update never has
to be skipped.

In the inverse-Hessian formulation used here the search direction is
`p_k = -H_k g_k`, hence `s_k = -alpha H_k g_k` and `B_k s_k = -alpha g_k`.
This gives `s_k^T B_k s_k = -alpha (s_k^T g_k)` and lets us evaluate the
damping criterion without inverting H.

Caveat
------
Powell damping is theoretically derived for plain BFGS. Layering it on top
of the SSBroyden self-scaling factor tau_k gives a heuristic combination,
not a published algorithm. Treat the result as a diagnostic about *why*
the staircase appears, not as a proposed final method.

Usage
-----
    python test_powell_damping.py --epochs 5000 --adam-epochs 2000

Outputs are written to ``../results/powell_damping_<timestamp>/``.
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
        reset_on_fail: bool = True,
        H_on_cpu: bool = False,
    ) -> None:
        if variant not in self._VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {self._VALID_VARIANTS}, got {variant!r}"
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
            reset_on_fail=reset_on_fail,
            H_on_cpu=H_on_cpu,
        )
        super().__init__(params, defaults)
        self.H: torch.Tensor | None = None

        # Per-iteration diagnostics. NaN means "not recorded this iteration".
        self.diag: dict[str, list[float]] = {
            "alpha": [],
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
        reset_on_fail = group["reset_on_fail"]
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

        # Armijo backtracking line search (matches base optimiser).
        alpha = lr
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
            if reset_on_fail:
                self._init_H(n, g, H_on_cpu)
            self._record(alpha=0.0, ls_failed=1.0, skipped_update=1.0)
            return loss

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
            if reset_on_fail:
                self._init_H(n, g_new, H_on_cpu)
            self._record(
                alpha=alpha,
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
            if reset_on_fail:
                self._init_H(n, g_new, H_on_cpu)
            self._record(
                alpha=alpha,
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
            ys_raw=float(ys_raw.item()),
            ys_eff=float(ys_eff_t.item()),
            sBs=sBs_val,
            theta=theta,
            ls_failed=0.0,
            skipped_update=0.0,
        )
        return new_loss


# =============================================================================
# Runner
# =============================================================================
def run_one(label: str, *, powell: bool, args: argparse.Namespace) -> dict[str, Any]:
    """Run one configuration and return logs + diagnostics."""
    print("\n" + "=" * 78)
    print(f"  Run: {label}  (powell_damping={powell})")
    print("=" * 78)

    # Identical seeding for both runs => identical Adam phase.
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

    # Replace the second-order optimiser with our Powell-damped version.
    pinn.quasi_newton = SSBroydenPowellOptimizer(
        model.parameters(),
        variant=args.qn_variant,
        powell_damping=powell,
        powell_eta=args.powell_eta,
        lr=1.0,
        line_search=True,
        c1=1e-4,
        backtrack=0.5,
        max_ls=20,
        damping=1e-12,
        tau_min=1e-6,
        tau_max=1.0,
        reset_on_fail=True,
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
        "powell": powell,
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
def plot_comparison(
    runs: list[dict[str, Any]], save_path: str, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colours = {"standard": "C3", "powell": "C0"}

    # (0,0) Train objective on log scale, both runs.
    ax = axes[0, 0]
    for r in runs:
        c = colours.get(r["label"], "C2")
        ax.semilogy(r["obj_train"], color=c, linewidth=1.0, label=f"{r['label']} (train)")
        ax.semilogy(
            r["obj_val"], color=c, linewidth=1.0, alpha=0.45, linestyle="--",
            label=f"{r['label']} (val)",
        )
    ax.axvline(args.adam_epochs, color="k", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.set_title("Objective J(theta) -- Adam->QN handoff at vertical line")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("J")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # (0,1) Relative L2 solution error.
    ax = axes[0, 1]
    for r in runs:
        c = colours.get(r["label"], "C2")
        ax.semilogy(r["sol_rel_l2"], color=c, linewidth=1.2, label=r["label"])
    ax.axvline(args.adam_epochs, color="k", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.set_title("Relative L^2 solution error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("rel-L2")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # (1,0) Per-QN-iteration accepted alpha (line-search step length).
    ax = axes[1, 0]
    for r in runs:
        c = colours.get(r["label"], "C2")
        alpha_arr = np.asarray(r["diag"]["alpha"], dtype=np.float64)
        if alpha_arr.size:
            iters = np.arange(args.adam_epochs + 1, args.adam_epochs + 1 + alpha_arr.size)
            ax.semilogy(iters, np.maximum(alpha_arr, 1e-30), color=c, linewidth=0.9,
                        label=r["label"])
    ax.set_title("Accepted line-search step length alpha_k")
    ax.set_xlabel("Epoch (QN phase only)")
    ax.set_ylabel("alpha")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # (1,1) Powell theta history + skipped-update rate per run.
    ax = axes[1, 1]
    for r in runs:
        c = colours.get(r["label"], "C2")
        theta_arr = np.asarray(r["diag"]["theta"], dtype=np.float64)
        skipped = np.asarray(r["diag"]["skipped_update"], dtype=np.float64)
        if theta_arr.size:
            iters = np.arange(args.adam_epochs + 1, args.adam_epochs + 1 + theta_arr.size)
            ax.plot(iters, theta_arr, color=c, linewidth=0.9,
                    label=f"{r['label']}: theta_k")
            n_total = max(skipped.size, 1)
            n_skip = int(np.nansum(skipped))
            ax.text(
                0.02, 0.92 - 0.08 * runs.index(r), transform=ax.transAxes,
                s=f"{r['label']}: skipped updates {n_skip}/{n_total} "
                  f"({100.0 * n_skip / n_total:.1f}%)",
                color=c, fontsize=9,
            )
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("Powell damping factor theta_k (1.0 = no damping)")
    ax.set_xlabel("Epoch (QN phase only)")
    ax.set_ylabel("theta")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison figure to: {save_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare standard vs Powell-damped SSBroyden on the 1D PINN BVP."
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
        f"powell_damping_k{args.wavenumber:g}_{args.qn_variant}_{run_tag}",
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nWriting outputs to: {os.path.abspath(save_dir)}")

    runs = [
        run_one("standard", powell=False, args=args),
        run_one("powell", powell=True, args=args),
    ]

    # Persist per-run logs as .npz so the thesis figures can be regenerated.
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
                "powell": r["powell"],
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
    for r in summary["runs"]:
        print(
            f"  {r['label']:>9s}: "
            f"final J(train) = {r['final_obj_train']:.3e}, "
            f"rel-L2 = {r['final_sol_rel_l2']:.3e}, "
            f"skipped = {r['qn_skipped_updates']}/{r['qn_iters_recorded']}, "
            f"ls_failed = {r['qn_ls_failed']}"
        )


if __name__ == "__main__":
    main()
