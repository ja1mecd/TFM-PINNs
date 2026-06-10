"""Vacuum Grad-Shafranov PINN validation run — closes the gap of section 4.3.2
of the thesis, which currently defers the psi = R^2 numerical results.

Operator (vacuum / current-free Grad-Shafranov, Delta-star):

    Delta*_psi = psi_RR - (1/R) psi_R + psi_ZZ = 0,   on [R_LO, R_HI] x [Z_LO, Z_HI],

with the exact test solution

    psi_exact(R, Z) = R^2,

which satisfies Delta*_psi = 0 exactly (psi_RR = 2, (1/R) psi_R = (1/R)(2R) = 2,
psi_ZZ = 0). The box keeps R bounded away from 0 so the 1/R coefficient stays
regular while still exercising the variable-coefficient term that distinguishes
this operator from the Laplacian.

Hard Dirichlet ansatz, in the spirit of the rest of chapter 4:

    psi_hat(R, Z; theta) = R^2 + b(R, Z) N(R, Z; theta),
    b(R, Z) = (R - R_LO)(R - R_HI)(Z - Z_LO)(Z - Z_HI),

so the surrogate matches psi = R^2 on the whole boundary regardless of the
weights, and the network correction b N is forced to zero by the residual. The
run is a genuine optimisation (the random initial N gives a nonzero residual the
optimiser must suppress) that doubles as an implementation check of the Delta*
operator and its automatic differentiation.

The training pipeline matches the 2D Poisson solver: Adam warm-up followed by
self-scaled Broyden refinement, identity loss by default, multi-seed averaging,
and a four-panel figure (exact, learnt, pointwise error, solution + residual L2
curves) plus a summary table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_OPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimizers")
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)
from ssbroyden import SSBroydenOptimizer  # noqa: E402


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# =============================================================================
# Problem (R bounded away from 0 to keep 1/R regular)
# =============================================================================
R_LO, R_HI = 1.0, 2.0
Z_LO, Z_HI = -1.0, 1.0
AREA = (R_HI - R_LO) * (Z_HI - Z_LO)


def psi_exact_np(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return R ** 2 + 0.0 * Z


# =============================================================================
# Network and PINN
# =============================================================================
class Net(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (32, 32, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 2
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, rz: torch.Tensor) -> torch.Tensor:
        return self.net(rz)


def hard_ansatz(rz: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """psi_hat(R, Z) = R^2 + (R-R_LO)(R-R_HI)(Z-Z_LO)(Z-Z_HI) N(R, Z)."""
    R = rz[:, 0:1]
    Z = rz[:, 1:2]
    bubble = (R - R_LO) * (R - R_HI) * (Z - Z_LO) * (Z - Z_HI)
    return R ** 2 + bubble * raw


class VacuumGSPINN:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        loss_transform: str = "identity",
        loss_lambda: float = 1.0,
        loss_eps: float = 1e-12,
        qn_variant: str = "ssbroyden",
    ) -> None:
        self.model = model.to(device)
        self.loss_transform = str(loss_transform)
        self.loss_lambda = float(loss_lambda)
        self.loss_eps = float(loss_eps)
        self.adam = optim.Adam(self.model.parameters(), lr=lr)
        self.qn = SSBroydenOptimizer(
            self.model.parameters(),
            variant=qn_variant,
            lr=1.0,
            line_search=True,
            c1=1e-4,
            backtrack=0.5,
            max_ls=20,
            damping=1e-12,
            tau_min=1e-6,
            tau_max=1.0,
            reset_on_fail=True,
        )

        self.J_train: list[float] = []
        self.J_val: list[float] = []
        self.sol_l2: list[float] = []
        self.sol_rel_l2: list[float] = []

        self.best_state: dict | None = None
        self.best_val_ma = float("inf")
        self.early_stop_epoch: int | None = None
        # Box-Cox is engaged only from the start of the QN phase (identity
        # during the Adam warm-up), matching the Helmholtz/CFGS sweeps.
        self._in_qn_phase = False

    def _psi_hat(self, rz: torch.Tensor) -> torch.Tensor:
        return hard_ansatz(rz, self.model(rz))

    def _residual(self, rz: torch.Tensor, create_graph_second: bool) -> torch.Tensor:
        rz = rz.to(device)
        if not rz.requires_grad:
            rz = rz.requires_grad_(True)
        psi = self._psi_hat(rz)
        grad = torch.autograd.grad(
            psi, rz, grad_outputs=torch.ones_like(psi), create_graph=True
        )[0]
        psi_R = grad[:, 0:1]
        psi_Z = grad[:, 1:2]
        psi_RR = torch.autograd.grad(
            psi_R,
            rz,
            grad_outputs=torch.ones_like(psi_R),
            create_graph=create_graph_second,
            retain_graph=True,
        )[0][:, 0:1]
        psi_ZZ = torch.autograd.grad(
            psi_Z,
            rz,
            grad_outputs=torch.ones_like(psi_Z),
            create_graph=create_graph_second,
        )[0][:, 1:2]
        R = rz[:, 0:1]
        # Vacuum Grad-Shafranov: Delta*_psi = psi_RR - (1/R) psi_R + psi_ZZ = 0.
        return psi_RR - (1.0 / R) * psi_R + psi_ZZ

    def _transform(self, J: torch.Tensor) -> torch.Tensor:
        eps = self.loss_eps
        # Identity during the Adam warm-up; the transform acts only once the
        # QN phase has started (set at handover in train()).
        if self.loss_transform == "identity" or not self._in_qn_phase:
            return J
        if self.loss_transform == "sqrt":
            return torch.sqrt(J + eps)
        if self.loss_transform == "log":
            return torch.log(J + eps)
        if self.loss_transform == "boxcox":
            lam = self.loss_lambda
            if lam == 0.0:
                return torch.log(J + eps)
            return torch.expm1(lam * torch.log(J + eps)) / lam
        raise ValueError(f"unknown transform {self.loss_transform!r}")

    def compute_loss(self, rz: torch.Tensor, create_graph_second: bool):
        rz = rz.detach().clone().requires_grad_(True)
        r = self._residual(rz, create_graph_second=create_graph_second)
        J_raw = torch.mean(r ** 2)
        return self._transform(J_raw), J_raw.detach()

    # ---- diagnostics ----
    def _eval_grid(self, n: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        rs = np.linspace(R_LO, R_HI, n).astype(np.float32)
        zs = np.linspace(Z_LO, Z_HI, n).astype(np.float32)
        RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
        flat = np.stack([RR.ravel(), ZZ.ravel()], axis=1)
        return RR, ZZ, torch.from_numpy(flat).to(device)

    def compute_sol_l2(self, n: int = 200) -> tuple[float, float]:
        RR, ZZ, t = self._eval_grid(n)
        with torch.no_grad():
            psi_pred = self._psi_hat(t).cpu().numpy().reshape(n, n)
        psi_true = psi_exact_np(RR, ZZ)
        diff = psi_pred - psi_true
        dr = (R_HI - R_LO) / (n - 1)
        dz = (Z_HI - Z_LO) / (n - 1)
        l2_abs = float(np.sqrt(np.sum(diff ** 2) * dr * dz))
        denom = float(np.sqrt(np.sum(psi_true ** 2) * dr * dz))
        l2_rel = l2_abs / (denom + 1e-12)
        return l2_abs, l2_rel

    # ---- training ----
    def train(
        self,
        n_epochs: int,
        adam_epochs: int,
        n_collocation: int,
        train_split: float = 0.8,
        resample_every: int = 500,
        verbose_freq: int = 500,
        diag_grid_n: int = 200,
        handover_strategy: str = "fixed",
        handover_max_adam_epochs: int = 10000,
        plateau_patience: int = 200,
        plateau_min_delta: float = 1e-4,
        loss_threshold: float = 1.0,
        gradnorm_threshold: float = 1e-3,
        patience: int = 500,
        min_delta: float = 1e-12,
        moving_avg_window: int = 20,
        early_stop: bool = True,
        es_patience: int = 300,
        es_window: int = 20,
        es_min_delta: float = 1e-4,
        es_stop_loss: float = 0.0,
    ) -> None:
        valid_strategies = ("fixed", "plateau", "loss_threshold", "gradnorm")
        if handover_strategy not in valid_strategies:
            raise ValueError(
                f"handover_strategy must be one of {valid_strategies}, "
                f"got {handover_strategy!r}"
            )

        n_train = max(1, min(int(n_collocation * train_split), n_collocation - 1))

        def resample():
            x = torch.rand(n_collocation, 2, device=device)
            x[:, 0] = R_LO + (R_HI - R_LO) * x[:, 0]
            x[:, 1] = Z_LO + (Z_HI - Z_LO) * x[:, 1]
            perm = torch.randperm(n_collocation, device=device)
            x = x[perm]
            return x[:n_train].detach().clone(), x[n_train:].detach().clone()

        x_train, x_val = resample()

        plateau_best = float("inf")
        plateau_no_improve = 0
        handover_done = False
        self.handover_epoch: int | None = None

        ma_buf: list[float] = []
        epochs_no_improve = 0

        es_hist: "deque[float]" = deque(maxlen=es_window)
        es_best_ma = float("inf")
        es_bad = 0
        es_stopped_at = None
        es_reason = ""

        for epoch in range(1, n_epochs + 1):
            if epoch != 1 and ((epoch - 1) % resample_every == 0):
                x_train, x_val = resample()

            use_adam = not handover_done
            grad_norm: float | None = None

            if use_adam:
                self.adam.zero_grad()
                J_obj, J_raw = self.compute_loss(x_train, create_graph_second=True)
                J_obj.backward()
                if handover_strategy == "gradnorm":
                    with torch.no_grad():
                        sq = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                sq += float((p.grad ** 2).sum().item())
                        grad_norm = float(sq ** 0.5)
                self.adam.step()
            else:
                holder: dict = {}

                def closure():
                    self.qn.zero_grad()
                    J_obj_c, J_raw_c = self.compute_loss(
                        x_train, create_graph_second=True
                    )
                    holder["J_raw"] = J_raw_c
                    J_obj_c.backward()
                    return J_obj_c

                def loss_eval():
                    J_obj_e, _ = self.compute_loss(x_train, create_graph_second=False)
                    return J_obj_e

                J_obj = self.qn.step(closure, loss_eval)
                J_raw = holder["J_raw"]

            with torch.set_grad_enabled(True):
                _, val_raw = self.compute_loss(x_val, create_graph_second=False)

            self.J_train.append(float(J_raw.item()))
            self.J_val.append(float(val_raw.item()))

            if use_adam:
                v = float(val_raw.item())
                if v + plateau_min_delta < plateau_best:
                    plateau_best = v
                    plateau_no_improve = 0
                else:
                    plateau_no_improve += 1

                handover_now = False
                if epoch >= handover_max_adam_epochs:
                    handover_now = True
                elif handover_strategy == "fixed":
                    handover_now = epoch >= adam_epochs
                elif handover_strategy == "plateau":
                    handover_now = plateau_no_improve >= plateau_patience
                elif handover_strategy == "loss_threshold":
                    handover_now = v < loss_threshold
                elif handover_strategy == "gradnorm":
                    handover_now = (
                        grad_norm is not None and grad_norm < gradnorm_threshold
                    )

                if handover_now:
                    handover_done = True
                    self._in_qn_phase = True
                    self.handover_epoch = epoch
                    self.best_val_ma = float("inf")
                    self.best_state = None
                    ma_buf = []
                    epochs_no_improve = 0
                    es_hist.clear()
                    es_best_ma = float("inf")
                    es_bad = 0
                    print(
                        f"  [handover] epoch {epoch}: Adam -> QN "
                        f"(strategy={handover_strategy}, "
                        f"plateau_no_improve={plateau_no_improve}, "
                        f"J_val={v:.3e})"
                    )
            else:
                ma_buf.append(float(val_raw.item()))
                if len(ma_buf) > moving_avg_window:
                    ma_buf.pop(0)
                val_ma = float(np.mean(ma_buf))

                if val_ma + min_delta < self.best_val_ma:
                    self.best_val_ma = val_ma
                    self.best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if early_stop:
                    jv = float(val_raw.item())
                    es_hist.append(jv)
                    if es_stop_loss > 0.0 and math.isfinite(jv) and jv <= es_stop_loss:
                        es_reason = f"J_val={jv:.3e} <= stop_loss={es_stop_loss:.1e}"
                        es_stopped_at = epoch
                    elif len(es_hist) == es_hist.maxlen:
                        ma = float(np.mean(es_hist))
                        if math.isfinite(ma) and ma < es_best_ma * (1.0 - es_min_delta):
                            es_best_ma = ma
                            es_bad = 0
                        else:
                            es_bad += 1
                            if es_bad >= es_patience:
                                es_reason = (
                                    f"no >{es_min_delta:.1e} rel. improvement in "
                                    f"MA(J_val, w={es_window}) for {es_patience} "
                                    f"epochs (MA={ma:.3e}, best={es_best_ma:.3e})"
                                )
                                es_stopped_at = epoch

            l2_abs, l2_rel = self.compute_sol_l2(n=diag_grid_n)
            self.sol_l2.append(l2_abs)
            self.sol_rel_l2.append(l2_rel)
            if epoch == 1 or (epoch % verbose_freq == 0):
                phase = "ADAM" if use_adam else "QN"
                print(
                    f"Epoch {epoch:6d} [{phase}] | J_train={self.J_train[-1]:.3e} | "
                    f"J_val={self.J_val[-1]:.3e} | solL2={l2_abs:.3e} | relL2={l2_rel:.3e}"
                )

            if es_stopped_at is not None:
                self.early_stop_epoch = epoch
                n_qn = epoch - (self.handover_epoch or adam_epochs)
                print(
                    f"  [QN early stop] epoch {epoch} ({n_qn} QN steps): "
                    f"{es_reason}"
                )
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


# =============================================================================
# Multi-seed
# =============================================================================
@dataclass(frozen=True)
class SeedResult:
    seed: int
    J_val: np.ndarray
    sol_l2: np.ndarray
    final_J_val: float
    final_sol_l2: float
    final_sol_rel_l2: float
    field_pred: np.ndarray


def run_seeds(
    seeds: tuple[int, ...],
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    hidden: tuple[int, ...],
    lr: float,
    qn_variant: str,
    loss_transform: str,
    loss_lambda: float,
    handover_strategy: str,
    handover_max_adam_epochs: int,
    plateau_patience: int,
    plateau_min_delta: float,
    patience: int,
    min_delta: float,
    moving_avg_window: int,
) -> tuple[SeedResult, ...]:
    out: list[SeedResult] = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = Net(hidden=hidden)
        pinn = VacuumGSPINN(
            model=model,
            lr=lr,
            loss_transform=loss_transform,
            loss_lambda=loss_lambda,
            qn_variant=qn_variant,
        )
        print(
            f"\n[seed={seed}] training vacuum Grad-Shafranov PINN  "
            f"(handover={handover_strategy}, plateau_patience={plateau_patience}, "
            f"qn_patience={patience})"
        )
        pinn.train(
            n_epochs=n_epochs,
            adam_epochs=adam_epochs,
            n_collocation=n_collocation,
            verbose_freq=max(1, n_epochs // 10),
            diag_grid_n=200,
            handover_strategy=handover_strategy,
            handover_max_adam_epochs=handover_max_adam_epochs,
            plateau_patience=plateau_patience,
            plateau_min_delta=plateau_min_delta,
            patience=patience,
            min_delta=min_delta,
            moving_avg_window=moving_avg_window,
        )
        RR, ZZ, t = pinn._eval_grid(150)
        with torch.no_grad():
            psi_pred = pinn._psi_hat(t).cpu().numpy().reshape(150, 150)
        out.append(
            SeedResult(
                seed=seed,
                J_val=np.asarray(pinn.J_val, dtype=np.float64),
                sol_l2=np.asarray(pinn.sol_l2, dtype=np.float64),
                final_J_val=float(pinn.J_val[-1]),
                final_sol_l2=float(pinn.sol_l2[-1]) if pinn.sol_l2 else float("nan"),
                final_sol_rel_l2=float(pinn.sol_rel_l2[-1]) if pinn.sol_rel_l2 else float("nan"),
                field_pred=psi_pred,
            )
        )
    return tuple(out)


def _pad_and_stack(seq: list[np.ndarray]) -> np.ndarray:
    if not seq:
        return np.empty((0, 0), dtype=np.float64)
    max_len = max(len(a) for a in seq)
    out = np.full((len(seq), max_len), np.nan, dtype=np.float64)
    for i, a in enumerate(seq):
        if len(a) == 0:
            continue
        out[i, : len(a)] = a
        out[i, len(a):] = a[-1]
    return out


def plot_results(
    results: tuple[SeedResult, ...],
    out_path: str,
    n: int = 150,
) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    rs = np.linspace(R_LO, R_HI, n)
    zs = np.linspace(Z_LO, Z_HI, n)
    RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
    psi_true = psi_exact_np(RR, ZZ)
    psi_pred = results[-1].field_pred
    err = np.abs(psi_pred - psi_true)

    extent = (R_LO, R_HI, Z_LO, Z_HI)
    im0 = ax[0, 0].imshow(psi_true.T, origin="lower", extent=extent,
                          cmap="viridis", aspect="auto", interpolation="nearest")
    ax[0, 0].set_title(r"$\psi_{\mathrm{exact}}(R, Z) = R^2$")
    fig.colorbar(im0, ax=ax[0, 0], shrink=0.8)

    im1 = ax[0, 1].imshow(psi_pred.T, origin="lower", extent=extent,
                          cmap="viridis", aspect="auto", interpolation="nearest")
    ax[0, 1].set_title(r"$\widehat{\psi}_\theta(R, Z)$ (last seed)")
    fig.colorbar(im1, ax=ax[0, 1], shrink=0.8)

    # Faithful per-pixel render of the noise-floor error (near-Nyquist in R):
    # 'nearest' avoids antialiasing moire, and the log range is clipped to the
    # top two decades so the map reads as amplitude, not zero-crossing spikes.
    err_vmax = float(err.max()) + 1e-14
    im2 = ax[1, 0].imshow(
        err.T, origin="lower", extent=extent, cmap="inferno", aspect="auto",
        interpolation="nearest",
        norm=matplotlib.colors.LogNorm(vmin=err_vmax / 100.0, vmax=err_vmax),
    )
    ax[1, 0].set_title(r"$|\widehat{\psi}_\theta - \psi_{\mathrm{exact}}|$ (log scale)")
    ax[1, 0].set_xlabel("R")
    ax[1, 0].set_ylabel("Z")
    fig.colorbar(im2, ax=ax[1, 0], shrink=0.8)

    # Convergence: solution L2 error and residual L2 error over epochs. The
    # residual L2 norm is sqrt(AREA * J_val); J_val is the mean squared residual
    # over the box, so sqrt(area * mean(r^2)) estimates ||Delta*_psi||_{L2}.
    sol_seeds = [np.asarray(r.sol_l2, dtype=np.float64) for r in results]
    res_seeds = [np.sqrt(AREA * np.asarray(r.J_val, dtype=np.float64)) for r in results]
    for sol, res in zip(sol_seeds, res_seeds):
        ax[1, 1].semilogy(sol, color="C0", alpha=0.25)
        ax[1, 1].semilogy(res, color="C1", alpha=0.25)
    sol_H = _pad_and_stack(sol_seeds)
    res_H = _pad_and_stack(res_seeds)
    ax[1, 1].semilogy(
        np.nanmean(sol_H, axis=0), color="C0", linewidth=1.8,
        label=r"solution $\|\widehat{\psi}_\theta - \psi_{\mathrm{exact}}\|_{L^2}$",
    )
    ax[1, 1].semilogy(
        np.nanmean(res_H, axis=0), color="C1", linewidth=1.8,
        label=r"residual $\|\Delta^\ast\widehat{\psi}_\theta\|_{L^2}$",
    )
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel(r"$L^2$ error")
    ax[1, 1].set_title(r"Solution and residual $L^2$ error")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


def write_summary(
    results: tuple[SeedResult, ...],
    out_path: str,
    qn_variant: str,
    n_epochs: int,
    adam_epochs: int,
    seeds: tuple[int, ...],
) -> None:
    means = {
        "final_J_val_mean": float(np.mean([r.final_J_val for r in results])),
        "final_J_val_std":  float(np.std([r.final_J_val for r in results])),
        "final_sol_l2_mean": float(np.mean([r.final_sol_l2 for r in results])),
        "final_sol_l2_std":  float(np.std([r.final_sol_l2 for r in results])),
        "final_rel_l2_mean": float(np.mean([r.final_sol_rel_l2 for r in results])),
        "final_rel_l2_std":  float(np.std([r.final_sol_rel_l2 for r in results])),
    }
    payload = {
        "problem": f"Vacuum Grad-Shafranov psi=R^2 on [{R_LO},{R_HI}]x[{Z_LO},{Z_HI}]",
        "qn_variant": qn_variant,
        "n_epochs": n_epochs,
        "adam_epochs": adam_epochs,
        "seeds": list(seeds),
        "means": means,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vacuum Grad-Shafranov PINN validation (multi-seed).")
    p.add_argument("--qn-variant", type=str, default="ssbroyden",
                   choices=["bfgs", "ssbfgs", "ssbroyden"])
    p.add_argument("--loss-transform", type=str, default="identity",
                   choices=["identity", "sqrt", "log", "boxcox"])
    p.add_argument("--loss-lambda", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=10000,
                   help="Total budget cap (2000 Adam + up to 8000 QN); QN-phase "
                        "early stopping ends most runs well before this.")
    p.add_argument("--adam-epochs", type=int, default=2000,
                   help="Standardised fixed Adam warm-up before handover.")
    p.add_argument("--n-collocation", type=int, default=2000)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32, 32])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--handover-strategy", type=str, default="fixed",
                   choices=["fixed", "plateau", "loss_threshold", "gradnorm"])
    p.add_argument("--handover-max-adam-epochs", type=int, default=10000)
    p.add_argument("--plateau-patience", type=int, default=200)
    p.add_argument("--plateau-min-delta", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=500)
    p.add_argument("--min-delta", type=float, default=1e-12)
    p.add_argument("--moving-avg-window", type=int, default=20)
    p.add_argument("--results-dir", type=str, default=os.path.join("..", "results"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.results_dir,
        f"vacuum_gs_{args.qn_variant}_{args.loss_transform}_{run_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    results = run_seeds(
        seeds=tuple(args.seeds),
        n_epochs=args.epochs,
        adam_epochs=args.adam_epochs,
        n_collocation=args.n_collocation,
        hidden=tuple(args.hidden),
        lr=args.lr,
        qn_variant=args.qn_variant,
        loss_transform=args.loss_transform,
        loss_lambda=args.loss_lambda,
        handover_strategy=args.handover_strategy,
        handover_max_adam_epochs=args.handover_max_adam_epochs,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        patience=args.patience,
        min_delta=args.min_delta,
        moving_avg_window=args.moving_avg_window,
    )

    plot_results(results, out_path=os.path.join(out_dir, "vacuum_gs_results.png"))
    write_summary(
        results=results,
        out_path=os.path.join(out_dir, "summary.json"),
        qn_variant=args.qn_variant,
        n_epochs=args.epochs,
        adam_epochs=args.adam_epochs,
        seeds=tuple(args.seeds),
    )

    np.savez(
        os.path.join(out_dir, "raw_histories.npz"),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        **{f"J_val_seed{r.seed}": r.J_val for r in results},
        **{f"sol_l2_seed{r.seed}": r.sol_l2 for r in results},
        **{f"field_seed{r.seed}": r.field_pred for r in results},
    )
    print(f"All artefacts written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
