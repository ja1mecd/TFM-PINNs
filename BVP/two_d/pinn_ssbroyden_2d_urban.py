"""2D CFGS PINN — Urban-style self-scaled quasi-Newton refinement.

Companion to ``pinn_ssbroyden_2d.py`` that ports the missing ingredients of
Jorge Urbán's reference implementation
(https://github.com/jorgeurban/self_scaled_algorithms_pinns) into the
PyTorch CFGS pipeline so the two scripts can be compared directly:

1. Strong Wolfe line search (Armijo + curvature) replaces pure Armijo
   backtracking — see ``BVP/optimizers/ssbroyden_urban.py``.
2. ``loss_transform`` knob applies the QN phase to ``log(J + eps)``,
   ``sqrt(J + eps)`` or the Box-Cox family — Adam still optimises raw J.
3. ``rad_resample_every`` triggers residual-adaptive distribution
   sampling on the (q, mu) rectangle every K iterations of the QN phase,
   warm-starting the inverse Hessian after a Cholesky positive-definite
   check (matching Urbán's ``cholesky(H0)`` block in ``AC.py``).
4. ``initial_scale`` enables the H == I scaling branch from Urban.
5. All seven variants (BFGS, BFGS_scipy, SSBFGS_OL, SSBFGS_AB,
   SSBroyden1/2/3) are selectable.

The CFGS problem definition (analytic solution, hard-Dirichlet ansatz,
Grad-Shafranov operator) is imported verbatim from
``pinn_ssbroyden_2d.py`` so any final difference in results is
attributable to the QN changes alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_OPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "optimizers"
)
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)
from ssbroyden_urban import SSBroydenUrbanOptimizer  # noqa: E402

# Re-use problem definition from the existing 2D script so the only
# difference between the two pipelines is the QN phase.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_ssbroyden_2d import (  # noqa: E402
    B_COEFFS,
    NeuralNetwork,
    P_exact,
    f_b,
    h_b,
    legendre_derivatives,  # noqa: F401  (kept for parity / external imports)
    mu_max,
    mu_min,
    q_max,
    q_min,
)


# =============================================================================
# DEVICE
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    # Match Urbán's float64 numerics; disable TF32 since it would silently
    # downcast matmuls used inside the dense Hessian update.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


# =============================================================================
# Loss transformation g(J) used during the QN phase
# =============================================================================
def transform_loss(J_raw: torch.Tensor, kind: str, lam: float, eps: float) -> torch.Tensor:
    if kind == "identity":
        return J_raw
    if kind == "sqrt":
        return torch.sqrt(J_raw + eps)
    if kind == "log":
        return torch.log(J_raw + eps)
    if kind == "boxcox":
        shifted = J_raw + eps
        if lam == 0.0:
            return torch.log(shifted)
        return torch.expm1(lam * torch.log(shifted)) / lam
    raise ValueError(f"Unknown loss_transform={kind!r}")


# =============================================================================
# PINN solver
# =============================================================================
class PINN_CFGS_Solver_Urban:
    def __init__(
        self,
        model: nn.Module,
        *,
        lr_adam: float = 1e-3,
        lambda_pde: float = 1.0,
        b_coeffs: tuple[float, ...] = B_COEFFS,
        variant: str = "SSBroyden2",
        loss_transform: str = "identity",
        loss_lambda: float = 0.5,
        loss_eps: float = 1e-12,
        rel_err_eps: float = 1e-12,
        wolfe_c1: float = 1e-4,
        wolfe_c2: float = 0.9,
        wolfe_max_ls: int = 25,
        initial_scale: bool = False,
        H_dtype: torch.dtype = torch.float64,
        H_on_cpu: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.lambda_pde = float(lambda_pde)
        self.b_coeffs = tuple(b_coeffs)
        self.loss_transform = loss_transform
        self.loss_lambda = float(loss_lambda)
        self.loss_eps = float(loss_eps)
        self.rel_err_eps = float(rel_err_eps)

        self.adam = optim.Adam(self.model.parameters(), lr=lr_adam)
        self.qn = SSBroydenUrbanOptimizer(
            self.model.parameters(),
            variant=variant,
            c1=wolfe_c1,
            c2=wolfe_c2,
            max_ls=wolfe_max_ls,
            initial_scale=initial_scale,
            tau_min=1e-12,
            tau_max=None,
            damping=1e-30,
            dtype=H_dtype,
            H_device="cpu" if H_on_cpu else None,
        )

        # Logs
        self.obj_train: list[float] = []
        self.obj_val: list[float] = []
        self.J_train: list[float] = []
        self.J_val: list[float] = []
        self.pde_l2: list[float] = []
        self.sol_l2: list[float] = []
        self.sol_rel_l2: list[float] = []

    # ----- hard Dirichlet ansatz / Grad-Shafranov operator -----
    def _P_hat(self, qmu: torch.Tensor) -> torch.Tensor:
        return f_b(qmu, self.b_coeffs) + h_b(qmu) * self.model(qmu)

    def _delta_gs(self, qmu: torch.Tensor, create_graph_second: bool) -> torch.Tensor:
        qmu = qmu.to(device)
        if not qmu.requires_grad:
            qmu = qmu.requires_grad_(True)
        P = self._P_hat(qmu)
        grads = torch.autograd.grad(
            P, qmu, grad_outputs=torch.ones_like(P), create_graph=True
        )[0]
        P_q = grads[:, 0:1]
        P_mu = grads[:, 1:2]
        P_qq = torch.autograd.grad(
            P_q, qmu, grad_outputs=torch.ones_like(P_q),
            create_graph=create_graph_second, retain_graph=True,
        )[0][:, 0:1]
        P_mumu = torch.autograd.grad(
            P_mu, qmu, grad_outputs=torch.ones_like(P_mu),
            create_graph=create_graph_second,
        )[0][:, 1:2]
        q = qmu[:, 0:1]
        mu = qmu[:, 1:2]
        return q**4 * P_qq + 2.0 * q**3 * P_q + q**2 * (1.0 - mu**2) * P_mumu

    def compute_loss(self, qmu_interior: torch.Tensor, *, transform: bool):
        qmu = qmu_interior.detach().clone().requires_grad_(True)
        res = self._delta_gs(qmu, create_graph_second=True)
        area = (q_max - q_min) * (mu_max - mu_min)
        J_raw = self.lambda_pde * (torch.mean(res**2) * area)
        if transform:
            J_obj = transform_loss(
                J_raw, self.loss_transform, self.loss_lambda, self.loss_eps
            )
        else:
            J_obj = J_raw
        return J_obj, J_raw

    # ----- diagnostics on a uniform (q, mu) grid -----
    def _grid(self, n: int):
        qs = np.linspace(q_min, q_max, n)
        mus = np.linspace(mu_min, mu_max, n)
        QQ, MM = np.meshgrid(qs, mus, indexing="xy")
        QM = np.stack([QQ.ravel(), MM.ravel()], axis=1).astype(np.float32)
        return qs, mus, QQ, MM, QM

    def compute_pde_l2(self, n: int = 60) -> float:
        qs, mus, _, _, QM = self._grid(n)
        QMt = torch.from_numpy(QM).to(device)
        res = (
            self._delta_gs(QMt, create_graph_second=False)
            .detach()
            .cpu()
            .numpy()
            .reshape(n, n)
        )
        intMu = np.trapz(res**2, mus, axis=0)
        return float(np.sqrt(np.trapz(intMu, qs, axis=0)))

    def compute_sol_l2(self, n: int = 60) -> float:
        qs, mus, _, _, QM = self._grid(n)
        QMt = torch.from_numpy(QM).to(device)
        u_true = P_exact(QMt).detach().cpu().numpy().reshape(n, n)
        with torch.no_grad():
            u_pred = self._P_hat(QMt).cpu().numpy().reshape(n, n)
        diff = u_pred - u_true
        intMu = np.trapz(diff**2, mus, axis=0)
        return float(np.sqrt(np.trapz(intMu, qs, axis=0)))

    def compute_sol_rel_l2(self, n: int = 60) -> float:
        qs, mus, _, _, QM = self._grid(n)
        QMt = torch.from_numpy(QM).to(device)
        u_true = P_exact(QMt).detach().cpu().numpy().reshape(n, n)
        with torch.no_grad():
            u_pred = self._P_hat(QMt).cpu().numpy().reshape(n, n)
        diff = u_pred - u_true
        num = np.trapz(np.trapz(diff**2, mus, axis=0), qs, axis=0)
        den = np.trapz(np.trapz(u_true**2, mus, axis=0), qs, axis=0)
        return float(np.sqrt(num) / (np.sqrt(den) + self.rel_err_eps))

    # ----- residual-adaptive sampling on the (q, mu) rectangle -----
    @torch.no_grad()
    def _rad_resample(
        self, n_pool: int, n_keep: int, k1: float = 1.0, k2: float = 1.0
    ) -> torch.Tensor:
        q_pool = torch.empty(n_pool, 1, device=device).uniform_(q_min, q_max)
        mu_pool = torch.empty(n_pool, 1, device=device).uniform_(mu_min, mu_max)
        qmu_pool = torch.cat([q_pool, mu_pool], dim=1)
        with torch.enable_grad():
            r = self._delta_gs(qmu_pool, create_graph_second=False).detach().abs()
        r = r.flatten()
        if k1 != 1.0:
            r = r ** k1
        weight = r / (r.mean() + 1e-30) + k2
        weight = weight / weight.sum()
        idx = torch.multinomial(weight, num_samples=n_keep, replacement=False)
        return qmu_pool[idx]

    # ----- per-step helpers -----
    def _adam_step(self, qmu_train: torch.Tensor) -> tuple[float, float]:
        self.adam.zero_grad()
        J_obj, J_raw = self.compute_loss(qmu_train, transform=False)
        J_obj.backward()
        self.adam.step()
        return float(J_obj.item()), float(J_raw.item())

    def _qn_step(self, qmu_train: torch.Tensor) -> tuple[float, float, float]:
        latest_raw = {"value": float("nan")}

        def loss_and_grad(x_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                offset = 0
                for p in self.model.parameters():
                    n = p.numel()
                    p.copy_(x_vec[offset : offset + n].to(p.dtype).view_as(p))
                    offset += n
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            J_obj, J_raw = self.compute_loss(qmu_train, transform=True)
            J_obj.backward()
            grads = torch.cat(
                [
                    (p.grad if p.grad is not None else torch.zeros_like(p)).detach().view(-1)
                    for p in self.model.parameters()
                ]
            )
            latest_raw["value"] = float(J_raw.item())
            return J_obj.detach(), grads

        J_obj_t = self.qn.step(loss_and_grad)
        return float(J_obj_t.item()), latest_raw["value"], self.qn._last_alpha

    # ----- training loop -----
    def train(
        self,
        *,
        n_epochs: int = 5000,
        adam_epochs: int = 2000,
        n_collocation: int = 1000,
        train_split: float = 0.8,
        rad_resample_every: int = 500,
        rad_pool_size: int = 10000,
        rad_k1: float = 1.0,
        rad_k2: float = 1.0,
        verbose_freq: int = 200,
        diag_grid_n: int = 60,
        seed: int = 5,
    ) -> None:
        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be in (0, 1).")
        if rad_resample_every < 1:
            raise ValueError("rad_resample_every must be >= 1.")
        if adam_epochs < 0 or adam_epochs > n_epochs:
            raise ValueError("0 <= adam_epochs <= n_epochs required.")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        n_train = max(1, min(int(n_collocation * train_split), n_collocation - 1))

        def fresh_uniform() -> tuple[torch.Tensor, torch.Tensor]:
            q = torch.empty(n_collocation, 1, device=device).uniform_(q_min, q_max)
            mu = torch.empty(n_collocation, 1, device=device).uniform_(mu_min, mu_max)
            qmu = torch.cat([q, mu], dim=1)
            perm = torch.randperm(n_collocation, device=device)
            qmu = qmu[perm]
            return qmu[:n_train].detach().clone(), qmu[n_train:].detach().clone()

        qmu_train, qmu_val = fresh_uniform()

        # ---- Adam warmup (uniform sampling) ----
        for epoch in range(1, adam_epochs + 1):
            if epoch != 1 and ((epoch - 1) % rad_resample_every == 0):
                qmu_train, qmu_val = fresh_uniform()
            J_obj, J_raw = self._adam_step(qmu_train)
            self.obj_train.append(J_obj)
            self.J_train.append(J_raw)

            with torch.enable_grad():
                _, J_val_raw = self.compute_loss(qmu_val, transform=False)
            self.J_val.append(float(J_val_raw.item()))
            self.obj_val.append(float(J_val_raw.item()))

            self.pde_l2.append(self.compute_pde_l2(n=diag_grid_n))
            self.sol_l2.append(self.compute_sol_l2(n=diag_grid_n))
            self.sol_rel_l2.append(self.compute_sol_rel_l2(n=diag_grid_n))

            if epoch % verbose_freq == 0:
                print(
                    f"[ADAM]      epoch {epoch:6d} | "
                    f"J_raw={J_raw:.4e} J_val={float(J_val_raw.item()):.4e} | "
                    f"PDE_L2={self.pde_l2[-1]:.4e} SOL_L2={self.sol_l2[-1]:.4e} "
                    f"REL={self.sol_rel_l2[-1]:.3e}"
                )

        # Switch to RAD for the QN phase.
        qmu_train = self._rad_resample(rad_pool_size, n_train, rad_k1, rad_k2)
        qmu_val = self._rad_resample(rad_pool_size, n_collocation - n_train, rad_k1, rad_k2)

        # ---- Quasi-Newton with periodic RAD resample + Cholesky-checked warm
        #      start of the inverse Hessian. ----
        qn_total = n_epochs - adam_epochs
        for epoch in range(adam_epochs + 1, n_epochs + 1):
            qn_iter = epoch - adam_epochs
            if qn_iter != 1 and ((qn_iter - 1) % rad_resample_every == 0):
                H = self.qn.H
                if H is not None:
                    self.qn.warm_start_from(H, cholesky_check=True)
                qmu_train = self._rad_resample(rad_pool_size, n_train, rad_k1, rad_k2)
                qmu_val = self._rad_resample(
                    rad_pool_size, n_collocation - n_train, rad_k1, rad_k2
                )

            J_obj, J_raw, alpha = self._qn_step(qmu_train)
            self.obj_train.append(J_obj)
            self.J_train.append(J_raw)

            with torch.enable_grad():
                J_val_obj, J_val_raw = self.compute_loss(qmu_val, transform=True)
            self.J_val.append(float(J_val_raw.item()))
            self.obj_val.append(float(J_val_obj.item()))

            self.pde_l2.append(self.compute_pde_l2(n=diag_grid_n))
            self.sol_l2.append(self.compute_sol_l2(n=diag_grid_n))
            self.sol_rel_l2.append(self.compute_sol_rel_l2(n=diag_grid_n))

            if epoch % verbose_freq == 0 or epoch == n_epochs:
                d = self.qn.diagnostics()
                print(
                    f"[QN/{self.qn.param_groups[0]['variant']}] "
                    f"epoch {epoch:6d} ({qn_iter}/{qn_total}) | "
                    f"J_raw={J_raw:.4e} J_val={float(J_val_raw.item()):.4e} | "
                    f"alpha={alpha:.2e} tau={d['last_tau']:.2e} | "
                    f"PDE_L2={self.pde_l2[-1]:.4e} SOL_L2={self.sol_l2[-1]:.4e} "
                    f"REL={self.sol_rel_l2[-1]:.3e} | "
                    f"resets={d['n_resets']} ls_fail={d['n_ls_failures']}"
                )

        print("-" * 60)
        d = self.qn.diagnostics()
        print(
            f"Final: J_raw={self.J_train[-1]:.4e}, J_val={self.J_val[-1]:.4e}, "
            f"PDE_L2={self.pde_l2[-1]:.4e}, SOL_L2={self.sol_l2[-1]:.4e}, "
            f"REL={self.sol_rel_l2[-1]:.3e}"
        )
        print(f"QN diagnostics: {d}")

    # ----- post-processing -----
    def plot_results(self, n: int = 80, save_path: str | None = None, dpi: int = 150) -> None:
        qs, mus, _, _, QM = self._grid(n)
        QMt = torch.from_numpy(QM).to(device)
        u_true = P_exact(QMt).detach().cpu().numpy().reshape(n, n)
        with torch.no_grad():
            u_pred = self._P_hat(QMt).cpu().numpy().reshape(n, n)
        abs_err = np.abs(u_pred - u_true)
        rel_err = abs_err / (np.abs(u_true) + self.rel_err_eps)

        fig, ax = plt.subplots(2, 3, figsize=(16, 9))

        im0 = ax[0, 0].imshow(
            u_true, origin="lower", extent=[q_min, q_max, mu_min, mu_max], aspect="auto"
        )
        ax[0, 0].set_title(r"$P_{\mathrm{exact}}(q, \mu)$")
        plt.colorbar(im0, ax=ax[0, 0], fraction=0.046)

        im1 = ax[0, 1].imshow(
            u_pred, origin="lower", extent=[q_min, q_max, mu_min, mu_max], aspect="auto"
        )
        ax[0, 1].set_title(r"$P_{\mathrm{PINN}}(q, \mu)$")
        plt.colorbar(im1, ax=ax[0, 1], fraction=0.046)

        im2 = ax[0, 2].imshow(
            abs_err, origin="lower",
            extent=[q_min, q_max, mu_min, mu_max], aspect="auto",
        )
        ax[0, 2].set_title(r"$|P_{\mathrm{PINN}} - P_{\mathrm{exact}}|$")
        plt.colorbar(im2, ax=ax[0, 2], fraction=0.046)

        rel_vmax = float(np.percentile(rel_err, 99))
        if not np.isfinite(rel_vmax) or rel_vmax <= 0.0:
            rel_vmax = float(np.nanmax(rel_err)) if np.isfinite(np.nanmax(rel_err)) else 1.0
        im3 = ax[1, 0].imshow(
            rel_err, origin="lower",
            extent=[q_min, q_max, mu_min, mu_max], aspect="auto",
            vmin=0.0, vmax=rel_vmax,
        )
        ax[1, 0].set_title(
            r"$|P_{\mathrm{PINN}} - P_{\mathrm{exact}}|/(|P_{\mathrm{exact}}| + \varepsilon)$"
            f"  (clipped at p99 = {rel_vmax:.2e})"
        )
        plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, extend="max")

        ax[1, 1].semilogy(np.maximum(self.J_train, 1e-300), label="J(train)")
        ax[1, 1].semilogy(np.maximum(self.J_val, 1e-300), label="J(val)")
        if self.loss_transform != "identity":
            ax[1, 1].semilogy(
                np.maximum(self.obj_train, 1e-300), "--",
                label=f"g(J) train [{self.loss_transform}]",
            )
        ax[1, 1].set_xlabel("iteration")
        ax[1, 1].set_title("Loss curves")
        ax[1, 1].grid(True, alpha=0.3)
        ax[1, 1].legend()

        ax[1, 2].semilogy(self.pde_l2, label=r"$\|res\|_{L^2}$")
        ax[1, 2].semilogy(self.sol_l2, label=r"$\|P_{\mathrm{NN}} - P_{\mathrm{exact}}\|_{L^2}$")
        ax[1, 2].semilogy(self.sol_rel_l2, label=r"rel $L^2$")
        ax[1, 2].set_xlabel("iteration")
        ax[1, 2].set_title(r"$L^2$ errors")
        ax[1, 2].grid(True, alpha=0.3)
        ax[1, 2].legend()

        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(os.path.dirname(__file__), "results.png")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {save_path}")

    def save_results(self, run_dir: str, n_eval: int = 80, extra_metadata: dict | None = None) -> None:
        os.makedirs(run_dir, exist_ok=True)
        np.savez(
            os.path.join(run_dir, "logs.npz"),
            obj_train=np.asarray(self.obj_train, dtype=np.float64),
            obj_val=np.asarray(self.obj_val, dtype=np.float64),
            J_train=np.asarray(self.J_train, dtype=np.float64),
            J_val=np.asarray(self.J_val, dtype=np.float64),
            pde_l2=np.asarray(self.pde_l2, dtype=np.float64),
            sol_l2=np.asarray(self.sol_l2, dtype=np.float64),
            sol_rel_l2=np.asarray(self.sol_rel_l2, dtype=np.float64),
        )
        meta = {
            "loss_transform": self.loss_transform,
            "loss_lambda": self.loss_lambda,
            "variant": self.qn.param_groups[0]["variant"],
            "qn_diagnostics": self.qn.diagnostics(),
            "final_J_train": (self.J_train[-1] if self.J_train else None),
            "final_J_val": (self.J_val[-1] if self.J_val else None),
            "final_pde_l2": (self.pde_l2[-1] if self.pde_l2 else None),
            "final_sol_l2": (self.sol_l2[-1] if self.sol_l2 else None),
            "final_sol_rel_l2": (self.sol_rel_l2[-1] if self.sol_rel_l2 else None),
        }
        if extra_metadata:
            meta.update(extra_metadata)
        with open(os.path.join(run_dir, "metadata.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

    def save_model(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        default="SSBroyden2",
        choices=[
            "BFGS", "BFGS_scipy", "SSBFGS_OL", "SSBFGS_AB",
            "SSBroyden1", "SSBroyden2", "SSBroyden3",
        ],
    )
    p.add_argument(
        "--loss-transform", default="identity",
        choices=["identity", "sqrt", "log", "boxcox"],
    )
    p.add_argument("--loss-lambda", type=float, default=0.5)
    p.add_argument("--initial-scale", action="store_true")
    p.add_argument("--n-epochs", type=int, default=5000)
    p.add_argument("--adam-epochs", type=int, default=2000)
    p.add_argument("--n-collocation", type=int, default=1000)
    p.add_argument("--rad-resample-every", type=int, default=500)
    p.add_argument("--rad-pool-size", type=int, default=10000)
    p.add_argument("--rad-k1", type=float, default=1.0)
    p.add_argument("--rad-k2", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--diag-grid-n", type=int, default=60)
    p.add_argument("--H-on-cpu", action="store_true")
    p.add_argument("--run-tag", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = NeuralNetwork(hidden_layers=(30,), activation=nn.Tanh())
    print("\nNeural Network Architecture:\n", model, "\n")
    print(
        f"Variant: {args.variant} | loss_transform: {args.loss_transform} "
        f"(lambda={args.loss_lambda}) | initial_scale: {args.initial_scale}"
    )

    pinn = PINN_CFGS_Solver_Urban(
        model,
        variant=args.variant,
        loss_transform=args.loss_transform,
        loss_lambda=args.loss_lambda,
        initial_scale=args.initial_scale,
        H_on_cpu=args.H_on_cpu,
    )
    pinn.train(
        n_epochs=args.n_epochs,
        adam_epochs=args.adam_epochs,
        n_collocation=args.n_collocation,
        rad_resample_every=args.rad_resample_every,
        rad_pool_size=args.rad_pool_size,
        rad_k1=args.rad_k1,
        rad_k2=args.rad_k2,
        seed=args.seed,
        diag_grid_n=args.diag_grid_n,
    )

    run_tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")
    transform_tag = (
        f"boxcox_lam{args.loss_lambda:g}"
        if args.loss_transform == "boxcox"
        else args.loss_transform
    )
    run_name = f"cfgs_urban_{args.variant}_{transform_tag}_{run_tag}"
    run_dir = os.path.join("..", "results", run_name)
    os.makedirs(run_dir, exist_ok=True)

    pinn.plot_results(n=80, save_path=os.path.join(run_dir, "results.png"))
    pinn.save_model(f"../models/pinn_cfgs_urban_{args.variant}_{transform_tag}.pth")
    pinn.save_results(
        run_dir, n_eval=80,
        extra_metadata={
            "run_name": run_name, "run_tag": run_tag,
            "argv": vars(args),
        },
    )
    print(f"\nAll run artefacts written to: {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
