"""1D PINN BVP solver — Urban-style self-scaled quasi-Newton refinement.

Companion to ``pinn_bvpsolver_l2_SSBroyden.py`` that ports the missing
ingredients of Jorge Urbán's reference implementation
(https://github.com/jorgeurban/self_scaled_algorithms_pinns) into the
PyTorch pipeline so the two scripts can be compared directly:

1. Strong Wolfe line search (Armijo + curvature condition) replaces pure
   Armijo backtracking — see ``BVP/optimizers/ssbroyden_urban.py``.
2. ``loss_transform`` knob applies the QN phase to ``log(J + eps)``,
   ``sqrt(J + eps)``, or the Box-Cox family — Adam still optimises raw J.
3. ``rad_resample_every`` triggers residual-adaptive distribution
   sampling (RAD; Wu et al., 2023) every K iterations of the QN phase,
   warm-starting the inverse Hessian after a Cholesky positive-definite
   check (matching Urbán's ``cholesky(H0)`` block in ``AC.py``).
4. ``initial_scale`` enables the H == I scaling branch from Urban.
5. All seven variants (BFGS, BFGS_scipy, SSBFGS_OL, SSBFGS_AB,
   SSBroyden1/2/3) are selectable.

The problem, hard Dirichlet ansatz, network architecture and Adam warmup
are intentionally identical to the existing scripts so that any final
``J``/``L^2`` difference is attributable to the QN phase changes alone.

Output figure: ``figures/pinn_bvpsolver_l2_SSBroyden_urban.png``.
"""

from __future__ import annotations

import argparse
import os
import sys

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

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


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
# PROBLEM: -u''(x) = (k pi)^2 sin(k pi x) on [0, 1] — same as the BFGS script
# =============================================================================
interval = [0.0, 1.0]
a, b = interval

K_WAVENUMBER = 4.0


def f(x):
    if isinstance(x, torch.Tensor):
        return -((K_WAVENUMBER * np.pi) ** 2) * torch.sin(K_WAVENUMBER * np.pi * x)
    return -((K_WAVENUMBER * np.pi) ** 2) * np.sin(K_WAVENUMBER * np.pi * x)


def u_exact(x):
    if isinstance(x, torch.Tensor):
        return torch.sin(K_WAVENUMBER * np.pi * x)
    return np.sin(K_WAVENUMBER * np.pi * x)


alpha_bc, beta_bc = u_exact(a), u_exact(b)


# =============================================================================
# Network
# =============================================================================
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers=(32, 32, 32), activation: nn.Module | None = None):
        super().__init__()
        activation = activation if activation is not None else nn.Tanh()
        layers: list[nn.Module] = []
        in_dim = 1
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Loss transformation g(J) used during the QN phase
# =============================================================================
def transform_loss(J_raw: torch.Tensor, kind: str, lam: float, eps: float) -> torch.Tensor:
    """Apply the QN-phase loss transform ``g``. Adam always sees ``identity``."""
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
        return torch.exp(lam * torch.log(shifted)) / lam
    raise ValueError(f"Unknown loss_transform={kind!r}")


# =============================================================================
# PINN solver
# =============================================================================
class PINN_BVP_Solver_Urban:
    def __init__(
        self,
        model: nn.Module,
        *,
        lr_adam: float = 1e-3,
        lambda_pde: float = 1.0,
        variant: str = "SSBroyden2",
        loss_transform: str = "identity",
        loss_lambda: float = 0.5,
        loss_eps: float = 1e-12,
        wolfe_c1: float = 1e-4,
        wolfe_c2: float = 0.9,
        wolfe_max_ls: int = 25,
        initial_scale: bool = False,
        H_dtype: torch.dtype = torch.float64,
        H_on_cpu: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.lambda_pde = float(lambda_pde)
        self.loss_transform = loss_transform
        self.loss_lambda = float(loss_lambda)
        self.loss_eps = float(loss_eps)

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

        # Logs (same shape as the existing 1D script for plot parity).
        self.losses: list[float] = []  # raw J on the training points
        self.obj_losses: list[float] = []  # transformed objective on training
        self.val_losses: list[float] = []  # raw J on the val points
        self.pde_l2_errors: list[float] = []
        self.solution_l2_errors: list[float] = []

    # ----- hard Dirichlet ansatz (same as existing scripts) -----
    def _u_hat(self, x: torch.Tensor) -> torch.Tensor:
        base = alpha_bc * (b - x) / (b - a) + beta_bc * (x - a) / (b - a)
        return base + (x - a) * (b - x) * self.model(x)

    # ----- PDE residual (autograd) -----
    def _pde_residual(self, x_interior: torch.Tensor) -> torch.Tensor:
        x_interior = x_interior.to(device).detach().clone().requires_grad_(True)
        u = self._u_hat(x_interior)
        u_x = torch.autograd.grad(
            u, x_interior, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_interior, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        return u_xx - f(x_interior)

    def compute_loss(self, x_interior: torch.Tensor, *, transform: bool):
        r = self._pde_residual(x_interior)
        J_raw = self.lambda_pde * torch.mean(r**2) * (b - a)
        if transform:
            J_obj = transform_loss(
                J_raw, self.loss_transform, self.loss_lambda, self.loss_eps
            )
        else:
            J_obj = J_raw
        return J_obj, J_raw

    # ----- diagnostics -----
    def compute_pde_l2_norm(self, n_points: int = 500) -> float:
        x_test = np.linspace(a, b, n_points)
        x_torch = torch.from_numpy(x_test.reshape(-1, 1)).float().to(device)
        x_torch.requires_grad_(True)
        u = self._u_hat(x_torch)
        u_x = torch.autograd.grad(
            u, x_torch, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_torch, grad_outputs=torch.ones_like(u_x), create_graph=False
        )[0]
        u_xx_np = u_xx.detach().cpu().numpy().flatten()
        residual = u_xx_np - f(x_test)
        return float(np.sqrt(np.trapz(residual**2, x_test)))

    def compute_solution_l2_norm(self, n_points: int = 500) -> float:
        x_test = np.linspace(a, b, n_points)
        x_torch = torch.from_numpy(x_test.reshape(-1, 1)).float().to(device)
        with torch.no_grad():
            u_pred = self._u_hat(x_torch).cpu().numpy().flatten()
        diff = u_pred - u_exact(x_test)
        return float(np.sqrt(np.trapz(diff**2, x_test)))

    # ----- residual-adaptive sampling (RAD) -----
    @torch.no_grad()
    def _rad_resample(
        self,
        n_pool: int,
        n_keep: int,
        k1: float = 1.0,
        k2: float = 1.0,
    ) -> torch.Tensor:
        """Sample ``n_keep`` interior points with probability ~|res|^k1 / mean + k2.

        This matches Wu et al. (2023) "RAD" / Urbán's ``adaptive_rad`` helper.
        ``k2`` adds a uniform floor (set to 0 for pure residual weighting,
        to 1 for the paper's "RAR" mix).
        """
        x_pool = torch.empty(n_pool, 1, device=device).uniform_(a, b)
        # Compute |residual| at the pool points.
        with torch.enable_grad():
            r = self._pde_residual(x_pool).detach().abs()
        r = r.flatten()
        if k1 != 1.0:
            r = r ** k1
        weight = r / (r.mean() + 1e-30) + k2
        weight = weight / weight.sum()
        idx = torch.multinomial(weight, num_samples=n_keep, replacement=False)
        return x_pool[idx]

    # ----- Adam warmup step -----
    def _adam_step(self, x_train: torch.Tensor) -> tuple[float, float]:
        self.adam.zero_grad()
        J_obj, J_raw = self.compute_loss(x_train, transform=False)
        J_obj.backward()
        self.adam.step()
        return float(J_obj.item()), float(J_raw.item())

    # ----- QN step via the Urban-style optimiser -----
    def _qn_step(self, x_train: torch.Tensor) -> tuple[float, float, float]:
        # Latest raw J for diagnostic logging — populated inside loss_and_grad.
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
            J_obj, J_raw = self.compute_loss(x_train, transform=True)
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
        n_collocation: int = 500,
        train_split: float = 0.7,
        rad_resample_every: int = 500,
        rad_pool_size: int = 5000,
        rad_k1: float = 1.0,
        rad_k2: float = 1.0,
        verbose_freq: int = 200,
        diag_grid_n: int = 2000,
        seed: int = 7,
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

        def fresh_collocation() -> tuple[torch.Tensor, torch.Tensor]:
            x_all = torch.empty(n_collocation, 1, device=device).uniform_(a, b)
            perm = torch.randperm(n_collocation, device=device)
            x_all = x_all[perm]
            return x_all[:n_train].detach().clone(), x_all[n_train:].detach().clone()

        x_train, x_val = fresh_collocation()

        # ---- Adam warmup ----
        for epoch in range(1, adam_epochs + 1):
            if epoch != 1 and ((epoch - 1) % rad_resample_every == 0):
                x_train, x_val = fresh_collocation()
            J_obj, J_raw = self._adam_step(x_train)
            self.obj_losses.append(J_obj)
            self.losses.append(J_raw)

            with torch.enable_grad():
                _, J_val_raw = self.compute_loss(x_val, transform=False)
            self.val_losses.append(float(J_val_raw.item()))

            self.pde_l2_errors.append(self.compute_pde_l2_norm(n_points=diag_grid_n))
            self.solution_l2_errors.append(
                self.compute_solution_l2_norm(n_points=diag_grid_n)
            )

            if epoch % verbose_freq == 0:
                print(
                    f"[ADAM]      epoch {epoch:6d} | "
                    f"J_raw={J_raw:.4e} J_val={float(J_val_raw.item()):.4e} | "
                    f"PDE_L2={self.pde_l2_errors[-1]:.4e} "
                    f"SOL_L2={self.solution_l2_errors[-1]:.4e}"
                )

        # Switch to RAD-resampled training set the moment we leave Adam — Urban
        # restarts the QN phase with a freshly resampled grid.
        x_train = self._rad_resample(rad_pool_size, n_train, rad_k1, rad_k2)
        x_val = self._rad_resample(rad_pool_size, n_collocation - n_train, rad_k1, rad_k2)

        # ---- Quasi-Newton with periodic RAD resampling and Cholesky-checked
        #      warm-starts of the Hessian. ----
        qn_total = n_epochs - adam_epochs
        for epoch in range(adam_epochs + 1, n_epochs + 1):
            qn_iter = epoch - adam_epochs
            if qn_iter != 1 and ((qn_iter - 1) % rad_resample_every == 0):
                # Keep the current H as a warm start, but verify it is still PD
                # on the new sample geometry. Reset if not.
                H = self.qn.H
                if H is not None:
                    self.qn.warm_start_from(H, cholesky_check=True)
                x_train = self._rad_resample(rad_pool_size, n_train, rad_k1, rad_k2)
                x_val = self._rad_resample(
                    rad_pool_size, n_collocation - n_train, rad_k1, rad_k2
                )

            J_obj, J_raw, alpha = self._qn_step(x_train)
            self.obj_losses.append(J_obj)
            self.losses.append(J_raw)

            with torch.enable_grad():
                _, J_val_raw = self.compute_loss(x_val, transform=False)
            self.val_losses.append(float(J_val_raw.item()))

            self.pde_l2_errors.append(self.compute_pde_l2_norm(n_points=diag_grid_n))
            self.solution_l2_errors.append(
                self.compute_solution_l2_norm(n_points=diag_grid_n)
            )

            if epoch % verbose_freq == 0 or epoch == n_epochs:
                d = self.qn.diagnostics()
                print(
                    f"[QN/{self.qn.param_groups[0]['variant']}] "
                    f"epoch {epoch:6d} ({qn_iter}/{qn_total}) | "
                    f"J_raw={J_raw:.4e} J_val={float(J_val_raw.item()):.4e} | "
                    f"alpha={alpha:.2e} tau={d['last_tau']:.2e} | "
                    f"PDE_L2={self.pde_l2_errors[-1]:.4e} "
                    f"SOL_L2={self.solution_l2_errors[-1]:.4e} | "
                    f"resets={d['n_resets']} ls_fail={d['n_ls_failures']}"
                )

        print("-" * 60)
        d = self.qn.diagnostics()
        print(
            f"Final: J_raw={self.losses[-1]:.4e}, J_val={self.val_losses[-1]:.4e}, "
            f"PDE_L2={self.pde_l2_errors[-1]:.4e}, "
            f"SOL_L2={self.solution_l2_errors[-1]:.4e}"
        )
        print(f"QN diagnostics: {d}")

    # ----- post-processing -----
    def plot_results(self, n_plot: int = 400, suffix: str = "") -> None:
        x_test = np.linspace(a, b, n_plot)
        x_torch = torch.from_numpy(x_test.reshape(-1, 1)).float().to(device)
        x_torch.requires_grad_(True)
        u = self._u_hat(x_torch)
        u_pred = u.detach().cpu().numpy().flatten()

        u_x = torch.autograd.grad(
            u, x_torch, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_torch, grad_outputs=torch.ones_like(u_x), create_graph=False
        )[0]
        u_xx_np = u_xx.detach().cpu().numpy().flatten()

        u_true = u_exact(x_test)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        ax.plot(x_test, u_true, "b-", lw=2, label=r"$u_{\mathrm{exact}}(x)$")
        ax.plot(x_test, u_pred, "r--", lw=2, label=r"$u_{\mathrm{NN}}(x)$")
        ax.set_title("Solution of BVP")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[0, 1]
        ax.plot(x_test, f(x_test), "k-", lw=2, label=r"$f(x)$")
        ax.plot(x_test, u_xx_np, "g--", lw=2, label=r"$u''_{\mathrm{NN}}(x)$")
        ax.set_title(r"Comparison of $u_{\mathrm{NN}}''(x)$ and $f(x)$")
        ax.set_xlabel("x")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 0]
        if self.losses:
            ep = np.arange(1, len(self.losses) + 1)
            ax.semilogy(ep, self.losses, "k-", lw=1, label="J (raw)")
            ax.semilogy(ep, self.val_losses, "r:", lw=1, label="J val")
            if self.loss_transform != "identity":
                ax.semilogy(
                    ep,
                    np.maximum(self.obj_losses, 1e-300),
                    "b--",
                    lw=1,
                    label=f"g(J) [{self.loss_transform}]",
                )
        ax.set_title("Training losses (log scale)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 1]
        if self.pde_l2_errors:
            ep = np.arange(1, len(self.pde_l2_errors) + 1)
            ax.semilogy(
                ep, self.pde_l2_errors, "r-", lw=2, label=r"$\|u''-f\|_{L^2}$"
            )
        if self.solution_l2_errors:
            ep = np.arange(1, len(self.solution_l2_errors) + 1)
            ax.semilogy(
                ep,
                np.array(self.solution_l2_errors),
                "b--",
                lw=2,
                label=r"$\|u_{\mathrm{NN}}-u_{\mathrm{exact}}\|_{L^2}$",
            )
        ax.set_title(r"$L^2$ norms over epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$L^2$ norm")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        out_path = os.path.join(
            FIG_DIR, f"pinn_bvpsolver_l2_SSBroyden_urban{suffix}.png"
        )
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {out_path}")

    def save_model(self, filepath: str = "../models/pinn_bvp_model_ssbroyden_urban.pth") -> None:
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
            "BFGS",
            "BFGS_scipy",
            "SSBFGS_OL",
            "SSBFGS_AB",
            "SSBroyden1",
            "SSBroyden2",
            "SSBroyden3",
        ],
    )
    p.add_argument(
        "--loss-transform",
        default="identity",
        choices=["identity", "sqrt", "log", "boxcox"],
    )
    p.add_argument("--loss-lambda", type=float, default=0.5)
    p.add_argument("--initial-scale", action="store_true")
    p.add_argument("--n-epochs", type=int, default=5000)
    p.add_argument("--adam-epochs", type=int, default=2000)
    p.add_argument("--n-collocation", type=int, default=500)
    p.add_argument("--rad-resample-every", type=int, default=500)
    p.add_argument("--rad-pool-size", type=int, default=5000)
    p.add_argument("--rad-k1", type=float, default=1.0)
    p.add_argument("--rad-k2", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--H-on-cpu", action="store_true")
    p.add_argument("--suffix", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = NeuralNetwork(hidden_layers=(32, 32, 32), activation=nn.Tanh())
    print("\nNeural Network Architecture:\n", model, "\n")

    # Print run config so the figure suffix can be matched to a header line.
    print(
        f"Variant: {args.variant} | loss_transform: {args.loss_transform} "
        f"(lambda={args.loss_lambda}) | initial_scale: {args.initial_scale}"
    )

    pinn = PINN_BVP_Solver_Urban(
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
    )
    suffix = args.suffix
    if not suffix:
        suffix = f"_{args.variant}_{args.loss_transform}"
        if args.loss_transform == "boxcox":
            suffix += f"_lam{args.loss_lambda}"
    pinn.plot_results(suffix=suffix)
    pinn.save_model(
        f"../models/pinn_bvp_model_ssbroyden_urban{suffix}.pth"
    )


if __name__ == "__main__":
    main()
