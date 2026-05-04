"""
1D PINN solver for the high-frequency boundary value problem

    -u''(x) = (k pi)^2 sin(k pi x),    x in (0, 1),    u(0) = u(1) = 0,

with exact solution u_exact(x) = sin(k pi x). The wavenumber k controls the
oscillation count on the unit interval; k = 4 (the default) matches the
Helmholtz-type benchmark in section 5 of

    Urban, Stefanou & Pons, "Unveiling the optimization process of physics
    informed neural networks: How accurate and competitive can PINNs be?",
    J. Comp. Phys. 523, 113656 (2025),

so that the 1D and 2D experiments in chapter 4 of the thesis share a wavenumber.

The boundary conditions are imposed exactly via the hard Dirichlet ansatz

    u_hat(x; theta) = x (1 - x) N(x; theta),

so the optimisation reduces to minimising the interior PDE residual.

Two knobs are exposed for the optimiser comparison and the Box-Cox sweep:

    --qn-variant       one of {"bfgs", "ssbfgs", "ssbroyden"}
    --loss-transform   one of {"identity", "sqrt", "log", "boxcox"}

When `--loss-transform=boxcox`, `--loss-lambda` selects the exponent of the
Box-Cox transformation g_lambda(J + eps) = (expm1(lambda log(J + eps))) / lambda
(falling back to log(J + eps) at lambda = 0). The expm1 form avoids
catastrophic cancellation for small |lambda|.
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

matplotlib.use("Agg")  # headless: SSH / no display
import matplotlib.pyplot as plt  # noqa: E402

# Make the shared optimizer importable when running from BVP/one_d/.
_OPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimizers")
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)
from ssbroyden import SSBroydenOptimizer  # noqa: E402


# =============================================================================
# DEVICE
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# PROBLEM SETUP: -u'' = (k pi)^2 sin(k pi x) on [0, 1]
# =============================================================================
A, B = 0.0, 1.0  # interval endpoints

# Wavenumber k. Default k=4 matches the 2D NLP benchmark of Urban et al.
# Override with --wavenumber.
DEFAULT_K = 4


def f_rhs(x: torch.Tensor, k: float) -> torch.Tensor:
    """Forcing f(x) = -u''(x) = (k pi)^2 sin(k pi x). The script optimises u'' = -f."""
    return -((k * np.pi) ** 2) * torch.sin(k * np.pi * x)


def u_exact(x: torch.Tensor | np.ndarray, k: float):
    """Exact solution sin(k pi x)."""
    if isinstance(x, torch.Tensor):
        return torch.sin(k * np.pi * x)
    return np.sin(k * np.pi * x)


# =============================================================================
# Neural network
# =============================================================================
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers=(64, 64, 64), activation: nn.Module | None = None) -> None:
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
# PINN solver with optional loss transform and SSBroyden refinement
# =============================================================================
class PINN_BVP_SSBroyden:
    def __init__(
        self,
        model: nn.Module,
        k: float = DEFAULT_K,
        lr: float = 1e-3,
        lambda_pde: float = 1.0,
        loss_transform: str = "identity",
        loss_lambda: float = 0.5,
        loss_eps: float = 1e-12,
        rel_err_eps: float = 1e-12,
        qn_variant: str = "ssbroyden",
        qn_H_on_cpu: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.k = float(k)
        self.lambda_pde = float(lambda_pde)
        self.loss_transform = str(loss_transform)
        self.loss_lambda = float(loss_lambda)
        self.loss_eps = float(loss_eps)
        self.rel_err_eps = float(rel_err_eps)

        self.adam = optim.Adam(self.model.parameters(), lr=lr)
        self.quasi_newton = SSBroydenOptimizer(
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
            H_on_cpu=qn_H_on_cpu,
        )

        # Logs
        self.obj_train: list[float] = []
        self.obj_val: list[float] = []
        self.J_train: list[float] = []
        self.J_val: list[float] = []
        self.pde_l2: list[float] = []
        self.sol_l2: list[float] = []
        self.sol_rel_l2: list[float] = []

        self.best_state: dict | None = None
        self.best_val_ma = float("inf")

    # ---- transform J -> objective ----
    def _transform_objective(self, J_raw: torch.Tensor) -> torch.Tensor:
        eps = self.loss_eps
        if self.loss_transform == "identity":
            return J_raw
        if self.loss_transform == "sqrt":
            return torch.sqrt(J_raw + eps)
        if self.loss_transform == "log":
            return torch.log(J_raw + eps)
        if self.loss_transform == "boxcox":
            lam = self.loss_lambda
            shifted = J_raw + eps
            if lam == 0.0:
                return torch.log(shifted)
            return torch.expm1(lam * torch.log(shifted)) / lam
        raise ValueError(f"Unknown loss_transform={self.loss_transform!r}")

    # ---- hard Dirichlet ansatz ----
    def _u_hat(self, x: torch.Tensor) -> torch.Tensor:
        # u(0) = u(1) = 0 by construction.
        return x * (1.0 - x) * self.model(x)

    # ---- PDE residual u'' - f ----
    def _residual(self, x: torch.Tensor, create_graph_second: bool) -> torch.Tensor:
        x = x.to(device)
        if not x.requires_grad:
            x = x.requires_grad_(True)
        u = self._u_hat(x)
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=create_graph_second,
        )[0]
        return u_xx - f_rhs(x, self.k)

    # ---- loss ----
    def compute_loss(self, x_interior: torch.Tensor, create_graph_second: bool):
        x = x_interior.detach().clone().requires_grad_(True)
        r = self._residual(x, create_graph_second=create_graph_second)
        J_raw = self.lambda_pde * (torch.mean(r**2) * (B - A))
        J_obj = self._transform_objective(J_raw)
        return J_obj, J_raw.detach()

    # ---- diagnostics on a uniform 1D grid ----
    def _grid(self, n: int):
        return np.linspace(A, B, n).astype(np.float32)

    def compute_pde_l2(self, n: int = 400) -> float:
        xs = self._grid(n)
        xt = torch.from_numpy(xs.reshape(-1, 1)).to(device)
        r = self._residual(xt, create_graph_second=False).detach().cpu().numpy().ravel()
        return float(np.sqrt(np.trapz(r**2, xs)))

    def compute_sol_l2(self, n: int = 400) -> float:
        xs = self._grid(n)
        xt = torch.from_numpy(xs.reshape(-1, 1)).to(device)
        u_true = u_exact(xs, self.k)
        with torch.no_grad():
            u_pred = self._u_hat(xt).cpu().numpy().ravel()
        diff = u_pred - u_true
        return float(np.sqrt(np.trapz(diff**2, xs)))

    def compute_sol_rel_l2(self, n: int = 400) -> float:
        xs = self._grid(n)
        xt = torch.from_numpy(xs.reshape(-1, 1)).to(device)
        u_true = u_exact(xs, self.k)
        with torch.no_grad():
            u_pred = self._u_hat(xt).cpu().numpy().ravel()
        num = np.trapz((u_pred - u_true) ** 2, xs)
        den = np.trapz(u_true**2, xs)
        return float(np.sqrt(num) / (np.sqrt(den) + self.rel_err_eps))

    # ---- training ----
    def train(
        self,
        n_epochs: int = 5000,
        n_collocation: int = 400,
        train_split: float = 0.8,
        resample_every: int = 500,
        adam_epochs: int = 2000,
        verbose_freq: int = 200,
        diag_grid_n: int = 400,
        patience: int = 5000,
        min_delta: float = 1e-10,
        moving_avg_window: int = 20,
        scheduler_patience: int = 300,
        scheduler_threshold: float = 1e-4,
        scheduler_gamma: float = 0.9,
        scheduler_min_lr: float = 1e-6,
    ) -> None:
        print(
            "\nTraining 1D PINN: -u'' = (k pi)^2 sin(k pi x), "
            f"k = {self.k:g}, domain [{A}, {B}]"
        )
        if self.loss_transform == "boxcox":
            print(
                f"  Loss transform:  boxcox(lambda={self.loss_lambda:g}, "
                f"eps={self.loss_eps:g})"
            )
        else:
            print(f"  Loss transform:  {self.loss_transform}  (eps={self.loss_eps:g})")
        print(
            f"  Optimizers:      Adam ({adam_epochs} iters)"
            f" then {self.quasi_newton.param_groups[0]['variant'].upper()}"
            f" ({n_epochs - adam_epochs} iters)"
        )
        print("-" * 72)

        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be in (0, 1).")
        if n_collocation < 2:
            raise ValueError("n_collocation must be >= 2.")
        if resample_every < 1:
            raise ValueError("resample_every must be >= 1.")
        if adam_epochs < 0 or adam_epochs >= n_epochs:
            raise ValueError("adam_epochs must be in [0, n_epochs - 1].")

        n_train = int(n_collocation * train_split)
        n_train = min(max(n_train, 1), n_collocation - 1)

        def resample_block():
            x = torch.empty(n_collocation, 1, device=device).uniform_(A, B)
            perm = torch.randperm(n_collocation, device=device)
            x = x[perm]
            return x[:n_train].detach().clone(), x[n_train:].detach().clone()

        x_train, x_val = resample_block()

        def make_plateau(opt):
            return optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=scheduler_gamma,
                patience=scheduler_patience,
                threshold=scheduler_threshold,
                min_lr=scheduler_min_lr,
            )

        sch_adam = make_plateau(self.adam)
        sch_qn = make_plateau(self.quasi_newton)

        ma_buf: list[float] = []
        epochs_no_improve = 0
        last_pde = np.nan
        last_sol = np.nan
        last_rel = np.nan

        for epoch in range(1, n_epochs + 1):
            if epoch != 1 and ((epoch - 1) % resample_every == 0):
                x_train, x_val = resample_block()

            use_adam = epoch <= adam_epochs
            opt = self.adam if use_adam else self.quasi_newton
            sch = sch_adam if use_adam else sch_qn

            if use_adam:
                opt.zero_grad()
                J_obj, J_raw = self.compute_loss(x_train, create_graph_second=True)
                J_obj.backward()
                opt.step()
            else:
                holder: dict = {}

                def closure():
                    opt.zero_grad()
                    J_obj_c, J_raw_c = self.compute_loss(
                        x_train, create_graph_second=True
                    )
                    holder["J_raw"] = J_raw_c
                    J_obj_c.backward()
                    return J_obj_c

                def loss_eval():
                    J_obj_e, _ = self.compute_loss(x_train, create_graph_second=False)
                    return J_obj_e

                J_obj = opt.step(closure, loss_eval)
                J_raw = holder["J_raw"]

            with torch.set_grad_enabled(True):
                val_obj, val_raw = self.compute_loss(
                    x_val, create_graph_second=False
                )

            self.obj_train.append(float(J_obj.item()))
            self.obj_val.append(float(val_obj.item()))
            self.J_train.append(float(J_raw.item()))
            self.J_val.append(float(val_raw.item()))

            ma_buf.append(float(val_obj.item()))
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

            sch.step(float(val_obj.item()))

            if epoch == 1 or (epoch % verbose_freq == 0):
                last_pde = self.compute_pde_l2(n=diag_grid_n)
                last_sol = self.compute_sol_l2(n=diag_grid_n)
                last_rel = self.compute_sol_rel_l2(n=diag_grid_n)

            self.pde_l2.append(last_pde)
            self.sol_l2.append(last_sol)
            self.sol_rel_l2.append(last_rel)

            if epoch == 1 or (epoch % verbose_freq == 0):
                lr_now = opt.param_groups[0]["lr"]
                phase = "ADAM" if use_adam else self.quasi_newton.param_groups[0][
                    "variant"
                ].upper()
                print(
                    f"Epoch {epoch:6d} [{phase}] | "
                    f"obj={self.obj_train[-1]:.3e}, val={self.obj_val[-1]:.3e} | "
                    f"J={self.J_train[-1]:.3e}, valJ={self.J_val[-1]:.3e} | "
                    f"pdeL2={last_pde:.3e}, solL2={last_sol:.3e}, "
                    f"relL2={last_rel:.3e} | lr={lr_now:.2e}"
                )

            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val-MA improvement for {patience} epochs)."
                )
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        print("-" * 72)
        print(f"Done. Best val objective moving average: {self.best_val_ma:.6e}")

    # ---- plotting ----
    def plot_results(
        self, n: int = 400, save_path: str | None = None, dpi: int = 150
    ) -> None:
        xs = self._grid(n)
        xt = torch.from_numpy(xs.reshape(-1, 1)).to(device)

        # Solution and second derivative on the dense grid.
        xt_grad = xt.clone().requires_grad_(True)
        u_pred_t = self._u_hat(xt_grad)
        u_x = torch.autograd.grad(
            u_pred_t, xt_grad, grad_outputs=torch.ones_like(u_pred_t), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, xt_grad, grad_outputs=torch.ones_like(u_x), create_graph=False
        )[0]
        u_pred = u_pred_t.detach().cpu().numpy().ravel()
        uxx_pred = u_xx.detach().cpu().numpy().ravel()
        u_true = u_exact(xs, self.k)
        f_vals = (-((self.k * np.pi) ** 2) * np.sin(self.k * np.pi * xs)).astype(np.float64)

        fig, ax = plt.subplots(2, 2, figsize=(13, 9))

        # (0,0) Solution
        ax[0, 0].plot(xs, u_true, "k--", linewidth=2, label=r"$u_{\mathrm{exact}}(x)$")
        ax[0, 0].plot(xs, u_pred, "C0-", linewidth=1.5, label=r"$\widehat{u}_\theta(x)$")
        ax[0, 0].set_title(f"Solution (k = {self.k:g})")
        ax[0, 0].set_xlabel("x")
        ax[0, 0].set_ylabel("u(x)")
        ax[0, 0].grid(True, alpha=0.3)
        ax[0, 0].legend()

        # (0,1) Second derivative vs forcing
        ax[0, 1].plot(xs, f_vals, "k--", linewidth=2, label=r"$f(x)$")
        ax[0, 1].plot(xs, uxx_pred, "C3-", linewidth=1.5, label=r"$\widehat{u}_\theta''(x)$")
        ax[0, 1].set_title(r"Comparison of $\widehat{u}_\theta''(x)$ and $f(x)$")
        ax[0, 1].set_xlabel("x")
        ax[0, 1].set_ylabel("value")
        ax[0, 1].grid(True, alpha=0.3)
        ax[0, 1].legend()

        # (1,0) Loss curves
        ax[1, 0].semilogy(self.obj_train, label="obj(train)")
        ax[1, 0].semilogy(self.obj_val, label="obj(val)")
        ax[1, 0].semilogy(self.J_train, "--", label="J(train)")
        ax[1, 0].semilogy(self.J_val, "--", label="J(val)")
        ax[1, 0].grid(True, alpha=0.3)
        ax[1, 0].legend()
        ax[1, 0].set_title("Objective / loss curves")
        ax[1, 0].set_xlabel("Epoch")

        # (1,1) Errors over epochs
        ax[1, 1].semilogy(self.pde_l2, label=r"$\|\widehat{u}_\theta'' - f\|_{L^2}$")
        ax[1, 1].semilogy(self.sol_l2, label=r"$\|\widehat{u}_\theta - u_{\mathrm{exact}}\|_{L^2}$")
        ax[1, 1].semilogy(self.sol_rel_l2, label=r"relative $L^2$")
        ax[1, 1].grid(True, alpha=0.3)
        ax[1, 1].legend()
        ax[1, 1].set_title("Errors over epochs")
        ax[1, 1].set_xlabel("Epoch")

        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure to: {save_path}")
        plt.close(fig)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to: {path}")

    def save_results(
        self,
        run_dir: str,
        n_eval: int = 400,
        extra_metadata: dict | None = None,
    ) -> None:
        """Dump training history, field snapshots, and a summary.json to `run_dir`."""
        os.makedirs(run_dir, exist_ok=True)

        np.savez(
            os.path.join(run_dir, "history.npz"),
            obj_train=np.asarray(self.obj_train, dtype=np.float64),
            obj_val=np.asarray(self.obj_val, dtype=np.float64),
            J_train=np.asarray(self.J_train, dtype=np.float64),
            J_val=np.asarray(self.J_val, dtype=np.float64),
            pde_l2=np.asarray(self.pde_l2, dtype=np.float64),
            sol_l2=np.asarray(self.sol_l2, dtype=np.float64),
            sol_rel_l2=np.asarray(self.sol_rel_l2, dtype=np.float64),
        )

        xs = self._grid(n_eval)
        xt = torch.from_numpy(xs.reshape(-1, 1)).to(device)
        with torch.no_grad():
            u_pred = self._u_hat(xt).cpu().numpy().ravel()
        u_true = u_exact(xs, self.k)
        abs_err = np.abs(u_pred - u_true)

        np.savez(
            os.path.join(run_dir, "fields.npz"),
            x=xs.astype(np.float64),
            u_exact=u_true.astype(np.float64),
            u_pred=u_pred.astype(np.float64),
            abs_err=abs_err.astype(np.float64),
        )

        summary = {
            "problem": "1D BVP -u'' = (k pi)^2 sin(k pi x) on [0,1] (Dirichlet, hard ansatz)",
            "wavenumber_k": self.k,
            "qn_variant": self.quasi_newton.param_groups[0]["variant"],
            "loss_transform": self.loss_transform,
            "loss_lambda": self.loss_lambda,
            "loss_eps": self.loss_eps,
            "lambda_pde": self.lambda_pde,
            "domain": [A, B],
            "n_epochs_run": len(self.obj_train),
            "best_val_objective_ma": float(self.best_val_ma),
            "final_obj_train": float(self.obj_train[-1]) if self.obj_train else None,
            "final_obj_val": float(self.obj_val[-1]) if self.obj_val else None,
            "final_J_train": float(self.J_train[-1]) if self.J_train else None,
            "final_J_val": float(self.J_val[-1]) if self.J_val else None,
            "final_pde_l2": float(self.pde_l2[-1]) if self.pde_l2 else None,
            "final_sol_l2": float(self.sol_l2[-1]) if self.sol_l2 else None,
            "final_sol_rel_l2": (
                float(self.sol_rel_l2[-1]) if self.sol_rel_l2 else None
            ),
            "max_abs_err": float(np.max(abs_err)),
            "mean_abs_err": float(np.mean(abs_err)),
            "device": str(device),
            "torch_version": torch.__version__,
        }
        if extra_metadata is not None:
            summary.update(extra_metadata)

        with open(os.path.join(run_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

        print(f"All run artefacts written to: {os.path.abspath(run_dir)}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="1D PINN with Adam + SSBroyden refinement and Box-Cox loss transforms."
    )
    p.add_argument("--wavenumber", type=float, default=DEFAULT_K, help="Wavenumber k.")
    p.add_argument(
        "--qn-variant",
        type=str,
        default="ssbroyden",
        choices=["bfgs", "ssbfgs", "ssbroyden"],
    )
    p.add_argument(
        "--loss-transform",
        type=str,
        default="identity",
        choices=["identity", "sqrt", "log", "boxcox"],
    )
    p.add_argument(
        "--loss-lambda",
        type=float,
        default=0.5,
        help="Box-Cox exponent (only used when --loss-transform=boxcox).",
    )
    p.add_argument("--loss-eps", type=float, default=1e-12)
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
        help="Parent directory for run artefacts.",
    )
    p.add_argument(
        "--run-tag-prefix",
        type=str,
        default="bvp1d",
        help="Prefix for the run directory name.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = NeuralNetwork(hidden_layers=tuple(args.hidden), activation=nn.Tanh())
    print("\nNeural Network Architecture:\n")
    print(model, "\n")

    pinn = PINN_BVP_SSBroyden(
        model=model,
        k=args.wavenumber,
        lr=args.lr,
        lambda_pde=1.0,
        loss_transform=args.loss_transform,
        loss_lambda=args.loss_lambda,
        loss_eps=args.loss_eps,
        qn_variant=args.qn_variant,
    )

    pinn.train(
        n_epochs=args.epochs,
        n_collocation=args.n_collocation,
        train_split=0.8,
        resample_every=args.resample_every,
        adam_epochs=args.adam_epochs,
        verbose_freq=max(1, args.epochs // 25),
        diag_grid_n=400,
        patience=args.epochs,  # disable early stop by default
        min_delta=1e-12,
        moving_avg_window=20,
    )

    transform_tag = (
        f"boxcox_lam{args.loss_lambda:g}"
        if args.loss_transform == "boxcox"
        else args.loss_transform
    )
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{args.run_tag_prefix}_k{args.wavenumber:g}_"
        f"{args.qn_variant}_{transform_tag}_{run_tag}"
    )
    run_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    pinn.plot_results(n=400, save_path=os.path.join(run_dir, "results.png"))
    pinn.save(
        os.path.join(
            "..",
            "models",
            f"pinn_bvp1d_{args.qn_variant}_{transform_tag}.pth",
        )
    )
    pinn.save_results(
        run_dir,
        n_eval=400,
        extra_metadata={"run_name": run_name, "run_tag": run_tag, "args": vars(args)},
    )


if __name__ == "__main__":
    main()
