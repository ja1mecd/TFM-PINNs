"""
2D Poisson PINN on the unit square — closes the gap of section 4.3.1 of the
thesis, which currently defers the numerical results.

Equation:

    u_xx + u_yy = -2 pi^2 sin(pi x) sin(pi y)    on (0, 1) x (0, 1),
    u = 0 on the boundary,

with exact solution

    u_exact(x, y) = sin(pi x) sin(pi y).

The hard Dirichlet ansatz from the thesis is used directly:

    u_hat(x, y; theta) = x (1 - x) y (1 - y) N(x, y; theta).

The training pipeline matches the rest of chapter 4: Adam warm-up followed by
either BFGS or self-scaled Broyden refinement, identity loss by default, and
multi-seed averaging for statistical stability. The script saves a four-panel
figure (exact, learnt, pointwise error, loss + L^2 curves) plus a summary
table.

Run options expose the optimiser variant, the loss transform (so the same
script is reusable for an eventual Box-Cox sweep on the 2D Poisson), and
the seed list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
# Problem
# =============================================================================
X_LO, X_HI = 0.0, 1.0
Y_LO, Y_HI = 0.0, 1.0


def f_rhs(xy: torch.Tensor) -> torch.Tensor:
    """Forcing -2 pi^2 sin(pi x) sin(pi y); satisfies u_xx + u_yy = f."""
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return -2.0 * (np.pi ** 2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)


def u_exact_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y)


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

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


def hard_ansatz(xy: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """u_hat(x, y) = x (1-x) y (1-y) N(x, y)."""
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return x * (1.0 - x) * y * (1.0 - y) * raw


class PoissonPINN:
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

    def _u_hat(self, xy: torch.Tensor) -> torch.Tensor:
        return hard_ansatz(xy, self.model(xy))

    def _residual(self, xy: torch.Tensor, create_graph_second: bool) -> torch.Tensor:
        xy = xy.to(device)
        if not xy.requires_grad:
            xy = xy.requires_grad_(True)
        u = self._u_hat(xy)
        grad = torch.autograd.grad(
            u, xy, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = grad[:, 0:1]
        u_y = grad[:, 1:2]
        u_xx = torch.autograd.grad(
            u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=create_graph_second
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=create_graph_second
        )[0][:, 1:2]
        return (u_xx + u_yy) - f_rhs(xy)

    def _transform(self, J: torch.Tensor) -> torch.Tensor:
        eps = self.loss_eps
        if self.loss_transform == "identity":
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

    def compute_loss(self, xy: torch.Tensor, create_graph_second: bool):
        xy = xy.detach().clone().requires_grad_(True)
        r = self._residual(xy, create_graph_second=create_graph_second)
        J_raw = torch.mean(r ** 2)
        return self._transform(J_raw), J_raw.detach()

    # ---- diagnostics ----
    def _eval_grid(self, n: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        xs = np.linspace(X_LO, X_HI, n).astype(np.float32)
        ys = np.linspace(Y_LO, Y_HI, n).astype(np.float32)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        flat = np.stack([XX.ravel(), YY.ravel()], axis=1)
        return XX, YY, torch.from_numpy(flat).to(device)

    def compute_sol_l2(self, n: int = 200) -> tuple[float, float]:
        XX, YY, t = self._eval_grid(n)
        with torch.no_grad():
            u_pred = self._u_hat(t).cpu().numpy().reshape(n, n)
        u_true = u_exact_np(XX, YY)
        diff = u_pred - u_true
        # Trapezoid on the unit square.
        dx = (X_HI - X_LO) / (n - 1)
        dy = (Y_HI - Y_LO) / (n - 1)
        l2_abs = float(np.sqrt(np.sum(diff ** 2) * dx * dy))
        denom = float(np.sqrt(np.sum(u_true ** 2) * dx * dy))
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
    ) -> None:
        n_train = max(1, min(int(n_collocation * train_split), n_collocation - 1))

        def resample():
            x = torch.rand(n_collocation, 2, device=device)
            x[:, 0] = X_LO + (X_HI - X_LO) * x[:, 0]
            x[:, 1] = Y_LO + (Y_HI - Y_LO) * x[:, 1]
            perm = torch.randperm(n_collocation, device=device)
            x = x[perm]
            return x[:n_train].detach().clone(), x[n_train:].detach().clone()

        x_train, x_val = resample()

        for epoch in range(1, n_epochs + 1):
            if epoch != 1 and ((epoch - 1) % resample_every == 0):
                x_train, x_val = resample()

            use_adam = epoch <= adam_epochs

            if use_adam:
                self.adam.zero_grad()
                J_obj, J_raw = self.compute_loss(x_train, create_graph_second=True)
                J_obj.backward()
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

            if epoch == 1 or (epoch % verbose_freq == 0):
                l2_abs, l2_rel = self.compute_sol_l2(n=diag_grid_n)
                self.sol_l2.append(l2_abs)
                self.sol_rel_l2.append(l2_rel)
                phase = "ADAM" if use_adam else "QN"
                print(
                    f"Epoch {epoch:6d} [{phase}] | J_train={self.J_train[-1]:.3e} | "
                    f"J_val={self.J_val[-1]:.3e} | solL2={l2_abs:.3e} | relL2={l2_rel:.3e}"
                )
            else:
                # Reuse the most recent dense-grid evaluation to keep histories
                # the same length as J_train without re-running an expensive eval.
                if self.sol_l2:
                    self.sol_l2.append(self.sol_l2[-1])
                    self.sol_rel_l2.append(self.sol_rel_l2[-1])
                else:
                    self.sol_l2.append(float("nan"))
                    self.sol_rel_l2.append(float("nan"))


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
    field_pred: np.ndarray  # last seed's field for the figure


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
) -> tuple[SeedResult, ...]:
    out: list[SeedResult] = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = Net(hidden=hidden)
        pinn = PoissonPINN(
            model=model,
            lr=lr,
            loss_transform=loss_transform,
            loss_lambda=loss_lambda,
            qn_variant=qn_variant,
        )
        print(f"\n[seed={seed}] training 2D Poisson PINN")
        pinn.train(
            n_epochs=n_epochs,
            adam_epochs=adam_epochs,
            n_collocation=n_collocation,
            verbose_freq=max(1, n_epochs // 10),
            diag_grid_n=200,
        )
        XX, YY, t = pinn._eval_grid(150)
        with torch.no_grad():
            u_pred = pinn._u_hat(t).cpu().numpy().reshape(150, 150)
        out.append(
            SeedResult(
                seed=seed,
                J_val=np.asarray(pinn.J_val, dtype=np.float64),
                sol_l2=np.asarray(pinn.sol_l2, dtype=np.float64),
                final_J_val=float(pinn.J_val[-1]),
                final_sol_l2=float(pinn.sol_l2[-1]) if pinn.sol_l2 else float("nan"),
                final_sol_rel_l2=float(pinn.sol_rel_l2[-1]) if pinn.sol_rel_l2 else float("nan"),
                field_pred=u_pred,
            )
        )
    return tuple(out)


def plot_results(
    results: tuple[SeedResult, ...],
    out_path: str,
    n: int = 150,
) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    xs = np.linspace(X_LO, X_HI, n)
    ys = np.linspace(Y_LO, Y_HI, n)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    u_true = u_exact_np(XX, YY)
    u_pred = results[-1].field_pred  # last seed
    err = np.abs(u_pred - u_true)

    im0 = ax[0, 0].imshow(u_true.T, origin="lower", extent=(X_LO, X_HI, Y_LO, Y_HI),
                          cmap="viridis", aspect="equal")
    ax[0, 0].set_title(r"$u_{\mathrm{exact}}(x, y)$")
    fig.colorbar(im0, ax=ax[0, 0], shrink=0.8)

    im1 = ax[0, 1].imshow(u_pred.T, origin="lower", extent=(X_LO, X_HI, Y_LO, Y_HI),
                          cmap="viridis", aspect="equal")
    ax[0, 1].set_title(r"$\widehat{u}_\theta(x, y)$ (last seed)")
    fig.colorbar(im1, ax=ax[0, 1], shrink=0.8)

    im2 = ax[1, 0].imshow(err.T, origin="lower", extent=(X_LO, X_HI, Y_LO, Y_HI),
                          cmap="inferno", aspect="equal", norm=matplotlib.colors.LogNorm(vmin=max(1e-12, err.min() + 1e-12), vmax=err.max() + 1e-12))
    ax[1, 0].set_title(r"$|\widehat{u}_\theta - u_{\mathrm{exact}}|$ (log scale)")
    fig.colorbar(im2, ax=ax[1, 0], shrink=0.8)

    for r in results:
        ax[1, 1].semilogy(r.J_val, alpha=0.4, label=f"seed {r.seed}")
    H = np.stack([r.J_val for r in results], axis=0)
    ax[1, 1].semilogy(np.mean(H, axis=0), color="k", linewidth=1.6, label="mean")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel(r"$\mathcal{J}_{\mathrm{val}}$")
    ax[1, 1].set_title("Validation residual MSE")
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
        "problem": "2D Poisson on (0,1)^2, hard Dirichlet ansatz",
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
    p = argparse.ArgumentParser(description="2D Poisson PINN on the unit square (multi-seed).")
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
    p.add_argument("--loss-lambda", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=15000)
    p.add_argument("--adam-epochs", type=int, default=5000)
    p.add_argument("--n-collocation", type=int, default=2000)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32, 32])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--results-dir", type=str, default=os.path.join("..", "results"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.results_dir,
        f"poisson2d_unitsquare_{args.qn_variant}_{args.loss_transform}_{run_tag}",
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
    )

    plot_results(results, out_path=os.path.join(out_dir, "poisson2d_results.png"))
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
    )
    print(f"All artefacts written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
