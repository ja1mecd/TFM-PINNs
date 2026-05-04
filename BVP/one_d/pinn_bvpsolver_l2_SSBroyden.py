"""1D PINN BVP solver with Adam warm start followed by self-scaled Broyden refinement.

This is the SSBroyden counterpart of `pinn_bvpsolver_l2_BFGS.py`. The problem,
domain, hard Dirichlet ansatz, network architecture, training protocol, and
diagnostics are identical to the BFGS script: only the post-warm-up optimiser
is swapped for the self-scaled Broyden update of Urban et al. (2025), provided
by `BVP/optimizers/ssbroyden.py`. The objective is the unmodified mean-squared
PDE residual; no loss transformation is applied.

Output figure: `figures/pinn_bvpsolver_l2_SSBroyden.png`, with the same 2x2
panel layout as the Adam-only and Adam->BFGS figures so the three runs can be
displayed side by side in the optimiser comparison of section 4.2.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")  # headless backend (SSH / no display)
import matplotlib.pyplot as plt  # noqa: E402

# Make the shared optimiser importable when running from BVP/one_d/.
_OPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "optimizers"
)
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)
from ssbroyden import SSBroydenOptimizer  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# DEVICE
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# PROBLEM DEFINITION: -u''(x) = (k pi)^2 sin(k pi x) on [0, 1]
# -------------------------------------------------------------------------
# Exact solution u(x) = sin(k pi x), with u(0) = u(1) = 0 (homogeneous Dirichlet).
# Wavenumber k = 4 matches the 2D NLP benchmark of Urban et al. 2025 and is the
# same value used by the Adam-only and Adam->BFGS scripts in this directory.

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


alpha, beta = u_exact(a), u_exact(b)  # both 0 (homogeneous)


# -------------------------------------------------------------------------
# NEURAL NETWORK
# -------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers=[32, 32, 32], activation=nn.Tanh()):
        super(NeuralNetwork, self).__init__()

        layers = []
        input_dim = 1

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# -------------------------------------------------------------------------
# PINN FOR BVP: ENFORCE u'' ~ f AND BOUNDARY CONDITIONS
# -------------------------------------------------------------------------
class PINN_BVP_Solver:
    def __init__(self, model, lr=1e-3, lambda_bc=10.0, lambda_pde=1.0):
        """
        Parameters
        ----------
        model : nn.Module
            Neural network u_theta(x).
        lr : float
            Learning rate for the Adam warm-up phase.
        lambda_bc : float
            Weight of boundary-condition loss in the total loss. Inert under
            the hard Dirichlet ansatz, kept for parity with the BFGS script.
        lambda_pde : float
            Weight of the PDE residual loss.
        """
        self.model = model.to(device)
        self.adam_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.ssbroyden_optimizer = SSBroydenOptimizer(
            self.model.parameters(),
            variant="ssbroyden",
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
        self.optimizer = self.adam_optimizer
        self.lambda_bc = lambda_bc
        self.lambda_pde = lambda_pde

        self.losses = []  # total loss (PDE + BC)
        self.val_losses = []  # validation loss
        self.pde_losses = []  # PDE residual loss
        self.bc_losses = []  # boundary loss
        self.pde_l2_errors = []  # L2 norm of PDE residual
        self.solution_l2_errors = []  # L2 norm of solution error (u_NN - u_exact)

        self.best_model_state = None
        self.best_loss = float("inf")

    # ------------------- hard-enforced solution -------------------
    def _u_hat(self, x):
        base = alpha * (b - x) / (b - a) + beta * (x - a) / (b - a)
        return base + (x - a) * (b - x) * self.model(x)

    # ------------------- core loss components -------------------
    def _pde_residual(self, x_interior):
        x_interior = x_interior.to(device)
        x_interior.requires_grad_(True)

        u = self._u_hat(x_interior)
        # First derivative u_x
        u_x = torch.autograd.grad(
            u, x_interior, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        # Second derivative u_xx
        u_xx = torch.autograd.grad(
            u_x, x_interior, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]

        # Right-hand side f(x)
        f_vals = f(x_interior)
        r = u_xx - f_vals
        return r

    def compute_loss(self, x_interior=None, n_collocation_points=100):
        """Compute total loss = lambda_pde * PDE-loss + lambda_bc * BC-loss.

        Notes
        -----
        If `x_interior` is reused across epochs (as in block-resampling), we
        create a fresh leaf tensor each call via detach().clone().requires_grad_(True).
        This prevents accumulating gradients on the collocation-point tensor itself.
        """

        # Interior collocation points
        if x_interior is None:
            x_interior = torch.empty(n_collocation_points, 1, device=device).uniform_(a, b)
        else:
            x_interior = x_interior.to(device)

        # IMPORTANT: make a fresh leaf tensor each call (prevents .grad accumulation on reused points)
        x_interior = x_interior.detach().clone().requires_grad_(True)

        r = self._pde_residual(x_interior)
        loss_pde = torch.mean(r**2) * (b - a)  # scale by interval length

        # Boundary points (zero under the hard ansatz)
        loss_bc = torch.zeros((), device=device)

        loss_total = self.lambda_pde * loss_pde + self.lambda_bc * loss_bc
        return loss_total, loss_pde.detach(), loss_bc.detach()

    # ------------------- diagnostics -------------------
    def compute_pde_l2_norm(self, n_points=500):
        x_test = np.linspace(a, b, n_points)
        x_torch = torch.FloatTensor(x_test.reshape(-1, 1)).to(device)
        x_torch.requires_grad_(True)

        u = self._u_hat(x_torch)

        u_x = torch.autograd.grad(
            u, x_torch, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_torch, grad_outputs=torch.ones_like(u_x), create_graph=False
        )[0]

        u_xx_np = u_xx.detach().cpu().numpy().flatten()
        f_np = f(x_test)
        residual = u_xx_np - f_np

        l2_norm = np.sqrt(np.trapz(residual**2, x_test))
        return l2_norm

    def compute_solution_l2_norm(self, n_points=500):
        x_test = np.linspace(a, b, n_points)

        try:
            u_true = u_exact(x_test)
        except Exception:
            return None

        x_torch = torch.FloatTensor(x_test.reshape(-1, 1)).to(device)
        with torch.no_grad():
            u_pred = self._u_hat(x_torch).cpu().numpy().flatten()

        diff = u_pred - u_true
        l2_norm = np.sqrt(np.trapz(diff**2, x_test))
        return l2_norm

    # ------------------- training -------------------
    def train(
        self,
        n_epochs=5000,
        n_collocation_points=500,
        verbose_freq=1000,
        patience=5000,
        min_delta=1e-7,
        moving_avg_window=20,
        pde_l2_points=2000,
        train_split=0.7,
        scheduler_patience=5000,
        scheduler_threshold=1e-4,
        scheduler_gamma=0.9,
        scheduler_min_lr=1e-6,
        resample_every=500,
        adam_epochs=2000,
    ):
        print("\nStarting PINN training for BVP u''(x)=f(x) with Adam -> SSBroyden...")
        print(f"Domain: [{a}, {b}], BC: u(a)={alpha}, u(b)={beta}")
        print(f"lambda_bc = {self.lambda_bc}")
        print(f"lambda_pde = {self.lambda_pde}")
        print("-" * 60)

        self.best_model_state = None
        self.best_loss = float("inf")
        self.losses.clear()
        self.val_losses.clear()
        self.pde_losses.clear()
        self.bc_losses.clear()
        self.pde_l2_errors.clear()
        self.solution_l2_errors.clear()

        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be between 0 and 1 (exclusive).")
        if n_collocation_points < 2:
            raise ValueError(
                "n_collocation_points must be at least 2 to allow a validation split."
            )
        if resample_every < 1:
            raise ValueError("resample_every must be >= 1.")
        if adam_epochs < 0 or adam_epochs >= n_epochs:
            raise ValueError("adam_epochs must be >= 0 and < n_epochs.")

        n_train = int(n_collocation_points * train_split)
        n_train = min(max(n_train, 1), n_collocation_points - 1)

        def resample_collocation_block():
            """Sample a fresh uniform block of collocation points and split into train/val."""
            x_all = torch.empty(n_collocation_points, 1, device=device).uniform_(a, b)
            perm = torch.randperm(n_collocation_points, device=device)
            x_all = x_all[perm]
            x_train_block = x_all[:n_train].detach().clone()
            x_val_block = x_all[n_train:].detach().clone()
            return x_train_block, x_val_block

        x_train_block, x_val_block = resample_collocation_block()

        self.adam_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.adam_optimizer,
            mode="min",
            factor=scheduler_gamma,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            verbose=True,
            min_lr=scheduler_min_lr,
        )
        self.ssbroyden_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.ssbroyden_optimizer,
            mode="min",
            factor=scheduler_gamma,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            verbose=True,
            min_lr=scheduler_min_lr,
        )

        epochs_without_improvement = 0
        moving_avg_losses = []
        actual_epochs = 0

        for epoch in range(1, n_epochs + 1):
            if epoch != 1 and ((epoch - 1) % resample_every == 0):
                x_train_block, x_val_block = resample_collocation_block()

            if epoch <= adam_epochs:
                self.optimizer = self.adam_optimizer
                scheduler = self.adam_scheduler
            else:
                self.optimizer = self.ssbroyden_optimizer
                scheduler = self.ssbroyden_scheduler

            if self.optimizer is self.adam_optimizer:
                self.optimizer.zero_grad()
                loss_total, loss_pde, loss_bc = self.compute_loss(
                    x_interior=x_train_block, n_collocation_points=n_collocation_points
                )
                loss_total.backward()
                self.optimizer.step()
            else:
                loss_parts = {}

                def closure():
                    self.optimizer.zero_grad()
                    loss_total, loss_pde, loss_bc = self.compute_loss(
                        x_interior=x_train_block,
                        n_collocation_points=n_collocation_points,
                    )
                    loss_parts["pde"] = loss_pde
                    loss_parts["bc"] = loss_bc
                    loss_total.backward()
                    return loss_total

                def loss_eval():
                    loss_total, _, _ = self.compute_loss(
                        x_interior=x_train_block,
                        n_collocation_points=n_collocation_points,
                    )
                    return loss_total

                loss_total = self.optimizer.step(closure, loss_eval)
                loss_pde = loss_parts["pde"]
                loss_bc = loss_parts["bc"]

            with torch.set_grad_enabled(True):
                val_loss, _, _ = self.compute_loss(
                    x_interior=x_val_block, n_collocation_points=x_val_block.shape[0]
                )

            loss_value = loss_total.item()
            self.losses.append(loss_value)
            self.pde_losses.append(loss_pde.item())
            self.bc_losses.append(loss_bc.item())
            self.val_losses.append(val_loss.item())

            pde_l2 = self.compute_pde_l2_norm(n_points=pde_l2_points)
            self.pde_l2_errors.append(pde_l2)

            sol_l2 = self.compute_solution_l2_norm(n_points=pde_l2_points)
            if sol_l2 is not None:
                self.solution_l2_errors.append(sol_l2)
            else:
                self.solution_l2_errors.append(np.nan)

            actual_epochs = epoch

            moving_avg_losses.append(val_loss.item())
            if len(moving_avg_losses) > moving_avg_window:
                moving_avg_losses.pop(0)
            moving_avg = np.mean(moving_avg_losses)

            if moving_avg + min_delta < self.best_loss:
                self.best_loss = moving_avg
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler.step(val_loss.item())

            if epoch % verbose_freq == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                phase = "ADAM" if self.optimizer is self.adam_optimizer else "SSBROYDEN"
                print(
                    f"Epoch {epoch:6d} [{phase}] | "
                    f"Loss: {loss_value:.4e} | "
                    f"Val: {val_loss.item():.4e} | "
                    f"PDE: {loss_pde.item():.4e} | "
                    f"BC: {loss_bc.item():.4e} | "
                    f"LR: {current_lr:.2e} | "
                    f"||u''-f||_L2: {pde_l2:.4e} "
                    f"||u_NN-u_exact||_L2: {sol_l2:.4e}"
                )

            if epoch > adam_epochs and epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"No improvement in moving average loss for {patience} epochs.")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(
                f"\nLoaded best model with moving-average loss: {self.best_loss:.6e}"
            )

        print("-" * 60)
        print(f"Training completed: {actual_epochs} / {n_epochs} epochs.")

    # ------------------- post-processing -------------------
    def plot_results(self, n_plot_points=200):
        x_test = np.linspace(a, b, n_plot_points)

        x_torch = torch.tensor(
            x_test.reshape(-1, 1),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        u = self.model(x_torch)
        u = self._u_hat(x_torch)
        u_pred = u.detach().cpu().numpy().flatten()

        try:
            u_true = u_exact(x_test)
            have_exact = True
            abs_error = np.abs(u_pred - u_true)
        except Exception:
            u_true = None
            have_exact = False
            abs_error = None

        u_x = torch.autograd.grad(
            u, x_torch, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x_torch, grad_outputs=torch.ones_like(u_x), create_graph=False
        )[0]

        u_xx_np = u_xx.detach().cpu().numpy().flatten()
        f_np = f(x_test)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # =========================================================
        # (0,0) Solution: u_exact (solid) and u_NN (dashed on top)
        # =========================================================
        ax = axes[0, 0]
        if have_exact:
            ax.plot(
                x_test,
                u_true,
                "b-",
                linewidth=2,
                label=r"$u_{\mathrm{exact}}(x)$",
                zorder=1,
            )
        ax.plot(
            x_test,
            u_pred,
            "r--",
            linewidth=2,
            label=r"$u_{\mathrm{NN}}(x)$",
            zorder=2,
        )
        ax.set_title("Solution of BVP")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # =========================================================
        # (0,1) PDE comparison: f (solid) and u_NN'' (dashed on top)
        # =========================================================
        ax = axes[0, 1]
        ax.plot(
            x_test,
            f_np,
            "k-",
            linewidth=2,
            label=r"$f(x)$",
            zorder=1,
        )
        ax.plot(
            x_test,
            u_xx_np,
            "g--",
            linewidth=2,
            label=r"$u_{\mathrm{NN}}''(x)$",
            zorder=2,
        )
        ax.set_title(r"Comparison of $u_{\mathrm{NN}}''(x)$ and $f(x)$")
        ax.set_xlabel("x")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # =========================================================
        # (1,0) Training losses: total (solid), PDE/BC (dashed on top)
        # =========================================================
        ax = axes[1, 0]
        if self.losses:
            epochs_arr = np.arange(1, len(self.losses) + 1)
            ax.semilogy(
                epochs_arr,
                self.losses,
                "k-",
                linewidth=1,
                label="Total loss",
                zorder=3,
            )
            ax.semilogy(
                epochs_arr,
                self.pde_losses,
                "b--",
                linewidth=1,
                label="PDE loss",
                zorder=2,
            )
            ax.semilogy(
                epochs_arr,
                self.bc_losses,
                "g--",
                linewidth=1,
                label="BC loss",
                zorder=1,
            )
        if self.val_losses:
            epochs_val = np.arange(1, len(self.val_losses) + 1)
            ax.semilogy(
                epochs_val,
                self.val_losses,
                "r:",
                linewidth=1,
                label="Val loss",
                zorder=4,
            )
        ax.set_title("Training losses (log scale)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # =========================================================
        # (1,1) L2 norms: PDE residual (solid) and solution error (dashed on top)
        # =========================================================
        ax = axes[1, 1]
        if self.pde_l2_errors:
            epochs_arr = np.arange(1, len(self.pde_l2_errors) + 1)
            ax.semilogy(
                epochs_arr,
                self.pde_l2_errors,
                "r-",
                linewidth=2,
                label=r"$\|u''_{\mathrm{NN}} - f\|_{L^2}$",
                zorder=1,
            )

        if self.solution_l2_errors:
            sol_errors = np.array(self.solution_l2_errors, dtype=float)
            if np.isfinite(sol_errors).any():
                epochs_arr2 = np.arange(1, len(sol_errors) + 1)
                ax.semilogy(
                    epochs_arr2,
                    sol_errors,
                    "b--",
                    linewidth=2,
                    label=r"$\|u_{\mathrm{NN}} - u_{\mathrm{exact}}\|_{L^2}$",
                    zorder=2,
                )

        ax.set_title(r"$L^2$ norms over epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$L^2$ norm")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        out_path = os.path.join(FIG_DIR, "pinn_bvpsolver_l2_SSBroyden.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {out_path}")

        if have_exact:
            print(f"Max |u_NN - u_exact|:  {abs_error.max():.4e}")
            print(f"Mean |u_NN - u_exact|: {abs_error.mean():.4e}")
        if self.pde_l2_errors:
            print(f"Final ||u'' - f||_L2: {self.pde_l2_errors[-1]:.4e}")
        if self.solution_l2_errors and np.isfinite(self.solution_l2_errors[-1]):
            print(f"Final ||u_NN - u_exact||_L2: {self.solution_l2_errors[-1]:.4e}")

    # ------------------- utility: approximant, saving, loading -------------------
    def get_approximant(self):
        def NN(x):
            arr = np.asarray(x, dtype=float)
            x_tensor = torch.from_numpy(arr.reshape(-1, 1)).float().to(device)
            with torch.no_grad():
                y_tensor = self._u_hat(x_tensor)
            y_numpy = y_tensor.cpu().numpy().reshape(-1)
            return y_numpy if arr.ndim > 0 else float(y_numpy.item())

        return NN

    def save_model(self, filepath="../models/pinn_bvp_model_ssbroyden.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="../models/pinn_bvp_model_ssbroyden.pth"):
        state_dict = torch.load(filepath, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        print(f"Model loaded from {filepath}")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    # Mirror the BFGS script exactly: same architecture, same protocol, identity loss.
    torch.manual_seed(7)
    np.random.seed(7)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)

    model = NeuralNetwork(hidden_layers=[32, 32, 32], activation=nn.Tanh())
    print("\nNeural Network Architecture:\n")
    print(model, "\n")

    pinn = PINN_BVP_Solver(model, lr=1e-3, lambda_bc=10.0, lambda_pde=1)

    pinn.train(
        n_epochs=5000,
        n_collocation_points=500,
        verbose_freq=1000,
        patience=5000,
        min_delta=1e-7,
        moving_avg_window=20,
        pde_l2_points=2000,
        train_split=0.7,
        scheduler_patience=5000,
        scheduler_threshold=1e-4,
        scheduler_gamma=0.9,
        scheduler_min_lr=1e-6,
        resample_every=500,
        adam_epochs=2000,
    )

    pinn.plot_results()

    pinn.save_model("../models/pinn_bvp_model_ssbroyden.pth")
    pinn.load_model("../models/pinn_bvp_model_ssbroyden.pth")

    NN = pinn.get_approximant()
    test_x = np.linspace(a, b, 5)
    print("\nSample approximant outputs:")
    print(f"x: {test_x}")
    print(f"NN(x): {NN(test_x)}")
    print(f"u_exact(x): {u_exact(test_x)}")
    print(f"|NN(x) - u_exact(x)|: {np.abs(NN(test_x) - u_exact(test_x))}")

    return model, pinn


if __name__ == "__main__":
    model, pinn = main()
