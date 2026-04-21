import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------------------------
# DEVICE
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# PROBLEM DEFINITION: Poisson in 2D on [ax,bx]x[ay,by]
# u_xx + u_yy = f(x,y), Dirichlet BC on the boundary
# Example exact solution: u(x,y) = sin(pi x) sin(pi y)
# Then f(x,y) = -2 pi^2 sin(pi x) sin(pi y)
# with u = 0 on the boundary of the unit square.
# -------------------------------------------------------------------------
interval_x = [0.0, 1.0]
interval_y = [0.0, 1.0]
ax, bx = interval_x
ay, by = interval_y


def f(xy):
    if isinstance(xy, torch.Tensor):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        return -2.0 * (np.pi**2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    else:
        x = xy[:, 0]
        y = xy[:, 1]
        return -2.0 * (np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)


def u_exact(xy):
    if isinstance(xy, torch.Tensor):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        return torch.sin(np.pi * x) * torch.sin(np.pi * y)
    else:
        x = xy[:, 0]
        y = xy[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)


# -------------------------------------------------------------------------
# NEURAL NETWORK
# -------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers=[64, 64, 64], activation=nn.Tanh()):
        super(NeuralNetwork, self).__init__()

        layers = []
        input_dim = 2

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# -------------------------------------------------------------------------
# BFGS OPTIMIZER (FROM SCRATCH)
# -------------------------------------------------------------------------
class BFGSOptimizer(optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        line_search=True,
        c1=1e-4,
        tau=0.5,
        max_ls=20,
        damping=1e-10,
    ):
        defaults = dict(
            lr=lr,
            line_search=line_search,
            c1=c1,
            tau=tau,
            max_ls=max_ls,
            damping=damping,
        )
        super().__init__(params, defaults)
        self.H = None

    def _get_param_vector(self):
        return torch.cat(
            [p.data.view(-1) for group in self.param_groups for p in group["params"]]
        )

    def _set_param_vector(self, vec):
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                p.data.copy_(vec[offset : offset + numel].view_as(p))
                offset += numel

    def _get_grad_vector(self):
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p.data).view(-1))
                else:
                    grads.append(p.grad.data.view(-1))
        return torch.cat(grads)

    def step(self, closure, loss_eval):
        loss = closure()
        g = self._get_grad_vector().detach()
        x = self._get_param_vector().detach()

        n_params = g.numel()
        if self.H is None or self.H.shape[0] != n_params:
            self.H = torch.eye(n_params, device=g.device, dtype=g.dtype)

        p_dir = -self.H.matmul(g)
        group = self.param_groups[0]
        lr = group["lr"]
        line_search = group["line_search"]
        c1 = group["c1"]
        tau = group["tau"]
        max_ls = group["max_ls"]
        damping = group["damping"]
        alpha = lr

        if line_search:
            gTp = torch.dot(g, p_dir).item()
            f0 = loss.item()
            for _ in range(max_ls):
                new_x = x + alpha * p_dir
                self._set_param_vector(new_x)
                f_new = loss_eval().item()
                if f_new <= f0 + c1 * alpha * gTp:
                    break
                alpha *= tau
            else:
                alpha = 0.0

        s = alpha * p_dir
        new_x = x + s
        self._set_param_vector(new_x)

        new_loss = closure()
        g_new = self._get_grad_vector().detach()
        y = g_new - g
        ys = torch.dot(y, s)

        if ys > damping:
            rho = 1.0 / ys
            I = torch.eye(n_params, device=g.device, dtype=g.dtype)
            syT = torch.outer(s, y)
            ysT = torch.outer(y, s)
            self.H = (I - rho * syT) @ self.H @ (I - rho * ysT) + rho * torch.outer(
                s, s
            )
        else:
            self.H = torch.eye(n_params, device=g.device, dtype=g.dtype)

        return new_loss


# -------------------------------------------------------------------------
# PINN FOR 2D BVP: ENFORCE u_xx + u_yy ~ f
# -------------------------------------------------------------------------
class PINN_BVP2D_Solver:
    def __init__(self, model, lr=1e-3, lambda_pde=1.0):
        """
        Parameters
        ----------
        model : nn.Module
            Neural network u_theta(x,y).
        lr : float
            Learning rate.
        lambda_pde : float
            Weight of PDE loss in the total loss.
        """
        self.model = model.to(device)
        self.adam_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.bfgs_optimizer = BFGSOptimizer(self.model.parameters(), lr=lr)
        self.optimizer = self.adam_optimizer
        self.lambda_pde = lambda_pde

        self.losses = []  # total loss (PDE)
        self.val_losses = []  # validation loss
        self.pde_losses = []  # PDE residual loss
        self.pde_l2_errors = []  # L2 norm of PDE residual
        self.solution_l2_errors = []  # L2 norm of solution error (u_NN - u_exact)

        self.best_model_state = None
        self.best_loss = float("inf")

    # ------------------- hard-enforced solution -------------------
    def _u_hat(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        factor = (x - ax) * (bx - x) * (y - ay) * (by - y)
        return factor * self.model(xy)

    # ------------------- core loss components -------------------
    def _pde_residual(self, xy_interior):
        xy_interior = xy_interior.to(device)
        xy_interior.requires_grad_(True)

        u = self._u_hat(xy_interior)

        grads = torch.autograd.grad(
            u, xy_interior, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, xy_interior, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            u_y, xy_interior, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0][:, 1:2]

        f_vals = f(xy_interior)
        r = u_xx + u_yy - f_vals
        return r

    def compute_loss(self, xy_interior=None, n_collocation_points=1000):
        """Compute total loss = lambda_pde * PDE-loss.

        Notes
        -----
        If `xy_interior` is reused across epochs (as in block-resampling), we create a fresh
        leaf tensor each call via detach().clone().requires_grad_(True). This prevents
        accumulating gradients on the collocation-point tensor itself.
        """

        if xy_interior is None:
            x = torch.empty(n_collocation_points, 1, device=device).uniform_(ax, bx)
            y = torch.empty(n_collocation_points, 1, device=device).uniform_(ay, by)
            xy_interior = torch.cat([x, y], dim=1)
        else:
            xy_interior = xy_interior.to(device)

        xy_interior = xy_interior.detach().clone().requires_grad_(True)

        r = self._pde_residual(xy_interior)
        loss_pde = torch.mean(r**2) * (bx - ax) * (by - ay)

        loss_total = self.lambda_pde * loss_pde
        return loss_total, loss_pde.detach()

    # ------------------- diagnostics -------------------
    def compute_pde_l2_norm(self, n_points=50):
        x = np.linspace(ax, bx, n_points)
        y = np.linspace(ay, by, n_points)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)

        xy_torch = torch.FloatTensor(xy).to(device)
        xy_torch.requires_grad_(True)

        u = self._u_hat(xy_torch)
        grads = torch.autograd.grad(
            u, xy_torch, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_xx = torch.autograd.grad(
            u_x, xy_torch, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            u_y, xy_torch, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0][:, 1:2]

        u_xx_np = u_xx.detach().cpu().numpy().reshape(X.shape)
        u_yy_np = u_yy.detach().cpu().numpy().reshape(X.shape)
        f_np = f(xy).reshape(X.shape)
        residual = u_xx_np + u_yy_np - f_np

        l2_norm = np.sqrt(np.trapz(np.trapz(residual**2, x, axis=1), y, axis=0))
        return l2_norm

    def compute_solution_l2_norm(self, n_points=50):
        x = np.linspace(ax, bx, n_points)
        y = np.linspace(ay, by, n_points)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)

        try:
            u_true = u_exact(xy).reshape(X.shape)
        except Exception:
            return None

        xy_torch = torch.FloatTensor(xy).to(device)
        with torch.no_grad():
            u_pred = self._u_hat(xy_torch).cpu().numpy().reshape(X.shape)

        diff = u_pred - u_true
        l2_norm = np.sqrt(np.trapz(np.trapz(diff**2, x, axis=1), y, axis=0))
        return l2_norm

    # ------------------- training -------------------
    def train(
        self,
        n_epochs=20000,
        n_collocation_points=2000,
        verbose_freq=1000,
        patience=500,
        min_delta=1e-7,
        moving_avg_window=20,
        pde_l2_points=50,
        train_split=0.7,
        scheduler_patience=200,
        scheduler_threshold=1e-4,
        scheduler_gamma=0.9,
        scheduler_min_lr=1e-6,
        resample_every=500,
        adam_epochs=2000,
    ):
        print("\nStarting PINN training for 2D Poisson...")
        print(f"Domain: [{ax}, {bx}] x [{ay}, {by}]")
        print(f"lambda_pde = {self.lambda_pde}")
        print("-" * 60)

        self.best_model_state = None
        self.best_loss = float("inf")
        self.losses.clear()
        self.val_losses.clear()
        self.pde_losses.clear()
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
            x = torch.empty(n_collocation_points, 1, device=device).uniform_(ax, bx)
            y = torch.empty(n_collocation_points, 1, device=device).uniform_(ay, by)
            xy_all = torch.cat([x, y], dim=1)
            perm = torch.randperm(n_collocation_points, device=device)
            xy_all = xy_all[perm]
            xy_train_block = xy_all[:n_train].detach().clone()
            xy_val_block = xy_all[n_train:].detach().clone()
            return xy_train_block, xy_val_block

        xy_train_block, xy_val_block = resample_collocation_block()

        self.adam_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.adam_optimizer,
            mode="min",
            factor=scheduler_gamma,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            verbose=True,
            min_lr=scheduler_min_lr,
        )
        self.bfgs_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.bfgs_optimizer,
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
                xy_train_block, xy_val_block = resample_collocation_block()

            if epoch <= adam_epochs:
                self.optimizer = self.adam_optimizer
                scheduler = self.adam_scheduler
            else:
                self.optimizer = self.bfgs_optimizer
                scheduler = self.bfgs_scheduler

            if self.optimizer is self.adam_optimizer:
                self.optimizer.zero_grad()
                loss_total, loss_pde = self.compute_loss(
                    xy_interior=xy_train_block,
                    n_collocation_points=n_collocation_points,
                )
                loss_total.backward()
                self.optimizer.step()
            else:
                loss_parts = {}

                def closure():
                    self.optimizer.zero_grad()
                    loss_total, loss_pde = self.compute_loss(
                        xy_interior=xy_train_block,
                        n_collocation_points=n_collocation_points,
                    )
                    loss_parts["pde"] = loss_pde
                    loss_total.backward()
                    return loss_total

                def loss_eval():
                    loss_total, _ = self.compute_loss(
                        xy_interior=xy_train_block,
                        n_collocation_points=n_collocation_points,
                    )
                    return loss_total

                loss_total = self.optimizer.step(closure, loss_eval)
                loss_pde = loss_parts["pde"]

            with torch.set_grad_enabled(True):
                val_loss, _ = self.compute_loss(
                    xy_interior=xy_val_block,
                    n_collocation_points=xy_val_block.shape[0],
                )

            loss_value = loss_total.item()
            self.losses.append(loss_value)
            self.pde_losses.append(loss_pde.item())
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
                print(
                    f"Epoch {epoch:6d} | "
                    f"Loss: {loss_value:.4e} | "
                    f"Val: {val_loss.item():.4e} | "
                    f"PDE: {loss_pde.item():.4e} | "
                    f"LR: {current_lr:.2e} | "
                    f"||Lap(u)-f||_L2: {pde_l2:.4e} "
                    f"||u_NN-u_exact||_L2: {sol_l2:.4e}"
                )

            if epoch > adam_epochs and epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"No improvement in moving average loss for {patience} epochs.")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model with moving-average loss: {self.best_loss:.6e}")

        print("-" * 60)
        print(f"Training completed: {actual_epochs} / {n_epochs} epochs.")

    # ------------------- post-processing -------------------
    def plot_results(self, n_plot_points=60):
        x = np.linspace(ax, bx, n_plot_points)
        y = np.linspace(ay, by, n_plot_points)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)

        xy_torch = torch.FloatTensor(xy).to(device)
        with torch.no_grad():
            u_pred = self._u_hat(xy_torch).cpu().numpy().reshape(X.shape)

        try:
            u_true = u_exact(xy).reshape(X.shape)
            have_exact = True
            abs_error = np.abs(u_pred - u_true)
        except Exception:
            u_true = None
            have_exact = False
            abs_error = None

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax0 = axes[0, 0]
        im0 = ax0.contourf(X, Y, u_pred, levels=30, cmap="viridis")
        fig.colorbar(im0, ax=ax0)
        ax0.set_title("u_NN(x,y)")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")

        ax1 = axes[0, 1]
        if have_exact:
            im1 = ax1.contourf(X, Y, u_true, levels=30, cmap="viridis")
            fig.colorbar(im1, ax=ax1)
            ax1.set_title("u_exact(x,y)")
        else:
            ax1.text(0.5, 0.5, "No exact solution", ha="center", va="center")
            ax1.set_title("u_exact(x,y)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax2 = axes[1, 0]
        if self.losses:
            epochs_arr = np.arange(1, len(self.losses) + 1)
            ax2.semilogy(epochs_arr, self.losses, "k-", linewidth=1, label="Total")
            ax2.semilogy(epochs_arr, self.pde_losses, "b--", linewidth=1, label="PDE")
        if self.val_losses:
            epochs_val = np.arange(1, len(self.val_losses) + 1)
            ax2.semilogy(epochs_val, self.val_losses, "r:", linewidth=1, label="Val")
        ax2.set_title("Training losses (log scale)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = axes[1, 1]
        if have_exact:
            im3 = ax3.contourf(X, Y, abs_error, levels=30, cmap="magma")
            fig.colorbar(im3, ax=ax3)
            ax3.set_title("|u_NN - u_exact|")
        else:
            ax3.text(0.5, 0.5, "No exact solution", ha="center", va="center")
            ax3.set_title("|u_NN - u_exact|")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")

        plt.tight_layout()
        plt.show()

        if have_exact:
            print(f"Max |u_NN - u_exact|:  {abs_error.max():.4e}")
            print(f"Mean |u_NN - u_exact|: {abs_error.mean():.4e}")
        if self.pde_l2_errors:
            print(f"Final ||Lap(u) - f||_L2: {self.pde_l2_errors[-1]:.4e}")
        if self.solution_l2_errors and np.isfinite(self.solution_l2_errors[-1]):
            print(
                f"Final ||u_NN - u_exact||_L2: {self.solution_l2_errors[-1]:.4e}"
            )

    # ------------------- utility: approximant, saving, loading -------------------
    def get_approximant(self):
        def NN(xy):
            arr = np.asarray(xy, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            x_tensor = torch.from_numpy(arr).float().to(device)
            with torch.no_grad():
                y_tensor = self._u_hat(x_tensor)
            y_numpy = y_tensor.cpu().numpy().reshape(-1)
            return y_numpy if arr.ndim > 0 else float(y_numpy.item())

        return NN

    def save_model(self, filepath="../models/pinn_bvp2d_model.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="../models/pinn_bvp2d_model.pth"):
        state_dict = torch.load(filepath, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        print(f"Model loaded from {filepath}")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    model = NeuralNetwork(hidden_layers=[64, 64, 64], activation=nn.Tanh())
    print("\nNeural Network Architecture:\n")
    print(model, "\n")

    pinn = PINN_BVP2D_Solver(model, lr=1e-3, lambda_pde=1.0)

    pinn.train(
        n_epochs=20000,
        n_collocation_points=4000,
        verbose_freq=1000,
        patience=200,
        min_delta=1e-7,
        moving_avg_window=20,
        pde_l2_points=60,
        train_split=0.7,
        scheduler_patience=300,
        scheduler_threshold=1e-4,
        scheduler_gamma=0.9,
        scheduler_min_lr=1e-6,
        resample_every=500,
        adam_epochs=2000,
    )

    pinn.plot_results()

    pinn.save_model("../models/pinn_bvp2d_model.pth")
    pinn.load_model("../models/pinn_bvp2d_model.pth")

    NN = pinn.get_approximant()
    test_xy = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.1]])
    print("\nSample approximant outputs:")
    print(f"xy: {test_xy}")
    print(f"NN(xy): {NN(test_xy)}")
    print(f"u_exact(xy): {u_exact(test_xy)}")
    print(f"|NN(xy) - u_exact(xy)|: {np.abs(NN(test_xy) - u_exact(test_xy))}")

    return model, pinn


if __name__ == "__main__":
    model, pinn = main()
