import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# DEVICE
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# PROBLEM DEFINITION: u''(x) = f(x) on [a,b], u(a)=alpha, u(b)=beta
# -------------------------------------------------------------------------
# Example (edit as needed):
interval = [0.25, 1.0]
a, b = interval

def f(x):
    # RHS f(x) (supports torch.Tensor and numpy)
    if isinstance(x, torch.Tensor):
        return 2.0 + x - x
    return 2.0 + x - x

def u_exact(x):
    # Exact solution (optional; used only for diagnostics if provided)
    return (x - 0.5) ** 2

alpha, beta = u_exact(a), u_exact(b)

# -------------------------------------------------------------------------
# NEURAL NETWORK
# -------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers=[32, 32, 32], activation=nn.Tanh()):
        super().__init__()
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
        """PINN solver for u''(x)=f(x), with Dirichlet BCs u(a)=alpha, u(b)=beta."""
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lambda_bc = float(lambda_bc)
        self.lambda_pde = float(lambda_pde)

        # Training history
        self.losses = []             # total training loss
        self.pde_losses = []         # training PDE loss
        self.bc_losses = []          # training BC loss
        self.val_losses = []         # total validation loss
        self.val_pde_losses = []     # validation PDE loss
        self.val_bc_losses = []      # validation BC loss
        self.lr_history = []         # learning rate per epoch

        # Diagnostics
        self.pde_l2_errors = []      # L2 norm of PDE residual ||u''-f||_L2
        self.solution_l2_errors = [] # L2 norm of solution error ||u_NN-u_exact||_L2

        # Best model checkpointing
        self.best_loss = np.inf
        self.best_model_state = None

    # ------------------- helpers -------------------
    @staticmethod
    def _normalize_split(train_frac, val_frac=None):
        """Accepts either fractions in [0,1] or percentages in [0,100]."""
        if val_frac is None:
            val_frac = 1.0 - train_frac

        # If user passed percentages
        if train_frac > 1.0 or val_frac > 1.0:
            train_frac = train_frac / 100.0
            val_frac = val_frac / 100.0

        total = train_frac + val_frac
        if not np.isfinite(total) or total <= 0:
            raise ValueError("train_frac and val_frac must sum to a positive value.")
        train_frac = train_frac / total
        val_frac = val_frac / total
        return float(train_frac), float(val_frac)

    def _build_train_val_points(
        self,
        n_points,
        train_frac=0.8,
        val_frac=None,
        seed=1234,
        distribution="grid",
    ):
        """Creates fixed train/val collocation points with uniform coverage on [a,b]."""
        train_frac, val_frac = self._normalize_split(train_frac, val_frac)

        rng = np.random.default_rng(seed)

        if distribution == "grid":
            x_all = np.linspace(a, b, int(n_points))
        elif distribution == "random_uniform":
            # i.i.d. uniform on [a,b]
            x_all = rng.uniform(a, b, size=int(n_points))
            x_all.sort()
        else:
            raise ValueError("distribution must be 'grid' or 'random_uniform'.")

        idx = np.arange(len(x_all))
        rng.shuffle(idx)

        n_train = int(np.round(train_frac * len(x_all)))
        n_train = max(1, min(n_train, len(x_all) - 1))  # keep both non-empty

        train_idx = idx[:n_train]
        val_idx = idx[n_train:]

        x_train = torch.tensor(x_all[train_idx].reshape(-1, 1), dtype=torch.float32, device=device)
        x_val = torch.tensor(x_all[val_idx].reshape(-1, 1), dtype=torch.float32, device=device)
        return x_train, x_val, train_frac, val_frac

    # ------------------- PDE residual -------------------
    def _pde_residual(self, x_interior, create_graph=True):
        """Returns r(x)=u_xx(x)-f(x)."""
        x_interior = x_interior.to(device)
        x_interior.requires_grad_(True)

        u = self.model(x_interior)

        # First derivative u_x (must be create_graph=True to allow u_xx)
        u_x = torch.autograd.grad(
            u,
            x_interior,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        # Second derivative u_xx
        u_xx = torch.autograd.grad(
            u_x,
            x_interior,
            grad_outputs=torch.ones_like(u_x),
            create_graph=create_graph
        )[0]

        f_vals = f(x_interior)
        return u_xx - f_vals

    def compute_loss_on_points(self, x_interior, create_graph=True):
        """Loss computed on a given set of collocation points."""
        r = self._pde_residual(x_interior, create_graph=create_graph)
        loss_pde = torch.mean(r ** 2) * (b - a)

        # Boundary points (fixed)
        x_bc = torch.tensor([[a], [b]], dtype=torch.float32, device=device)
        u_bc = self.model(x_bc)
        target_bc = torch.tensor([[alpha], [beta]], dtype=torch.float32, device=device)
        loss_bc = torch.mean((u_bc - target_bc) ** 2)

        loss_total = self.lambda_pde * loss_pde + self.lambda_bc * loss_bc
        return loss_total, loss_pde.detach(), loss_bc.detach()

    # Backwards-compatible: random collocation sampling (not used for validation splitting)
    def compute_loss(self, n_collocation_points=100):
        x_interior = torch.empty(int(n_collocation_points), 1, device=device).uniform_(a, b)
        x_interior.requires_grad_(True)
        return self.compute_loss_on_points(x_interior, create_graph=True)

    # ------------------- diagnostics -------------------
    def compute_pde_l2_norm(self, n_points=500):
        x_test = np.linspace(a, b, n_points)
        x_torch = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)

        r = self._pde_residual(x_torch, create_graph=False)
        r_np = r.detach().cpu().numpy().flatten()
        return float(np.sqrt(np.trapz(r_np**2, x_test)))

    def compute_solution_l2_norm(self, n_points=500):
        x_test = np.linspace(a, b, n_points)
        try:
            u_true = u_exact(x_test)
        except Exception:
            return None

        x_torch = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            u_pred = self.model(x_torch).cpu().numpy().flatten()

        diff = u_pred - u_true
        return float(np.sqrt(np.trapz(diff**2, x_test)))

    # ------------------- training -------------------
    def train(
        self,
        n_epochs=20000,
        n_points=400,
        train_frac=0.8,
        val_frac=None,
        split_seed=1234,
        point_distribution="grid",
        verbose_freq=1000,
        # Early stopping:
        patience=500,
        min_delta=1e-7,
        moving_avg_window=20,
        # Scheduler (threshold + patience + multiplicative decay):
        use_scheduler=True,
        scheduler_patience=200,
        scheduler_threshold=1e-4,
        scheduler_decay_factor=0.5,
        scheduler_min_lr=1e-6,
        # Diagnostics:
        pde_l2_points=200,
    ):
        """Train the PINN with a fixed uniform train/validation split."""

        x_train_base, x_val_base, train_frac_n, val_frac_n = self._build_train_val_points(
            n_points=n_points,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=split_seed,
            distribution=point_distribution,
        )

        print("\nStarting PINN training for BVP u''(x)=f(x)...")
        print(f"Domain: [{a}, {b}], BC: u(a)={alpha}, u(b)={beta}")
        print(f"lambda_bc = {self.lambda_bc}")
        print(f"lambda_pde = {self.lambda_pde}")
        print(f"Collocation points: total={n_points}, train={len(x_train_base)} ({train_frac_n:.0%}), val={len(x_val_base)} ({val_frac_n:.0%})")
        print(f"Point distribution: {point_distribution}")
        print("-" * 60)

        scheduler = None
        if use_scheduler:
            # Plateau scheduler: reduces LR by a multiplicative factor (exponential across multiple reductions)
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(scheduler_decay_factor),
                patience=int(scheduler_patience),
                threshold=float(scheduler_threshold),
                threshold_mode="rel",
                min_lr=float(scheduler_min_lr),
                verbose=False,
            )
            print(f"Scheduler: ReduceLROnPlateau(patience={scheduler_patience}, threshold={scheduler_threshold}, factor={scheduler_decay_factor}, min_lr={scheduler_min_lr})")
        else:
            print("Scheduler: disabled")

        epochs_without_improvement = 0
        moving_avg_val = []
        actual_epochs = 0

        for epoch in range(1, int(n_epochs) + 1):
            # ---- Training step on fixed train points ----
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            x_train = x_train_base.clone().detach().requires_grad_(True)
            loss_total, loss_pde, loss_bc = self.compute_loss_on_points(x_train, create_graph=True)
            loss_total.backward()
            self.optimizer.step()

            # ---- Validation loss on fixed val points ----
            self.model.eval()
            x_val = x_val_base.clone().detach().requires_grad_(True)
            val_total, val_pde, val_bc = self.compute_loss_on_points(x_val, create_graph=False)
            val_value = float(val_total.detach().cpu().item())

            # LR history
            lr_now = float(self.optimizer.param_groups[0]["lr"])
            self.lr_history.append(lr_now)

            # Record losses
            self.losses.append(float(loss_total.detach().cpu().item()))
            self.pde_losses.append(float(loss_pde.cpu().item()))
            self.bc_losses.append(float(loss_bc.cpu().item()))
            self.val_losses.append(val_value)
            self.val_pde_losses.append(float(val_pde.cpu().item()))
            self.val_bc_losses.append(float(val_bc.cpu().item()))

            # Diagnostics
            pde_l2 = self.compute_pde_l2_norm(n_points=pde_l2_points)
            self.pde_l2_errors.append(pde_l2)

            sol_l2 = self.compute_solution_l2_norm(n_points=pde_l2_points)
            self.solution_l2_errors.append(sol_l2 if sol_l2 is not None else np.nan)

            actual_epochs = epoch

            # ---- Scheduler step ----
            if scheduler is not None:
                scheduler.step(val_value)

            # ---- Early stopping on validation moving average ----
            moving_avg_val.append(val_value)
            if len(moving_avg_val) > moving_avg_window:
                moving_avg_val.pop(0)
            val_ma = float(np.mean(moving_avg_val))

            if val_ma + min_delta < self.best_loss:
                self.best_loss = val_ma
                self.best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % int(verbose_freq) == 0:
                print(
                    f"Epoch {epoch:6d} | "
                    f"TrainLoss: {self.losses[-1]:.4e} | "
                    f"ValLoss: {self.val_losses[-1]:.4e} | "
                    f"PDE: {self.pde_losses[-1]:.4e} | "
                    f"BC: {self.bc_losses[-1]:.4e} | "
                    f"LR: {lr_now:.2e} | "
                    f"||u''-f||_L2: {pde_l2:.4e} | "
                    f"||u_NN-u_exact||_L2: {self.solution_l2_errors[-1]:.4e}"
                )

            if epochs_without_improvement >= int(patience):
                print(f"Early stopping at epoch {epoch} (no val MA improvement for {patience} epochs).")
                break

        # Restore best model (validation MA)
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        print("-" * 60)
        print(f"Training finished after {actual_epochs} epochs.")
        print(f"Best validation moving-average loss: {self.best_loss:.6e}")
        return actual_epochs

    # ------------------- post-processing -------------------
    def get_approximant(self):
        """Returns a callable NN(x) on numpy arrays."""
        def NN(x_np):
            x_torch = torch.tensor(np.array(x_np).reshape(-1, 1), dtype=torch.float32, device=device)
            with torch.no_grad():
                y = self.model(x_torch).cpu().numpy().flatten()
            return y
        return NN

    def plot_results(self, n_plot_points=200):
        x_test = np.linspace(a, b, n_plot_points)
        x_torch = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)

        # Forward
        u = self.model(x_torch)

        # Derivatives
        u_x = torch.autograd.grad(
            u, x_torch, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_torch, grad_outputs=torch.ones_like(u_x), create_graph=False
        )[0]

        u_np = u.detach().cpu().numpy().flatten()
        uxx_np = u_xx.detach().cpu().numpy().flatten()
        f_np = f(torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32)).detach().cpu().numpy().flatten()

        # Exact (if available)
        try:
            u_true = u_exact(x_test)
        except Exception:
            u_true = None

        # ---- Plot ----
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("PINN BVP Solver Diagnostics", fontsize=16)

        # (0,0) Solution
        ax = axes[0, 0]
        ax.plot(x_test, u_np, label=r"$u_{\mathrm{NN}}(x)$", linewidth=2, zorder=2)
        if u_true is not None:
            ax.plot(x_test, u_true, "k--", label=r"$u_{\mathrm{exact}}(x)$", linewidth=2, zorder=3)
        ax.set_title("Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # (0,1) Second derivative vs f
        ax = axes[0, 1]
        ax.plot(x_test, f_np, "k--", linewidth=2, label=r"$f(x)$", zorder=3)
        ax.plot(x_test, uxx_np, linewidth=2, label=r"$u_{\mathrm{NN}}''(x)$", zorder=2)
        ax.set_title(r"Comparison of $u_{\mathrm{NN}}''(x)$ and $f(x)$")
        ax.set_xlabel("x")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # (1,0) Loss curves
        ax = axes[1, 0]
        if self.losses:
            epochs_arr = np.arange(1, len(self.losses) + 1)
            ax.semilogy(epochs_arr, self.losses, "k-", linewidth=1, label="Train total", zorder=2)
            ax.semilogy(epochs_arr, self.val_losses, "r:", linewidth=2, label="Val total", zorder=4)
            ax.semilogy(epochs_arr, self.pde_losses, "b--", linewidth=1, label="Train PDE", zorder=3)
            ax.semilogy(epochs_arr, self.bc_losses, "g--", linewidth=1, label="Train BC", zorder=3)
        ax.set_title("Losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # (1,1) L2 diagnostics
        ax = axes[1, 1]
        if self.pde_l2_errors:
            ax.semilogy(np.arange(1, len(self.pde_l2_errors) + 1), self.pde_l2_errors, label=r"$\|u''-f\|_{L^2}$", zorder=2)
        if self.solution_l2_errors:
            ax.semilogy(np.arange(1, len(self.solution_l2_errors) + 1), self.solution_l2_errors, "k--", linewidth=2, label=r"$\|u_{NN}-u_{exact}\|_{L^2}$", zorder=3)
        ax.set_title(r"L2 diagnostics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L2 norm (log scale)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    hidden_layers = [32, 32, 32]
    model = NeuralNetwork(hidden_layers=hidden_layers, activation=nn.Tanh())

    pinn = PINN_BVP_Solver(
        model=model,
        lr=1e-3,
        lambda_bc=1e6,
        lambda_pde=1.0
    )

    pinn.train(
        n_epochs=20000,
        n_points=400,           # total points (train+val)
        train_frac=0.8,         # 80% train, 20% val
        point_distribution="grid",  # uniform grid on [a,b]
        verbose_freq=2000,
        patience=1500,
        use_scheduler=True,
        scheduler_patience=300,
        scheduler_threshold=1e-4,
        scheduler_decay_factor=0.5,
        scheduler_min_lr=1e-6,
        pde_l2_points=400,
    )

    pinn.plot_results(n_plot_points=400)

    # Test approximant
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
