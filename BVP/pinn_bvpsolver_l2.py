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
# PROBLEM DEFINITION: u''(x) = f(x) on [a,b], u(a)=alpha, u(b)=beta
# -------------------------------------------------------------------------
# Example: exact solution u(x) = sin(pi x)
# Then u''(x) = -pi^2 sin(pi x), with u(0)=0, u(1)=0

interval = [0.25, 1]
a, b = interval

def f(x):
    if isinstance(x, torch.Tensor):
        return 2.0 + x - x
    else:
        return 2.0 + x - x


def u_exact(x):
    return (x-1/2)**2

alpha, beta = u_exact(a), u_exact(b)   # boundary values u(a), u(b)


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
    def __init__(self, model, lr=1e-3, lambda_bc=10.0, lambda_pde = 1.0):
        """
        Parameters
        ----------
        model : nn.Module
            Neural network u_theta(x).
        lr : float
            Learning rate.
        lambda_bc : float
            Weight of boundary-condition loss in the total loss.
        """
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lambda_bc = lambda_bc
        self.lambda_pde = lambda_pde

        self.losses = []             # total loss (PDE + BC)
        self.pde_losses = []         # PDE residual loss
        self.bc_losses = []          # boundary loss
        self.pde_l2_errors = []      # L2 norm of PDE residual
        self.solution_l2_errors = [] # L2 norm of solution error (u_NN - u_exact)

        self.best_model_state = None
        self.best_loss = float("inf")

    # ------------------- core loss components -------------------
    def _pde_residual(self, x_interior):
        x_interior = x_interior.to(device)
        x_interior.requires_grad_(True)

        u = self.model(x_interior)
        # First derivative u_x
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
            create_graph=True
        )[0]

        # Right-hand side f(x)
        f_vals = f(x_interior)
        r = u_xx - f_vals
        return r

    def compute_loss(self, n_collocation_points=100):
        # Interior collocation points
        x_interior = torch.FloatTensor(n_collocation_points, 1).uniform_(a, b).to(device)
        x_interior.requires_grad_(True)

        r = self._pde_residual(x_interior)
        loss_pde = torch.mean(r ** 2) * (b - a)  # scale by interval length

        # Boundary points (fixed)
        x_bc = torch.tensor([[a], [b]], dtype=torch.float32, device=device)
        u_bc = self.model(x_bc)
        target_bc = torch.tensor([[alpha], [beta]], dtype=torch.float32, device=device)
        loss_bc = torch.mean((u_bc - target_bc) ** 2)

        loss_total = self.lambda_pde * loss_pde + self.lambda_bc * loss_bc
        return loss_total, loss_pde.detach(), loss_bc.detach()

    # ------------------- diagnostics -------------------
    def compute_pde_l2_norm(self, n_points=500):
        x_test = np.linspace(a, b, n_points)
        x_torch = torch.FloatTensor(x_test.reshape(-1, 1)).to(device)
        x_torch.requires_grad_(True)

        u = self.model(x_torch)

        u_x = torch.autograd.grad(
            u,
            x_torch,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            x_torch,
            grad_outputs=torch.ones_like(u_x),
            create_graph=False
        )[0]

        u_xx_np = u_xx.detach().cpu().numpy().flatten()
        f_np = f(x_test)
        residual = u_xx_np - f_np

        l2_norm = np.sqrt(np.trapz(residual ** 2, x_test))
        return l2_norm

    def compute_solution_l2_norm(self, n_points=500):
        x_test = np.linspace(a, b, n_points)

        # Try to evaluate exact solution
        try:
            u_true = u_exact(x_test)
        except Exception:
            return None

        # NN approximation
        x_torch = torch.FloatTensor(x_test.reshape(-1, 1)).to(device)
        with torch.no_grad():
            u_pred = self.model(x_torch).cpu().numpy().flatten()

        diff = u_pred - u_true
        l2_norm = np.sqrt(np.trapz(diff**2, x_test))
        return l2_norm

    # ------------------- training -------------------
    def train(self, n_epochs=20000, n_collocation_points=200, verbose_freq=1000, patience=500, 
              min_delta=1e-7, moving_avg_window=20, pde_l2_points=200):
        print("\nStarting PINN training for BVP u''(x)=f(x)...")
        print(f"Domain: [{a}, {b}], BC: u(a)={alpha}, u(b)={beta}")
        print(f"lambda_bc = {self.lambda_bc}")
        print(f"lambda_pde = {self.lambda_pde}")
        print("-" * 60)

        epochs_without_improvement = 0
        moving_avg_losses = []
        actual_epochs = 0

        for epoch in range(1, n_epochs + 1):
            self.optimizer.zero_grad()
            loss_total, loss_pde, loss_bc = self.compute_loss(
                n_collocation_points=n_collocation_points
            )
            loss_total.backward()
            self.optimizer.step()

            # Log
            loss_value = loss_total.item()
            self.losses.append(loss_value)
            self.pde_losses.append(loss_pde.item())
            self.bc_losses.append(loss_bc.item())

            # PDE residual L2 norm for monitoring
            pde_l2 = self.compute_pde_l2_norm(n_points=pde_l2_points)
            self.pde_l2_errors.append(pde_l2)

            # Solution L2 norm (if exact solution available)
            sol_l2 = self.compute_solution_l2_norm(n_points=pde_l2_points)
            if sol_l2 is not None:
                self.solution_l2_errors.append(sol_l2)
            else:
                self.solution_l2_errors.append(np.nan)

            actual_epochs = epoch

            # Moving average for early stopping
            moving_avg_losses.append(loss_value)
            if len(moving_avg_losses) > moving_avg_window:
                moving_avg_losses.pop(0)
            moving_avg = np.mean(moving_avg_losses)

            # Early stopping check on moving average
            if moving_avg + min_delta < self.best_loss:
                self.best_loss = moving_avg
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Verbose output
            if epoch % verbose_freq == 0:
                print(
                    f"Epoch {epoch:6d} | "
                    f"Loss: {loss_value:.4e} | "
                    f"PDE: {loss_pde.item():.4e} | "
                    f"BC: {loss_bc.item():.4e} | "
                    f"||u''-f||_L2: {pde_l2:.4e}"
                    f"||u_NN-u_exact||_L2: {sol_l2:.4e}"
                )

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"No improvement in moving average loss for {patience} epochs.")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model with moving-average loss: {self.best_loss:.6e}")

        print("-" * 60)
        print(f"Training completed: {actual_epochs} / {n_epochs} epochs.")

    # ------------------- post-processing -------------------
    def plot_results(self, n_plot_points=200):
        # --- grid for plotting ---
        x_test = np.linspace(a, b, n_plot_points)

        # Torch tensor with gradient tracking for derivatives
        x_torch = torch.tensor(
            x_test.reshape(-1, 1),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        # Forward pass u_NN(x)
        u = self.model(x_torch)                 # shape (N,1)
        u_pred = u.detach().cpu().numpy().flatten()

        # Exact solution (if available)
        try:
            u_true = u_exact(x_test)
            have_exact = True
            abs_error = np.abs(u_pred - u_true)
        except Exception:
            u_true = None
            have_exact = False
            abs_error = None

        # First derivative u_x
        u_x = torch.autograd.grad(
            u,
            x_torch,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        # Second derivative u_xx
        u_xx = torch.autograd.grad(
            u_x,
            x_torch,
            grad_outputs=torch.ones_like(u_x),
            create_graph=False
        )[0]

        u_xx_np = u_xx.detach().cpu().numpy().flatten()
        f_np = f(x_test)

        # --- set up 2x2 figure ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # =========================================================
        # (0,0) Solution: u_exact (solid) and u_NN (dashed on top)
        # =========================================================
        ax = axes[0, 0]
        if have_exact:
            ax.plot(
                x_test, u_true,
                "b-",
                linewidth=2,
                label=r"$u_{\mathrm{exact}}(x)$",
                zorder=1,
            )
        ax.plot(
            x_test, u_pred,
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
            x_test, f_np,
            "k-",
            linewidth=2,
            label=r"$f(x)$",
            zorder=1,
        )
        ax.plot(
            x_test, u_xx_np,
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
        plt.show()

        # --- print summary diagnostics ---
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
                y_tensor = self.model(x_tensor)
            y_numpy = y_tensor.cpu().numpy().reshape(-1)
            return y_numpy if arr.ndim > 0 else float(y_numpy.item())
        return NN

    def save_model(self, filepath="models/pinn_bvp_model.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="models/pinn_bvp_model.pth"):
        state_dict = torch.load(filepath, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        print(f"Model loaded from {filepath}")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    model = NeuralNetwork(hidden_layers=[32, 32, 32], activation=nn.Sigmoid())
    print("\nNeural Network Architecture:\n")
    print(model, "\n")

    pinn = PINN_BVP_Solver(model, lr=1e-3, lambda_bc=10.0, lambda_pde = 1)

    pinn.train(
        n_epochs=20000,
        n_collocation_points=200,
        verbose_freq=1000,
        patience=200,
        min_delta=1e-7,
        moving_avg_window=20,
        pde_l2_points=2000,
    )

    pinn.plot_results()

    # Save & reload example
    pinn.save_model("models/pinn_bvp_model.pth")
    pinn.load_model("models/pinn_bvp_model.pth")

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
