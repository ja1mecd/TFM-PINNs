import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import integrate
import os
import inspect


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to approximate using the PINN.
# Thesis section 4.1: "smooth scalar target f(x) on a bounded interval".
f = lambda x: (x - 0.5) ** 2

interval = [-1.0, 1.0]
a = interval[0]
b = interval[1]



# Neural Network definition

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers=[20, 20], activation=nn.ReLU()):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        input_dim = 1
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# -------------------------------------------------------------------------
# TRAINING CLASS
# -------------------------------------------------------------------------
class PINN_L2_Minimizer:
    def __init__(self, model, lr=1e-3):
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.losses = []
        self.l2_errors = []
        self.best_model_state = None
        self.best_loss = float('inf')

    def compute_l2_loss(self, x_batch):
        x_batch = x_batch.to(device)
        nn_output = self.model(x_batch)
        target_values = f(x_batch)
        loss = torch.mean((nn_output - target_values)**2) * (b-a)  # scale by interval length
        return loss

    def compute_exact_l2_norm(self, n_points=1000):
        """Compute L2 error via dense trapezoidal quadrature."""
        x_test = np.linspace(a, b, n_points)
        x_test_torch = torch.FloatTensor(x_test.reshape(-1, 1)).to(device)

        with torch.no_grad():
            nn_values_torch = self.model(x_test_torch)
            nn_values = nn_values_torch.cpu().numpy().flatten()
            target_values = f(x_test)
            squared_diff = (nn_values - target_values)**2
            l2_norm = np.sqrt(np.trapz(squared_diff, x_test))
        return l2_norm

    def compute_linf_error(self, n_points=2000):
        """Compute the L-infinity error on a dense validation grid.

        Used by the architecture sweep in `error_table_pinn.py`, which is
        the script that produces the Tanh / Sigmoid / ReLU / Softmax
        heatmaps cited in section 4.1 of the thesis.
        """
        x_test = np.linspace(a, b, n_points)
        x_test_torch = torch.FloatTensor(x_test.reshape(-1, 1)).to(device)

        with torch.no_grad():
            nn_values = self.model(x_test_torch).cpu().numpy().flatten()
            target_values = f(x_test)
            return float(np.max(np.abs(nn_values - target_values)))

    def train(self, n_epochs=5000, n_collocation_points=100, verbose_freq=500, 
              patience=50, min_delta=1e-6, moving_avg_window=10, l2_points=200):
        """
        Train the PINN with early stopping.
        
        Args:
            n_epochs: Maximum number of training epochs
            n_collocation_points: Number of collocation points for training
            verbose_freq: Frequency of printing training progress
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in loss to qualify as improvement
            moving_avg_window: Window size for computing moving average of loss
            l2_points: Number of points for computing L2 error each epoch
        """
        print("\nStarting PINN training to minimize L2 norm...")
        print(f"Target function: {inspect.getsource(f).strip()} on {interval}")
        print(f"Early stopping: patience={patience}, min_delta={min_delta}")
        print("-" * 50)

        epochs_without_improvement = 0
        moving_avg_losses = []
        actual_epochs = 0

        for epoch in range(n_epochs):
            x_collocation = torch.FloatTensor(n_collocation_points, 1).uniform_(a, b).to(device)
            self.optimizer.zero_grad()
            loss = self.compute_l2_loss(x_collocation)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
            actual_epochs = epoch + 1
            l2_norm = self.compute_exact_l2_norm(n_points=l2_points)
            self.l2_errors.append(l2_norm)

            # Track moving average of losses
            moving_avg_losses.append(loss.item())
            if len(moving_avg_losses) > moving_avg_window:
                moving_avg_losses.pop(0)
            
            current_avg_loss = np.mean(moving_avg_losses)

            # Check for improvement and save best model
            if current_avg_loss < self.best_loss - min_delta:
                self.best_loss = current_avg_loss
                self.best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Verbose output
            if (epoch + 1) % verbose_freq == 0:
                print(f"Epoch {epoch + 1:5d} | Loss: {loss.item():.6f} | "
                      f"Avg Loss: {current_avg_loss:.6f} | L2 Norm: {l2_norm:.6f} | "
                      f"No improv: {epochs_without_improvement}/{patience}")

            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Loss hasn't improved for {patience} epochs (min_delta={min_delta})")
                # Load best model
                if self.best_model_state is not None:
                    self.model.load_state_dict(self.best_model_state)
                    print(f"Loaded best model with loss: {self.best_loss:.6f}")
                break

        print("-" * 50)
        print(f"Training completed! Total epochs: {actual_epochs}/{n_epochs}\n")


    def get_approximant(self):
        def NN(x):
            arr = np.asarray(x, dtype=float)
            x_tensor = torch.from_numpy(arr.reshape(-1, 1)).float().to(device)
            with torch.no_grad():
                y_tensor = self.model(x_tensor)
            y_numpy = y_tensor.cpu().numpy().reshape(-1)
            return y_numpy if arr.ndim > 0 else float(y_numpy.item())
        return NN

    # ---------------------------------------------------------------------
    # Saving and loading
    # ---------------------------------------------------------------------
    def save_model(self, path="pinn_model.pth"):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="pinn_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded from {path}")

    # ---------------------------------------------------------------------
    # Plot results with convergence analysis
    # ---------------------------------------------------------------------
    def plot_results(self, save_path: str | None = None, show: bool = True):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        x_test = np.linspace(a, b, 200)
        x_test_torch = torch.FloatTensor(x_test.reshape(-1, 1)).to(device)

        with torch.no_grad():
            nn_predictions = self.model(x_test_torch).cpu().numpy().flatten()

        target_values = f(x_test)
        abs_error = np.abs(nn_predictions - target_values)

        # Plot 1: Function approximation
        axes[0, 0].plot(x_test, target_values, 'b-', label="Target f(x)", linewidth=2)
        axes[0, 0].plot(x_test, nn_predictions, 'r--', label='Neural Network', linewidth=2)
        axes[0, 0].set_title("Function Approximation")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Absolute Error
        axes[0, 1].plot(x_test, abs_error, 'm-', linewidth=2)
        axes[0, 1].set_title("Absolute Error |NN(x) - f(x)|")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("Error")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Loss history
        epochs = np.arange(1, len(self.losses) + 1)
        axes[1, 0].loglog(epochs, self.losses, 'b-', linewidth=2, label="Loss")
        axes[1, 0].set_title("Training Loss & L2 Error")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss (log scale)")
        axes[1, 0].grid(True, alpha=0.3)

        if self.l2_errors:
            ax2 = axes[1, 0].twinx()
            ax2.plot(epochs[:len(self.l2_errors)], self.l2_errors, 'r-', linewidth=1.5, label="L2 Error")
            ax2.set_ylabel("L2 Error")
            lines1, labels1 = axes[1, 0].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[1, 0].legend(lines1 + lines2, labels1 + labels2, loc="best")

        # Plot 4: Loss rate of change (to visualize convergence)
        if len(self.losses) > 1:
            loss_changes = np.abs(np.diff(self.losses))
            axes[1, 1].loglog(loss_changes, 'g-', linewidth=1, alpha=0.7)
            # Add moving average
            window = min(50, len(loss_changes)//10)
            if window > 1:
                moving_avg = np.convolve(loss_changes, np.ones(window)/window, mode='valid')
                axes[1, 1].semilogy(range(window-1, len(loss_changes)), moving_avg, 'r-', 
                                   linewidth=2, label=f'Moving avg (w={window})')
            axes[1, 1].set_title("Loss Rate of Change |Loss[i] - Loss[i-1]|")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Absolute Change")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        plt.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=200)
            print(f"Saved results figure to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        final_l2_norm = self.compute_exact_l2_norm()
        print(f"Final L2 Norm: {final_l2_norm:.6f}")
        print(f"Max Error: {abs_error.max():.6f}")
        print(f"Mean Error: {abs_error.mean():.6f}")

        theoretical_l2 = np.sqrt(integrate.quad(lambda x: f(x)**2, -1, 1)[0])
        print(f"Theoretical L2 of f(x): {theoretical_l2:.6f}")

# -------------------------------------------------------------------------
# MAIN ROUTINE
# -------------------------------------------------------------------------
def main():
    # Winning configuration from the cross-activation architecture sweep
    # (section 4.1 of the thesis): ReLU achieves the lowest L-infinity error
    # at (L = 2, W = 80). Tanh remains the baseline for the PDE experiments
    # in subsequent sections, where derivative regularity is what matters.
    model = NeuralNetwork(hidden_layers=[80, 80], activation=nn.ReLU())
    print("\nNeural Network Architecture:\n")
    print(model, "\n")

    pinn = PINN_L2_Minimizer(model, lr=1e-3)
    
    # Train with early stopping parameters
    pinn.train(
        n_epochs=10000, 
        n_collocation_points=200, 
        verbose_freq=1000,
        patience=100,        # Stop if no improvement for 100 epochs
        min_delta=1e-7,      # Minimum change to consider as improvement
        moving_avg_window=20  # Average over 20 epochs for stability
    )
    
    pinn.plot_results(save_path="figures/pinn_interpolant_l2.png", show=False)

    # Save and test loading
    pinn.save_model("models/pinn_l2_model.pth")
    pinn.load_model("models/pinn_l2_model.pth")

    # Test approximant
    NN = pinn.get_approximant()
    test_x = np.linspace(-1, 1, 5)
    
    print("Approximating the function: f = lambda x: (x-1/2)**2")
    print("Sample approximant outputs:")
    print(f"x: {test_x}")
    print(f"NN(x): {NN(test_x)}")
    print(f"f(x): {f(test_x)}")
    print(f"|f(x)-NN(x)|: {abs(NN(test_x)-f(test_x))}")

    return model, pinn

if __name__ == "__main__":
    model, pinn = main()
