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

# Function to approximate using the PINN
f = lambda x: (x-1/2)**2

interval = [-1, 1]
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

    def compute_l2_loss(self, x_batch):
        x_batch = x_batch.to(device)
        nn_output = self.model(x_batch)
        target_values = f(x_batch)
        loss = torch.mean((nn_output - target_values)**2) * (b-a)  # scale by interval length
        return loss

    def compute_exact_l2_norm(self, n_points=1000):
        """Compute exact L2 norm via dense numerical integration."""
        x_test = np.linspace(a, b, n_points)
        x_test_torch = torch.FloatTensor(x_test.reshape(a, b)).to(device)

        with torch.no_grad():
            nn_values_torch = self.model(x_test_torch)
            nn_values = nn_values_torch.cpu().numpy().flatten()
            target_values = f(x_test)
            squared_diff = (nn_values - target_values)**2
            l2_norm = np.sqrt(np.trapz(squared_diff, x_test))
        return l2_norm

    def train(self, n_epochs=5000, n_collocation_points=100, verbose_freq=500):
        print("\nStarting PINN training to minimize L2 norm...")
        print(f"Target function: {inspect.getsource(f)} on {interval}")
        print("-" * 50)

        for epoch in range(n_epochs):
            x_collocation = torch.FloatTensor(n_collocation_points, 1).uniform_(a, b).to(device)
            self.optimizer.zero_grad()
            loss = self.compute_l2_loss(x_collocation)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())

            if (epoch + 1) % verbose_freq == 0:
                l2_norm = self.compute_exact_l2_norm()
                print(f"Epoch {epoch + 1:5d} | Loss: {loss.item():.6f} | L2 Norm: {l2_norm:.6f}")

        print("-" * 50)
        print("Training completed!\n")


    def get_approximant(self):
        def NN(x):
            arr = np.asarray(x, dtype=float)
            x_tensor = torch.from_numpy(arr.reshape(a, b)).float().to(device)
            with torch.no_grad():
                y_tensor = self.model(x_tensor)
            y_numpy = y_tensor.cpu().numpy().reshape(-1)
            return y_numpy if arr.ndim > 0 else float(y_numpy.item())
        return NN

    # ---------------------------------------------------------------------
    # Saving and loading
    # ---------------------------------------------------------------------
    def save_model(self, path="pinn_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="pinn_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded from {path}")

    # ---------------------------------------------------------------------
    # Plot results
    # ---------------------------------------------------------------------
    def plot_results(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        x_test = np.linspace(a, b, 200)
        x_test_torch = torch.FloatTensor(x_test.reshape(a, b)).to(device)

        with torch.no_grad():
            nn_predictions = self.model(x_test_torch).cpu().numpy().flatten()

        target_values = f(x_test)
        abs_error = np.abs(nn_predictions - target_values)

        # Plot 1: Function approximation
        axes[0].plot(x_test, target_values, 'b-', label=f"Target {inspect.getsource(f)}", linewidth=2)
        axes[0].plot(x_test, nn_predictions, 'r--', label='Neural Network', linewidth=2)
        axes[0].set_title("Function Approximation")
        axes[0].legend()

        # Plot 2: Absolute Error
        axes[1].plot(x_test, abs_error, 'm-', linewidth=2)
        axes[1].set_title("Absolute Error |NN(x) - f(x)|")

        # Plot 3: Loss history
        axes[2].semilogy(self.losses, 'b-', linewidth=2)
        axes[2].set_title("Training Loss (loglog scale)")
        axes[2].set_xlabel("Epoch")

        plt.tight_layout()
        plt.show()

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
    model = NeuralNetwork(hidden_layers=[32, 32, 32], activation=nn.ReLU())
    print("\nNeural Network Architecture:\n")
    print(model, "\n")

    pinn = PINN_L2_Minimizer(model, lr=5e-3)
    pinn.train(n_epochs=10000, n_collocation_points=200, verbose_freq=1000)
    pinn.plot_results()

    # Save and test loading
    pinn.save_model("models/pinn_l2_model.pth")
    pinn.load_model("models/pinn_l2_model.pth")

    # Test approximant
    NN = pinn.get_approximant()
    test_x = np.linspace(-1, 1, 5)
    
    print(f"\nApproximating the function: {inspect.getsource(f)}")
    print("Sample approximant outputs:")
    print(f"x: {test_x}")
    print(f"NN(x): {NN(test_x)}")
    print(f"f(x): {f(test_x)}")
    print(f"|f(x)-NN(x)|: {abs(NN(test_x)-f(test_x))}")

    return model, pinn

if __name__ == "__main__":
    main()