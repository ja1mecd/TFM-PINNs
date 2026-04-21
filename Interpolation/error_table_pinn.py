import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pinn_interpolant_l2 import NeuralNetwork, PINN_L2_Minimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target function
f = lambda x: (x - 0.5)**2
a, b = -1, 1


# -----------------------------------------------------------
# Choose architecture grid
# -----------------------------------------------------------

layer_list   = [1, 2, 3, 4, 5, 6, 7]         # number of hidden layers (rows)
neuron_list  = [5, 10, 20, 40, 80]     # neurons per layer (columns)

L = len(layer_list)
M = len(neuron_list)

error_matrix = np.zeros((L, M))


# -----------------------------------------------------------
# Main training loop
# -----------------------------------------------------------
print("\nTRAINING ARCHITECTURE GRID...\n")

for i, n_layers in enumerate(layer_list):
    for j, n_neurons in enumerate(neuron_list):

        print(f"\nTraining model {i+1},{j+1}: {n_layers} layers × {n_neurons} neurons")

        hidden = [n_neurons] * n_layers
        model = NeuralNetwork(hidden_layers=hidden, activation=nn.Softmax())

        pinn = PINN_L2_Minimizer(model, lr=1e-3)

        # --- TRAIN THE MODEL ---
        pinn.train(
            n_epochs=10000,
            n_collocation_points=200,
            verbose_freq=1000,
            patience=200,
            min_delta=1e-7,
            moving_avg_window=20
        )

        # --- COMPUTE L2 ERROR ---
        l2_err = pinn.compute_exact_l2_norm()
        error_matrix[i, j] = l2_err

        print(f"L2 Error for ({n_layers} layers, {n_neurons} neurons) = {l2_err:.6e}")


# -----------------------------------------------------------
# Heatmap plot
# -----------------------------------------------------------

plt.figure(figsize=(12, 6))
im = plt.imshow(error_matrix, cmap="viridis", aspect="auto")

for i in range(L):
    for j in range(M):
        val = error_matrix[i, j]
        plt.text(j, i, f"{val:.3e}",
                 ha="center", va="center",
                 color="white" if val > 0.1 else "black")

plt.colorbar(im, label="L2 Error")
plt.xticks(range(M), neuron_list)
plt.yticks(range(L), layer_list)
plt.xlabel("Neurons per Layer")
plt.ylabel("Number of Hidden Layers")
plt.title("L2 Error Across PINN Architectures")
plt.tight_layout()
plt.show()

# print matrix in terminal
print("\nFINAL L2 ERROR MATRIX:\n")
print(error_matrix)


plt.figure(figsize=(12, 6))
im = plt.imshow(np.log(error_matrix), cmap="viridis", aspect="auto")

for i in range(L):
    for j in range(M):
        val = np.log(error_matrix[i, j])
        plt.text(j, i, f"{val:.3e}",
                 ha="center", va="center",
                 color="white" if val > 0.1 else "black")

plt.colorbar(im, label="L2 Error")
plt.xticks(range(M), neuron_list)
plt.yticks(range(L), layer_list)
plt.xlabel("Neurons per Layer")
plt.ylabel("Number of Hidden Layers")
plt.title("L2 Error Across PINN Architectures")
plt.tight_layout()
plt.show()

# print matrix in terminal
print("\nFINAL L2 ERROR MATRIX:\n")
print(error_matrix)