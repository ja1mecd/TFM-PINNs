"""Architecture sweep for the 1D interpolation benchmark.

Reproduces Figure 4.1 of the thesis (section 4.1, "Approximation of
functions in one dimension"). For a fixed activation, trains an
L x W grid of fully connected networks with empirical squared error
loss and reports the L-infinity error on a dense validation grid.

The target function and interval are imported from
`pinn_interpolant_l2` to avoid duplicate sources of truth — that
module is what defines the problem actually being solved during
training.

Usage
-----
    python error_table_pinn.py                       # Tanh, default grid
    python error_table_pinn.py --activation Sigmoid
    python error_table_pinn.py --activation ReLU --output custom.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pinn_interpolant_l2 import NeuralNetwork, PINN_L2_Minimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIVATIONS = {
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "ReLU": nn.ReLU,
    "Softmax": lambda: nn.Softmax(dim=-1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Architecture sweep over (depth, width) for 1D interpolation."
    )
    parser.add_argument(
        "--activation",
        choices=list(ACTIVATIONS),
        default="Tanh",
        help="Activation function used in every layer (default: Tanh).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7],
        help="Hidden-layer counts to sweep (default: 1..7).",
    )
    parser.add_argument(
        "--neurons",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40, 80],
        help="Neurons per layer to sweep (default: 5,10,20,40,80).",
    )
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--collocation-points", type=int, default=200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--moving-avg-window", type=int, default=20)
    parser.add_argument("--linf-points", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output PNG path. Defaults to "
            "<output-dir>/error_table_pinn_log_<activation>.png."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_sweep(args: argparse.Namespace) -> np.ndarray:
    activation_factory = ACTIVATIONS[args.activation]
    layers, neurons = args.layers, args.neurons
    error_matrix = np.zeros((len(layers), len(neurons)))

    print(f"\nArchitecture sweep — activation: {args.activation}\n")

    for i, n_layers in enumerate(layers):
        for j, n_neurons in enumerate(neurons):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            print(
                f"\n[{i + 1}/{len(layers)},{j + 1}/{len(neurons)}] "
                f"Training {n_layers} layers x {n_neurons} neurons"
            )

            hidden = [n_neurons] * n_layers
            model = NeuralNetwork(hidden_layers=hidden, activation=activation_factory())
            pinn = PINN_L2_Minimizer(model, lr=1e-3)

            pinn.train(
                n_epochs=args.epochs,
                n_collocation_points=args.collocation_points,
                verbose_freq=max(1, args.epochs),  # silence per-epoch logs
                patience=args.patience,
                min_delta=args.min_delta,
                moving_avg_window=args.moving_avg_window,
            )

            linf_err = pinn.compute_linf_error(n_points=args.linf_points)
            error_matrix[i, j] = linf_err
            print(
                f"L-inf error ({n_layers} layers, {n_neurons} neurons) = "
                f"{linf_err:.6e}"
            )

    return error_matrix


def plot_heatmap(
    error_matrix: np.ndarray,
    layers: list[int],
    neurons: list[int],
    activation: str,
    output_path: str,
) -> None:
    log_errors = np.log10(error_matrix)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(log_errors, cmap="viridis", aspect="auto", origin="lower")

    for i in range(len(layers)):
        for j in range(len(neurons)):
            val = log_errors[i, j]
            color = "white" if val < log_errors.mean() else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10} \varepsilon_\infty$")

    ax.set_xticks(range(len(neurons)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(neurons)
    ax.set_yticklabels(layers)
    ax.set_xlabel("Neurons per layer (W)")
    ax.set_ylabel("Hidden layers (L)")
    ax.set_title(
        rf"$\log_{{10}} \varepsilon_\infty$ on the depth/width grid — {activation}"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"\nSaved heatmap to {output_path}")


def main() -> None:
    args = parse_args()
    if args.output is None:
        os.makedirs(args.output_dir, exist_ok=True)
        args.output = os.path.join(
            args.output_dir, f"error_table_pinn_log_{args.activation}.png"
        )

    error_matrix = run_sweep(args)

    print("\nL-inf error matrix:\n")
    print(error_matrix)

    plot_heatmap(error_matrix, args.layers, args.neurons, args.activation, args.output)


if __name__ == "__main__":
    main()
