import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from pinn_bvpsolver_l2 import PINN_BVP_Solver, NeuralNetwork


def evaluate_architecture(num_layers, neurons_per_layer, train_args):
    """Train a model with the given architecture and return its final error metric."""
    torch.manual_seed(train_args.seed)
    np.random.seed(train_args.seed)
    hidden_layers = [neurons_per_layer] * num_layers

    model = NeuralNetwork(hidden_layers=hidden_layers, activation=nn.Tanh())
    pinn = PINN_BVP_Solver(
        model,
        lr=train_args.lr,
        lambda_bc=train_args.lambda_bc,
        lambda_pde=train_args.lambda_pde,
    )

    pinn.train(
        n_epochs=train_args.epochs,
        n_collocation_points=train_args.collocation_points,
        verbose_freq=train_args.verbose_freq,
        patience=train_args.patience,
        min_delta=train_args.min_delta,
        moving_avg_window=train_args.moving_avg_window,
        pde_l2_points=train_args.pde_l2_points,
        train_split=train_args.train_split,
        scheduler_patience=train_args.scheduler_patience,
        scheduler_threshold=train_args.scheduler_threshold,
        scheduler_gamma=train_args.scheduler_gamma,
        scheduler_min_lr=train_args.scheduler_min_lr,
    )

    metric = np.nan
    if pinn.solution_l2_errors and np.isfinite(pinn.solution_l2_errors[-1]):
        metric = pinn.solution_l2_errors[-1]
    elif pinn.pde_l2_errors:
        metric = pinn.pde_l2_errors[-1]
    return metric


def run_grid(layers, neurons, train_args):
    results = np.full((len(layers), len(neurons)), np.nan, dtype=float)
    for i, n_layers in enumerate(layers):
        for j, n_neurons in enumerate(neurons):
            print(f"\n=== Training {n_layers} layers x {n_neurons} neurons ===")
            metric = evaluate_architecture(n_layers, n_neurons, train_args)
            results[i, j] = metric
            print(
                f"Completed {n_layers} layers, {n_neurons} neurons -> "
                f"metric: {metric:.4e}"
            )
    return results


def plot_heatmap(results, layers, neurons, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(results, cmap="viridis", origin="lower", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final L2 error (solution preferred, PDE if missing)")

    ax.set_xticks(np.arange(len(neurons)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels(neurons)
    ax.set_yticklabels(layers)
    ax.set_xlabel("Neurons per layer")
    ax.set_ylabel("Number of hidden layers")
    ax.set_title("Architecture sweep: PINN BVP solver")

    finite_vals = results[np.isfinite(results)]
    mean_val = np.mean(finite_vals) if finite_vals.size else np.nan
    for (i, j), val in np.ndenumerate(results):
        if np.isfinite(val):
            text = f"{val:.1e}"
        else:
            text = "nan"
        if np.isfinite(val) and np.isfinite(mean_val) and val > mean_val:
            text_color = "white"
        else:
            text_color = "black"
        ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"\nSaved heatmap to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PINN BVP accuracy across architectures."
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[7,6,5,4,3,2,1],
        help="List of hidden layer counts to test.",
    )
    parser.add_argument(
        "--neurons",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40, 80, 160],
        help="List of neurons-per-layer counts to test.",
    )
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
    parser.add_argument(
        "--collocation-points",
        type=int,
        default=200,
        help="Number of collocation points per epoch.",
    )
    parser.add_argument(
        "--verbose-freq",
        type=int,
        default=10_000,
        help="Print frequency (set high to keep sweep logs compact).",
    )
    parser.add_argument("--patience", type=int, default=400, help="Early stop patience.")
    parser.add_argument("--min-delta", type=float, default=1e-7, help="Early stop delta.")
    parser.add_argument(
        "--moving-avg-window",
        type=int,
        default=20,
        help="Window for validation moving average.",
    )
    parser.add_argument(
        "--pde-l2-points",
        type=int,
        default=500,
        help="Number of evaluation points for L2 metrics.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Train split for collocation points (rest used for validation).",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=200,
        help="Scheduler patience (epochs without validation improvement).",
    )
    parser.add_argument(
        "--scheduler-threshold",
        type=float,
        default=1e-4,
        help="Scheduler threshold for measuring new minima.",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.9,
        help="Scheduler decay factor.",
    )
    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for scheduler.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument(
        "--lambda-bc",
        type=float,
        default=10.0,
        help="Boundary condition loss weight.",
    )
    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=1.0,
        help="PDE loss weight.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used per model for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="architecture_heatmap_{activation}.png",
        help="Where to save the heatmap figure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    layers = args.layers
    neurons = args.neurons
    results = run_grid(layers, neurons, args)
    plot_heatmap(results, layers, neurons, args.output)


if __name__ == "__main__":
    main()
