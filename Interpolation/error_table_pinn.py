"""Architecture sweep for the 1D interpolation benchmark (thesis section 4.1).

For a fixed activation, trains an L x W grid of fully connected networks
with empirical squared-error loss, repeated over a seed ensemble, and
records the L-infinity and L2 errors and training time per run. Mean
log10 L-infinity is shown as a heatmap; raw per-seed records are written
as JSON for `summarize_interpolation.py` and for the section 4.1 prose.

Usage
-----
    python error_table_pinn.py                        # Tanh, 20 seeds
    python error_table_pinn.py --activation ReLU --n-seeds 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from interpolation_stats import CellResult, aggregate, save_json
from pinn_interpolant_l2 import NeuralNetwork, PINN_L2_Minimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# float32 training tensors -> single-precision machine epsilon.
MACHINE_EPS = float(np.finfo(np.float32).eps)  # ~1.1920929e-07

ACTIVATIONS = {
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "ReLU": nn.ReLU,
    "Softmax": lambda: nn.Softmax(dim=-1),
}


def set_seed(seed: int) -> None:
    """Seed every RNG that affects a training run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed architecture sweep for 1D interpolation."
    )
    p.add_argument("--activation", choices=list(ACTIVATIONS), default="Tanh")
    p.add_argument("--layers", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--neurons", type=int, nargs="+",
                   default=[5, 10, 20, 40, 80])
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--collocation-points", type=int, default=200)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--moving-avg-window", type=int, default=20)
    p.add_argument("--linf-points", type=int, default=2000)
    p.add_argument("--l2-points", type=int, default=200)
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--failure-log-threshold", type=float, default=-0.5)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--output-dir", type=str, default="figures")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def _write_partial(results_dir: str, activation: str,
                    cells: list[CellResult]) -> None:
    """Checkpoint raw cells so an unattended crash loses at most one cell."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"error_table_pinn_{activation}.partial.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(c) for c in cells], fh, indent=2)


def run_sweep(args: argparse.Namespace) -> list[CellResult]:
    activation_factory = ACTIVATIONS[args.activation]
    seeds = list(range(args.seed_base, args.seed_base + args.n_seeds))
    n_cells = len(args.layers) * len(args.neurons)
    cells: list[CellResult] = []

    print(f"\nArchitecture sweep — activation: {args.activation} — "
          f"{n_cells} cells x {len(seeds)} seeds\n")

    done = 0
    for n_layers in args.layers:
        for n_neurons in args.neurons:
            done += 1
            print(f"[cell {done}/{n_cells}] L={n_layers} W={n_neurons} "
                  f"— starting {len(seeds)} seeds")
            for seed in seeds:
                t0 = time.perf_counter()
                try:
                    set_seed(seed)
                    hidden = [n_neurons] * n_layers
                    model = NeuralNetwork(
                        hidden_layers=hidden,
                        activation=activation_factory(),
                    )
                    pinn = PINN_L2_Minimizer(model, lr=1e-3)
                    epochs_run = pinn.train(
                        n_epochs=args.epochs,
                        n_collocation_points=args.collocation_points,
                        verbose_freq=max(1, args.epochs),
                        patience=args.patience,
                        min_delta=args.min_delta,
                        moving_avg_window=args.moving_avg_window,
                        l2_points=args.l2_points,
                    )
                    dt = time.perf_counter() - t0
                    linf = pinn.compute_linf_error(n_points=args.linf_points)
                    l2 = pinn.compute_exact_l2_norm(n_points=args.l2_points)
                    cells.append(CellResult(
                        layers=n_layers, neurons=n_neurons, seed=seed,
                        linf=float(linf), l2=float(l2),
                        train_time_s=float(dt), epochs_run=int(epochs_run),
                    ))
                    print(f"  seed={seed} linf={linf:.3e} dt={dt:.1f}s")
                except Exception as exc:  # unattended run: never lose the sweep
                    dt = time.perf_counter() - t0
                    print(f"  [WARN] L={n_layers} W={n_neurons} seed={seed} "
                          f"failed after {dt:.1f}s: {exc!r} — recording inf")
                    cells.append(CellResult(
                        layers=n_layers, neurons=n_neurons, seed=seed,
                        linf=float("inf"), l2=float("inf"),
                        train_time_s=float(dt), epochs_run=0,
                    ))
            _write_partial(args.results_dir, args.activation, cells)

    return cells


def plot_heatmap(linf_mean: Sequence[Sequence[float]], layers: list[int],
                 neurons: list[int], activation: str,
                 output_path: str) -> None:
    arr = np.array(linf_mean, dtype=float)
    with np.errstate(divide="ignore"):
        log_errors = np.log10(arr)
    masked = np.ma.masked_invalid(log_errors)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="lower")
    mean_finite = masked.mean() if masked.count() else 0.0
    for i in range(len(layers)):
        for j in range(len(neurons)):
            val = log_errors[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "fail", ha="center", va="center",
                        color="black")
                continue
            color = "white" if val < mean_finite else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10} \varepsilon_\infty$ (mean over seeds)")
    ax.set_xticks(range(len(neurons)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(neurons)
    ax.set_yticklabels(layers)
    ax.set_xlabel("Neurons per layer (W)")
    ax.set_ylabel("Hidden layers (L)")
    ax.set_title(
        rf"$\log_{{10}} \varepsilon_\infty$ on the depth/width grid — "
        rf"{activation} (mean over seeds)"
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved heatmap to {output_path}")


def persist(args: argparse.Namespace,
            cells: list[CellResult]) -> tuple[str, str]:
    """Aggregate, write JSON + heatmap. Returns (json_path, heatmap_path)."""
    sweep = aggregate(
        activation=args.activation,
        layers=args.layers,
        neurons=args.neurons,
        cells=cells,
        failure_log_threshold=args.failure_log_threshold,
        machine_eps=MACHINE_EPS,
    )
    os.makedirs(args.results_dir, exist_ok=True)
    json_path = os.path.join(
        args.results_dir, f"error_table_pinn_{args.activation}.json"
    )
    save_json(sweep, json_path)
    print(f"Saved raw results to {json_path}")

    if args.output is None:
        os.makedirs(args.output_dir, exist_ok=True)
        heatmap_path = os.path.join(
            args.output_dir, f"error_table_pinn_log_{args.activation}.png"
        )
    else:
        heatmap_path = args.output
    plot_heatmap(sweep.linf_mean, args.layers, args.neurons,
                 args.activation, heatmap_path)
    return json_path, heatmap_path


def main() -> None:
    args = parse_args()
    cells = run_sweep(args)
    persist(args, cells)


if __name__ == "__main__":
    main()
