"""Activation sweep for the 1D BVP benchmark (companion of section 4.1).

For a fixed activation, trains an L x W grid of fully connected networks on
the high-frequency Dirichlet problem

    -u''(x) = (k pi)^2 sin(k pi x),   x in (0, 1),   u(0)=u(1)=0,   k=4,

with the hard ansatz u_hat(x) = x(1-x) N_theta(x), repeated over a seed
ensemble. After training, each run is scored on a dense grid by three
metrics -- solution L-infinity error, relative solution L2 error and the
strong-form PDE residual L2 norm -- and the mean log10 of each is rendered
as a heatmap. Raw per-seed records are written as JSON for the summary
table and the BVP activation-comparison prose.

This mirrors ``Interpolation/error_table_pinn.py`` so the BVP heatmaps can
be read directly against the interpolation ones; the only differences are
the PDE solver in place of the L2 interpolant and the three reported
metrics in place of one.

Usage
-----
    python activation_sweep_bvp.py                       # Tanh, 20 seeds
    python activation_sweep_bvp.py --activation ReLU --n-seeds 20
    python activation_sweep_bvp.py --activation Sigmoid --resume
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless backend (SSH / no display)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from activation_stats_bvp import (
    METRICS,
    BVPCellResult,
    aggregate,
    save_json,
)
from pinn_bvpsolver_l2 import (
    NeuralNetwork,
    PINN_BVP_Solver,
    a,
    b,
    u_exact,
)

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


@contextmanager
def _suppress_stdout():
    """Silence the solver's per-epoch / scheduler prints during a sweep.

    The BVP solver and its ReduceLROnPlateau scheduler print verbosely on
    every run; over a 2800-training sweep that buries the per-seed summary
    lines we actually want. stderr is left untouched so warnings and
    tracebacks still surface.
    """
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        import sys
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def compute_metrics(
    pinn: PINN_BVP_Solver,
    *,
    linf_points: int,
    l2_points: int,
    residual_points: int,
) -> tuple[float, float, float]:
    """Score a trained model on a dense grid.

    Returns ``(sol_linf, sol_rel_l2, residual_l2)``:
      * ``sol_linf``    = max_x |u_hat(x) - u_exact(x)|            on linf_points
      * ``sol_rel_l2``  = ||u_hat - u_exact||_L2 / ||u_exact||_L2  on l2_points
      * ``residual_l2`` = ||u_hat'' - f||_L2                       on residual_points
    """
    nn_fn = pinn.get_approximant()

    x_dense = np.linspace(a, b, linf_points)
    u_pred = np.asarray(nn_fn(x_dense), dtype=float)
    u_true = np.asarray(u_exact(x_dense), dtype=float)
    sol_linf = float(np.max(np.abs(u_pred - u_true)))

    x_l2 = np.linspace(a, b, l2_points)
    u_pred_l2 = np.asarray(nn_fn(x_l2), dtype=float)
    u_true_l2 = np.asarray(u_exact(x_l2), dtype=float)
    num = float(np.sqrt(np.trapz((u_pred_l2 - u_true_l2) ** 2, x_l2)))
    den = float(np.sqrt(np.trapz(u_true_l2 ** 2, x_l2)))
    sol_rel_l2 = num / den if den > 0 else float("inf")

    residual_l2 = float(pinn.compute_pde_l2_norm(n_points=residual_points))
    return sol_linf, sol_rel_l2, residual_l2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed activation sweep for the 1D BVP."
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
    p.add_argument("--train-split", type=float, default=0.7)
    p.add_argument("--resample-every", type=int, default=500)
    # in-training residual monitor (kept coarse to save autograd time);
    # the final reported metrics use the dense grids below.
    p.add_argument("--pde-l2-points", type=int, default=200)
    # dense post-training evaluation grids
    p.add_argument("--linf-points", type=int, default=2000)
    p.add_argument("--l2-points", type=int, default=500)
    p.add_argument("--residual-points", type=int, default=500)
    # optimiser / loss
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-bc", type=float, default=10.0)
    p.add_argument("--lambda-pde", type=float, default=1.0)
    p.add_argument("--scheduler-patience", type=int, default=200)
    p.add_argument("--scheduler-threshold", type=float, default=1e-4)
    p.add_argument("--scheduler-gamma", type=float, default=0.9)
    p.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    # ensemble + bookkeeping
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--failure-log-threshold", type=float, default=-0.5)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--output-dir", type=str, default="figures")
    p.add_argument("--resume", action="store_true",
                   help="Skip (L,W,seed) cells already in the activation's "
                        ".partial.json so an interrupted sweep can continue.")
    return p.parse_args()


def _partial_path(results_dir: str, activation: str) -> str:
    return os.path.join(
        results_dir, f"activation_sweep_bvp_{activation}.partial.json"
    )


def _write_partial(results_dir: str, activation: str,
                   cells: list[BVPCellResult]) -> None:
    """Checkpoint raw cells so an unattended crash loses at most one cell."""
    os.makedirs(results_dir, exist_ok=True)
    path = _partial_path(results_dir, activation)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(c) for c in cells], fh, indent=2)


def _load_partial(results_dir: str, activation: str) -> list[BVPCellResult]:
    """Load checkpoint cells written by `_write_partial`; [] if missing/corrupt."""
    path = _partial_path(results_dir, activation)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return [BVPCellResult(**d) for d in json.load(fh)]
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        print(f"[WARN] resume: ignoring corrupt {path}: {exc!r}")
        return []


def _train_one(args: argparse.Namespace, activation_factory,
               n_layers: int, n_neurons: int) -> tuple[float, float, float, int]:
    """Train a single network and return its three metrics + epochs run."""
    hidden = [n_neurons] * n_layers
    model = NeuralNetwork(hidden_layers=hidden, activation=activation_factory())
    pinn = PINN_BVP_Solver(
        model, lr=args.lr, lambda_bc=args.lambda_bc, lambda_pde=args.lambda_pde
    )
    with _suppress_stdout():
        pinn.train(
            n_epochs=args.epochs,
            n_collocation_points=args.collocation_points,
            verbose_freq=max(1, args.epochs),
            patience=args.patience,
            min_delta=args.min_delta,
            moving_avg_window=args.moving_avg_window,
            pde_l2_points=args.pde_l2_points,
            train_split=args.train_split,
            scheduler_patience=args.scheduler_patience,
            scheduler_threshold=args.scheduler_threshold,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_min_lr=args.scheduler_min_lr,
            resample_every=args.resample_every,
        )
    epochs_run = len(pinn.losses)  # solver does not return this
    sol_linf, sol_rel_l2, residual_l2 = compute_metrics(
        pinn,
        linf_points=args.linf_points,
        l2_points=args.l2_points,
        residual_points=args.residual_points,
    )
    return sol_linf, sol_rel_l2, residual_l2, epochs_run


def run_sweep(args: argparse.Namespace) -> list[BVPCellResult]:
    activation_factory = ACTIVATIONS[args.activation]
    seeds = list(range(args.seed_base, args.seed_base + args.n_seeds))
    n_cells = len(args.layers) * len(args.neurons)

    if getattr(args, "resume", False):
        cells: list[BVPCellResult] = list(
            _load_partial(args.results_dir, args.activation)
        )
        done_keys = {(c.layers, c.neurons, c.seed) for c in cells}
        if cells:
            print(f"[resume] {args.activation}: loaded {len(cells)} cell-seeds "
                  f"from partial checkpoint; remaining will be run.")
    else:
        cells = []
        done_keys = set()

    print(f"\nActivation sweep (1D BVP) — activation: {args.activation} — "
          f"{n_cells} cells x {len(seeds)} seeds\n")

    done = 0
    for n_layers in args.layers:
        for n_neurons in args.neurons:
            done += 1
            todo_seeds = [s for s in seeds
                          if (n_layers, n_neurons, s) not in done_keys]
            if not todo_seeds:
                print(f"[cell {done}/{n_cells}] L={n_layers} W={n_neurons} "
                      f"— all {len(seeds)} seeds already in checkpoint, skipping")
                continue
            print(f"[cell {done}/{n_cells}] L={n_layers} W={n_neurons} "
                  f"— starting {len(todo_seeds)} seeds "
                  f"({len(seeds) - len(todo_seeds)} skipped from checkpoint)")
            for seed in todo_seeds:
                t0 = time.perf_counter()
                try:
                    set_seed(seed)
                    sol_linf, sol_rel_l2, residual_l2, epochs_run = _train_one(
                        args, activation_factory, n_layers, n_neurons
                    )
                    dt = time.perf_counter() - t0
                    cells.append(BVPCellResult(
                        layers=n_layers, neurons=n_neurons, seed=seed,
                        sol_linf=sol_linf, sol_rel_l2=sol_rel_l2,
                        residual_l2=residual_l2,
                        train_time_s=float(dt), epochs_run=int(epochs_run),
                    ))
                    print(f"  seed={seed} linf={sol_linf:.3e} "
                          f"relL2={sol_rel_l2:.3e} res={residual_l2:.3e} "
                          f"dt={dt:.1f}s")
                except Exception as exc:  # unattended run: never lose the sweep
                    dt = time.perf_counter() - t0
                    print(f"  [WARN] L={n_layers} W={n_neurons} seed={seed} "
                          f"failed after {dt:.1f}s: {exc!r} — recording inf")
                    cells.append(BVPCellResult(
                        layers=n_layers, neurons=n_neurons, seed=seed,
                        sol_linf=float("inf"), sol_rel_l2=float("inf"),
                        residual_l2=float("inf"),
                        train_time_s=float("nan"), epochs_run=0,
                    ))
            _write_partial(args.results_dir, args.activation, cells)

    return cells


def plot_heatmap(grid_mean: Sequence[Sequence[float]], layers: list[int],
                 neurons: list[int], activation: str, output_path: str,
                 cbar_label: str, vmin: float | None = None,
                 vmax: float | None = None) -> None:
    """Render one log10-error heatmap.

    Style is identical to ``Interpolation/error_table_pinn.plot_heatmap`` so
    the BVP and interpolation panels are visually interchangeable; only the
    colour-bar label changes with the metric. Pass a shared ``vmin``/``vmax``
    (log10 units) to put several panels on a common colour scale.
    """
    arr = np.array(grid_mean, dtype=float)
    with np.errstate(divide="ignore"):
        log_errors = np.log10(arr)
    masked = np.ma.masked_invalid(log_errors)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax)
    if vmin is not None and vmax is not None:
        thresh = 0.5 * (vmin + vmax)
    else:
        thresh = masked.mean() if masked.count() else 0.0
    for i in range(len(layers)):
        for j in range(len(neurons)):
            val = log_errors[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "fail", ha="center", va="center",
                        color="black", fontsize=16)
                continue
            color = "white" if val < thresh else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=18)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=15)
    cbar.ax.tick_params(labelsize=13)
    ax.set_xticks(range(len(neurons)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(neurons, fontsize=15)
    ax.set_yticklabels(layers, fontsize=15)
    ax.set_xlabel("Neurons per layer (W)", fontsize=16)
    ax.set_ylabel("Hidden layers (L)", fontsize=16)
    ax.set_title(activation, fontsize=17)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def heatmap_path(output_dir: str, metric_key: str, activation: str) -> str:
    token = METRICS[metric_key]["token"]
    return os.path.join(
        output_dir, f"activation_sweep_bvp_{token}_{activation}.png"
    )


def persist(args: argparse.Namespace, cells: list[BVPCellResult]) -> str:
    """Aggregate, write JSON, and render one auto-scaled heatmap per metric.

    Returns the JSON path. Shared-colour-scale panels for the thesis figure
    are produced afterwards by ``regenerate_activation_heatmaps_bvp.py``.
    """
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
        args.results_dir, f"activation_sweep_bvp_{args.activation}.json"
    )
    save_json(sweep, json_path)
    print(f"Saved raw results to {json_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    for metric_key, meta in METRICS.items():
        grid_mean = getattr(sweep, meta["mean_attr"])
        plot_heatmap(
            grid_mean, args.layers, args.neurons, args.activation,
            heatmap_path(args.output_dir, metric_key, args.activation),
            cbar_label=meta["label"],
        )
    return json_path


def main() -> None:
    args = parse_args()
    cells = run_sweep(args)
    persist(args, cells)


if __name__ == "__main__":
    main()
