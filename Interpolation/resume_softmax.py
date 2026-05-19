"""Finish the Softmax sweep from its partial checkpoint.

Drop this file into TFM-PINNs/Interpolation/ and run:

    python resume_softmax.py

What it does:
1. Loads results/error_table_pinn_Softmax.partial.json (cells already done).
2. Runs ONLY the missing (L,W,seed) trainings to complete the 7x5x20 grid.
3. Writes the final results/error_table_pinn_Softmax.json + heatmap.
4. Rebuilds thesis/tables/interpolation_summary.tex over all 4 activations.

Standalone: uses only modules already in this directory; does not
require the --resume code added later on main.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

from interpolation_stats import CellResult, aggregate, save_json
from error_table_pinn import ACTIVATIONS, MACHINE_EPS, set_seed, plot_heatmap
from pinn_interpolant_l2 import NeuralNetwork, PINN_L2_Minimizer
import summarize_interpolation as si

ACTIVATION = "Softmax"
LAYERS = [1, 2, 3, 4, 5, 6, 7]
NEURONS = [5, 10, 20, 40, 80]
N_SEEDS = 20
SEED_BASE = 42
EPOCHS = 10000
COLLOCATION = 200
PATIENCE = 200
MIN_DELTA = 1e-7
MOVING_AVG = 20
LINF_POINTS = 2000
L2_POINTS = 200
FAIL_THRESH = -0.5
RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def main() -> None:
    partial_path = os.path.join(
        RESULTS_DIR, f"error_table_pinn_{ACTIVATION}.partial.json"
    )
    cells: list[CellResult] = []
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as fh:
            cells = [CellResult(**d) for d in json.load(fh)]
        print(f"[resume] loaded {len(cells)} cell-seeds from {partial_path}")
    else:
        print(f"[resume] no partial at {partial_path}; starting Softmax from scratch")

    done_keys = {(c.layers, c.neurons, c.seed) for c in cells}
    seeds = list(range(SEED_BASE, SEED_BASE + N_SEEDS))
    n_cells = len(LAYERS) * len(NEURONS)
    activation_factory = ACTIVATIONS[ACTIVATION]

    done = 0
    for L in LAYERS:
        for W in NEURONS:
            done += 1
            todo = [s for s in seeds if (L, W, s) not in done_keys]
            if not todo:
                print(f"[cell {done}/{n_cells}] L={L} W={W} "
                      f"— all {N_SEEDS} seeds in checkpoint, skipping")
                continue
            print(f"[cell {done}/{n_cells}] L={L} W={W} "
                  f"— running {len(todo)} missing seed(s) "
                  f"({N_SEEDS - len(todo)} skipped)")
            for seed in todo:
                t0 = time.perf_counter()
                try:
                    set_seed(seed)
                    model = NeuralNetwork(
                        hidden_layers=[W] * L,
                        activation=activation_factory(),
                    )
                    pinn = PINN_L2_Minimizer(model, lr=1e-3)
                    epochs_run = pinn.train(
                        n_epochs=EPOCHS,
                        n_collocation_points=COLLOCATION,
                        verbose_freq=max(1, EPOCHS),
                        patience=PATIENCE,
                        min_delta=MIN_DELTA,
                        moving_avg_window=MOVING_AVG,
                        l2_points=L2_POINTS,
                    )
                    dt = time.perf_counter() - t0
                    linf = pinn.compute_linf_error(n_points=LINF_POINTS)
                    l2 = pinn.compute_exact_l2_norm(n_points=L2_POINTS)
                    cells.append(CellResult(
                        layers=L, neurons=W, seed=seed,
                        linf=float(linf), l2=float(l2),
                        train_time_s=float(dt), epochs_run=int(epochs_run),
                    ))
                    print(f"  seed={seed} linf={linf:.3e} dt={dt:.1f}s")
                except Exception as exc:
                    dt = time.perf_counter() - t0
                    print(f"  [WARN] L={L} W={W} seed={seed} failed "
                          f"after {dt:.1f}s: {exc!r} — recording inf")
                    cells.append(CellResult(
                        layers=L, neurons=W, seed=seed,
                        linf=float("inf"), l2=float("inf"),
                        train_time_s=float("nan"), epochs_run=0,
                    ))
            # Checkpoint after every cell — survive another crash.
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(partial_path, "w", encoding="utf-8") as fh:
                json.dump([asdict(c) for c in cells], fh, indent=2)

    # Final aggregate + JSON + heatmap.
    sweep = aggregate(
        activation=ACTIVATION, layers=LAYERS, neurons=NEURONS, cells=cells,
        failure_log_threshold=FAIL_THRESH, machine_eps=MACHINE_EPS,
    )
    final_json = os.path.join(RESULTS_DIR, f"error_table_pinn_{ACTIVATION}.json")
    save_json(sweep, final_json)
    print(f"\nSaved final JSON: {final_json}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    heatmap_path = os.path.join(
        FIGURES_DIR, f"error_table_pinn_log_{ACTIVATION}.png"
    )
    plot_heatmap(sweep.linf_mean, LAYERS, NEURONS, ACTIVATION, heatmap_path)

    # Rebuild the cross-activation summary table.
    si.build_summary(
        results_dir=RESULTS_DIR,
        activations=["Tanh", "Sigmoid", "ReLU", "Softmax"],
        output_path=si.DEFAULT_OUTPUT,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
