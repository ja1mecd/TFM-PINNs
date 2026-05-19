"""Run the full section 4.1 interpolation study: the multi-seed
architecture sweep for every activation, then the LaTeX summary table.

Intended for the GPU box. Defaults reproduce the thesis configuration
(20 seeds, 7x5 grid, 10000-epoch cap with early stopping).

Usage
-----
    python run_interpolation_study.py
    python run_interpolation_study.py --n-seeds 20 --epochs 10000
"""
from __future__ import annotations

import argparse

import error_table_pinn as et
import summarize_interpolation as si


def run(*, activations, layers, neurons, epochs, n_seeds,
        collocation_points, patience, min_delta, moving_avg_window,
        linf_points, l2_points, failure_log_threshold,
        results_dir, output_dir, summary_path, seed_base=42) -> int:
    """Run the sweep+persist for each activation, then build the summary.

    Returns 0 on success. Per-seed training failures are absorbed inside
    ``error_table_pinn.run_sweep`` (sentinel + per-cell checkpoint); an
    activation-level error (e.g. an unknown activation key) propagates
    uncaught so the unattended run fails loudly rather than silently.
    """
    for act in activations:
        args = argparse.Namespace(
            activation=act, layers=layers, neurons=neurons, epochs=epochs,
            collocation_points=collocation_points, patience=patience,
            min_delta=min_delta, moving_avg_window=moving_avg_window,
            linf_points=linf_points, l2_points=l2_points, n_seeds=n_seeds,
            seed_base=seed_base, failure_log_threshold=failure_log_threshold,
            results_dir=results_dir, output_dir=output_dir, output=None,
        )
        cells = et.run_sweep(args)
        et.persist(args, cells)

    si.build_summary(results_dir=results_dir, activations=list(activations),
                     output_path=summary_path)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Full section 4.1 study.")
    p.add_argument("--activations", nargs="+",
                   default=["Tanh", "Sigmoid", "ReLU", "Softmax"])
    p.add_argument("--layers", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--neurons", type=int, nargs="+",
                   default=[5, 10, 20, 40, 80])
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--collocation-points", type=int, default=200)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--moving-avg-window", type=int, default=20)
    p.add_argument("--linf-points", type=int, default=2000)
    p.add_argument("--l2-points", type=int, default=200)
    p.add_argument("--failure-log-threshold", type=float, default=-0.5)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default="figures")
    p.add_argument("--summary-path", default=si.DEFAULT_OUTPUT)
    a = p.parse_args()
    raise SystemExit(run(
        activations=a.activations, layers=a.layers, neurons=a.neurons,
        epochs=a.epochs, n_seeds=a.n_seeds,
        collocation_points=a.collocation_points, patience=a.patience,
        min_delta=a.min_delta, moving_avg_window=a.moving_avg_window,
        linf_points=a.linf_points, l2_points=a.l2_points,
        failure_log_threshold=a.failure_log_threshold,
        results_dir=a.results_dir, output_dir=a.output_dir,
        summary_path=a.summary_path,
    ))


if __name__ == "__main__":
    main()
