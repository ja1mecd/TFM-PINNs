"""Run the full 1D BVP activation study: the multi-seed sweep for every
activation, then the shared-scale heatmaps and the cross-activation summary
table.

Intended for the GPU box. Defaults reproduce the section 4.1 interpolation
protocol (20 seeds, 7x5 grid, 10000-epoch cap with early stopping) applied
to the high-frequency BVP -u''=(4 pi)^2 sin(4 pi x).

Usage
-----
    python run_activation_study_bvp.py
    python run_activation_study_bvp.py --n-seeds 20 --epochs 10000
    python run_activation_study_bvp.py --resume        # continue after a crash
"""
from __future__ import annotations

import argparse
import os

import activation_sweep_bvp as sw
import regenerate_activation_heatmaps_bvp as regen
from activation_stats_bvp import load_json, to_latex_summary

DEFAULT_SUMMARY = os.path.join("..", "..", "..", "thesis", "tables",
                               "bvp_activation_summary.tex")


def build_summary(results_dir: str, activations: list[str],
                  output_path: str) -> None:
    """Load every activation JSON and write the LaTeX summary table."""
    sweeps = []
    for act in activations:
        path = os.path.join(results_dir, f"activation_sweep_bvp_{act}.json")
        if os.path.exists(path):
            sweeps.append(load_json(path))
        else:
            print(f"[summary] {path} not found, omitting {act}")
    if not sweeps:
        print("[summary] no result JSONs; skipping table")
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(to_latex_summary(sweeps))
    print(f"[summary] wrote {output_path}")


def run(*, activations, wavenumber, layers, neurons, epochs, n_seeds,
        collocation_points, patience, min_delta, moving_avg_window,
        train_split, resample_every, pde_l2_points, linf_points, l2_points,
        residual_points, lr, lambda_bc, lambda_pde, scheduler_patience,
        scheduler_threshold, scheduler_gamma, scheduler_min_lr,
        failure_log_threshold, results_dir, output_dir, summary_path,
        seed_base=42, resume: bool = False) -> int:
    """Sweep+persist each activation, then regenerate panels and the table.

    With ``resume=True`` an activation whose final JSON already exists is
    skipped entirely, and the in-progress activation continues from its
    ``.partial.json`` checkpoint.
    """
    for act in activations:
        final_json = os.path.join(results_dir,
                                  f"activation_sweep_bvp_{act}.json")
        if resume and os.path.exists(final_json):
            print(f"[resume] {act}: final JSON already present, skipping sweep")
            continue
        args = argparse.Namespace(
            activation=act, wavenumber=wavenumber,
            layers=layers, neurons=neurons, epochs=epochs,
            collocation_points=collocation_points, patience=patience,
            min_delta=min_delta, moving_avg_window=moving_avg_window,
            train_split=train_split, resample_every=resample_every,
            pde_l2_points=pde_l2_points, linf_points=linf_points,
            l2_points=l2_points, residual_points=residual_points,
            lr=lr, lambda_bc=lambda_bc, lambda_pde=lambda_pde,
            scheduler_patience=scheduler_patience,
            scheduler_threshold=scheduler_threshold,
            scheduler_gamma=scheduler_gamma, scheduler_min_lr=scheduler_min_lr,
            n_seeds=n_seeds, seed_base=seed_base,
            failure_log_threshold=failure_log_threshold,
            results_dir=results_dir, output_dir=output_dir, resume=resume,
        )
        cells = sw.run_sweep(args)
        sw.persist(args, cells)

    regen.render_all(results_dir, output_dir, list(activations))
    build_summary(results_dir, list(activations), summary_path)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Full 1D BVP activation study.")
    p.add_argument("--activations", nargs="+",
                   default=["Tanh", "Sigmoid", "ReLU", "Softmax"])
    p.add_argument("--wavenumber", type=float, default=1.0,
                   help="Wavenumber k of -u''=(k pi)^2 sin(k pi x). Default 1 "
                        "(Adam-trainable); k=4 fails under Adam alone.")
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
    p.add_argument("--train-split", type=float, default=0.7)
    p.add_argument("--resample-every", type=int, default=500)
    p.add_argument("--pde-l2-points", type=int, default=200)
    p.add_argument("--linf-points", type=int, default=2000)
    p.add_argument("--l2-points", type=int, default=500)
    p.add_argument("--residual-points", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-bc", type=float, default=10.0)
    p.add_argument("--lambda-pde", type=float, default=1.0)
    p.add_argument("--scheduler-patience", type=int, default=200)
    p.add_argument("--scheduler-threshold", type=float, default=1e-4)
    p.add_argument("--scheduler-gamma", type=float, default=0.9)
    p.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    p.add_argument("--failure-log-threshold", type=float, default=-0.5)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default="figures")
    p.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    p.add_argument("--resume", action="store_true",
                   help="Skip activations with a final JSON and continue the "
                        "in-progress one from its .partial.json checkpoint.")
    a = p.parse_args()
    raise SystemExit(run(
        activations=a.activations, wavenumber=a.wavenumber,
        layers=a.layers, neurons=a.neurons,
        epochs=a.epochs, n_seeds=a.n_seeds,
        collocation_points=a.collocation_points, patience=a.patience,
        min_delta=a.min_delta, moving_avg_window=a.moving_avg_window,
        train_split=a.train_split, resample_every=a.resample_every,
        pde_l2_points=a.pde_l2_points, linf_points=a.linf_points,
        l2_points=a.l2_points, residual_points=a.residual_points,
        lr=a.lr, lambda_bc=a.lambda_bc, lambda_pde=a.lambda_pde,
        scheduler_patience=a.scheduler_patience,
        scheduler_threshold=a.scheduler_threshold,
        scheduler_gamma=a.scheduler_gamma, scheduler_min_lr=a.scheduler_min_lr,
        failure_log_threshold=a.failure_log_threshold,
        results_dir=a.results_dir, output_dir=a.output_dir,
        summary_path=a.summary_path, resume=a.resume,
    ))


if __name__ == "__main__":
    main()
