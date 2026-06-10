"""
Regenerate the optimiser-comparison figure + summary from a finished run's
saved artefacts, applying a (tighter) success threshold — no retraining.

The original run saves per-seed `J_val`/`sol_l2` histories to
`raw_histories.npz` and per-seed final metrics to `summary_table.txt`. The
`relL2 < 1` success threshold used at run time lets a barely-escaped seed
(relL2 ~ 0.8, essentially the zero predictor) count as a "success", which
inflates the conditional mean residual of the less-reliable pipelines. This
script reconstructs the in-memory objects and re-runs `plot_comparison` /
`write_summary` with a stricter threshold so the aggregates only reflect seeds
that genuinely converged.

Usage:
    python regenerate_optim_figure.py <results_dir> [--threshold 0.1]
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

from optimiser_comparison_1d import (
    PIPELINES,
    PipelineResult,
    SeedRun,
    plot_comparison,
    write_summary,
)

# Per-seed line in summary_table.txt:
#   <pipeline>  <seed>  <final J>  <final solL2>  <final relL2>  <status>
_SEED_LINE = re.compile(
    r"^\s*(adam(?:_\w+)?)\s+(\d+)\s+"
    r"([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+(ok|FAIL)\s*$"
)


def parse_final_metrics(summary_path: str) -> dict[tuple[str, int], tuple[float, float, float]]:
    """Map (pipeline, seed) -> (final_J_val, final_sol_l2, final_sol_rel_l2)."""
    out: dict[tuple[str, int], tuple[float, float, float]] = {}
    in_per_seed = False
    with open(summary_path) as fh:
        for line in fh:
            if "Per-seed final metrics" in line:
                in_per_seed = True
                continue
            if not in_per_seed:
                continue
            m = _SEED_LINE.match(line)
            if m:
                pipe, seed = m.group(1), int(m.group(2))
                out[(pipe, seed)] = (
                    float(m.group(3)), float(m.group(4)), float(m.group(5))
                )
    return out


def reconstruct(results_dir: str) -> tuple[PipelineResult, ...]:
    npz = np.load(os.path.join(results_dir, "raw_histories.npz"), allow_pickle=True)
    finals = parse_final_metrics(os.path.join(results_dir, "summary_table.txt"))
    seeds = [int(s) for s in npz["seeds"]]
    pipelines = [str(p) for p in npz["pipelines"]] if "pipelines" in npz else list(PIPELINES)

    results: list[PipelineResult] = []
    for pipe in pipelines:
        runs: list[SeedRun] = []
        for seed in seeds:
            jkey, skey = f"J_val_{pipe}_seed{seed}", f"sol_l2_{pipe}_seed{seed}"
            if jkey not in npz:
                continue
            fJ, fS, fR = finals[(pipe, seed)]
            runs.append(SeedRun(
                seed=seed,
                J_val_history=np.asarray(npz[jkey], dtype=np.float64),
                sol_l2_history=np.asarray(npz[skey], dtype=np.float64),
                final_J_val=fJ,
                final_pde_l2=float("nan"),  # unused by plot/threshold
                final_sol_l2=fS,
                final_sol_rel_l2=fR,
            ))
        results.append(PipelineResult(pipeline=pipe, seeds=tuple(runs)))
    return tuple(results)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir", type=str)
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Success cutoff on final relative L^2 error (default 0.1).")
    p.add_argument("--adam-warmup", type=int, default=2000)
    p.add_argument("--k", type=float, default=4.0)
    p.add_argument("--portrait", action="store_true",
                   help="Render the 2x2 grid in a near-square geometry "
                        "(11x10in) so the subplots keep natural proportions "
                        "when the figure is placed upright on a portrait page; "
                        "appends '_portrait' to the figure filename.")
    args = p.parse_args()

    results = reconstruct(args.results_dir)
    tag = f"thr{args.threshold:g}".replace(".", "p")
    if args.portrait:
        tag += "_portrait"
    fig_path = os.path.join(args.results_dir, f"optimiser_comparison_{tag}.png")
    sum_path = os.path.join(args.results_dir, f"summary_table_{tag}.txt")

    figsize = (11.0, 10.0) if args.portrait else (14.0, 10.0)
    plot_comparison(results, fig_path, k=args.k, adam_warmup=args.adam_warmup,
                    rel_l2_threshold=args.threshold, figsize=figsize)
    write_summary(results, sum_path, k=args.k, total_epochs=-1,
                  adam_warmup=args.adam_warmup, seeds=tuple(
                      int(s) for s in (results[0].seeds and
                                       [r.seed for r in results[0].seeds]) or []),
                  rel_l2_threshold=args.threshold)
    print(f"Regenerated with threshold {args.threshold:g}:\n  {fig_path}\n  {sum_path}")


if __name__ == "__main__":
    main()
