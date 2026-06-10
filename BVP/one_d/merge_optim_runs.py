"""
Merge two (or more) finished optimiser-comparison runs into one combined
ensemble and regenerate the figure + summary table — no retraining.

The Table 4.4 / Fig 4.3 protocol fixes everything except the seed, so runs
launched at different times (e.g. the original seeds 42-46 and the extension
seeds 47-61) are statistically homogeneous and can be pooled. Each input dir
must contain the `raw_histories.npz` + `summary_table.txt` artefacts written
by `optimiser_comparison_1d.py`; reconstruction reuses
`regenerate_optim_figure.reconstruct`.

Usage:
    python merge_optim_runs.py <results_dir_a> <results_dir_b> [...] \
        [--threshold 0.01] [--out-dir DIR] [--portrait]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from optimiser_comparison_1d import (
    PipelineResult,
    plot_comparison,
    write_summary,
)
from regenerate_optim_figure import reconstruct


def merge_results(
    *result_sets: tuple[PipelineResult, ...],
) -> tuple[PipelineResult, ...]:
    """Pool the seed ensembles of several runs, pipeline by pipeline.

    Every run must contain the same pipelines in the same order (the runs
    are only poolable if they executed the identical protocol), and no seed
    may appear twice within a pipeline.
    """
    if not result_sets:
        raise ValueError("need at least one result set to merge")

    reference = [r.pipeline for r in result_sets[0]]
    for i, rs in enumerate(result_sets[1:], start=1):
        names = [r.pipeline for r in rs]
        if names != reference:
            raise ValueError(
                f"pipeline mismatch between run 0 {reference} and "
                f"run {i} {names}; runs are not poolable"
            )

    merged: list[PipelineResult] = []
    for idx, pipeline in enumerate(reference):
        all_runs = tuple(
            run for rs in result_sets for run in rs[idx].seeds
        )
        seen: set[int] = set()
        for run in all_runs:
            if run.seed in seen:
                raise ValueError(
                    f"duplicate seed {run.seed} in pipeline {pipeline!r}; "
                    "the same seed cannot be pooled twice"
                )
            seen.add(run.seed)
        merged.append(PipelineResult(pipeline=pipeline, seeds=all_runs))
    return tuple(merged)


def save_merged_npz(
    results: tuple[PipelineResult, ...], out_path: str
) -> None:
    """Re-emit the merged ensemble in the raw_histories.npz layout so the
    combined dir is a drop-in input for regenerate_optim_figure.py."""
    seeds = [s.seed for s in results[0].seeds]
    np.savez(
        out_path,
        pipelines=np.asarray([r.pipeline for r in results]),
        seeds=np.asarray(seeds, dtype=np.int64),
        **{
            f"J_val_{r.pipeline}_seed{s.seed}": s.J_val_history
            for r in results for s in r.seeds
        },
        **{
            f"sol_l2_{r.pipeline}_seed{s.seed}": s.sol_l2_history
            for r in results for s in r.seeds
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dirs", type=str, nargs="+",
                   help="Two or more finished run directories to pool.")
    p.add_argument("--threshold", type=float, default=0.01,
                   help="Success cutoff on final relative L^2 (default 0.01, "
                        "the thesis Table 4.4 criterion).")
    p.add_argument("--adam-warmup", type=int, default=2000)
    p.add_argument("--k", type=float, default=4.0)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory; default <first_dir>_merged.")
    p.add_argument("--portrait", action="store_true",
                   help="Also render the portrait (9.5x13in) thesis figure.")
    args = p.parse_args()

    if len(args.results_dirs) < 2:
        raise SystemExit("need at least two run directories to merge")

    merged = merge_results(*(reconstruct(d) for d in args.results_dirs))
    seeds = tuple(s.seed for s in merged[0].seeds)

    out_dir = args.out_dir or args.results_dirs[0].rstrip("/") + "_merged"
    os.makedirs(out_dir, exist_ok=True)

    save_merged_npz(merged, os.path.join(out_dir, "raw_histories.npz"))

    tag = f"thr{args.threshold:g}".replace(".", "p")
    write_summary(
        merged, os.path.join(out_dir, f"summary_table_{tag}.txt"),
        k=args.k, total_epochs=-1, adam_warmup=args.adam_warmup,
        seeds=seeds, rel_l2_threshold=args.threshold,
    )
    # write_summary with the run-time threshold-1.0 convention, so the
    # merged dir carries the same baseline artefact as a fresh run.
    write_summary(
        merged, os.path.join(out_dir, "summary_table.txt"),
        k=args.k, total_epochs=-1, adam_warmup=args.adam_warmup,
        seeds=seeds, rel_l2_threshold=1.0,
    )
    plot_comparison(
        merged, os.path.join(out_dir, f"optimiser_comparison_{tag}.png"),
        k=args.k, adam_warmup=args.adam_warmup,
        rel_l2_threshold=args.threshold,
    )
    if args.portrait:
        plot_comparison(
            merged,
            os.path.join(out_dir, f"optimiser_comparison_{tag}_portrait.png"),
            k=args.k, adam_warmup=args.adam_warmup,
            rel_l2_threshold=args.threshold, figsize=(9.5, 13.0),
        )
    print(f"Merged {len(args.results_dirs)} runs "
          f"({len(seeds)} seeds) into: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
