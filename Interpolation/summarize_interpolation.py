"""Build the cross-activation LaTeX summary table (thesis section 4.1)
from the per-activation JSON files written by error_table_pinn.py.

Usage
-----
    python summarize_interpolation.py
    python summarize_interpolation.py --results-dir results \
        --output ../../thesis/tables/interpolation_summary.tex
"""
from __future__ import annotations

import argparse
import os

from interpolation_stats import load_json, to_latex_summary

DEFAULT_ACTIVATIONS = ["Tanh", "Sigmoid", "ReLU", "Softmax"]


def build_summary(results_dir: str, activations: list[str],
                  output_path: str) -> str:
    sweeps = []
    for act in activations:
        path = os.path.join(results_dir, f"error_table_pinn_{act}.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — skipping {act}")
            continue
        sweeps.append(load_json(path))
    if not sweeps:
        raise SystemExit("No result JSONs found; run error_table_pinn.py first.")

    tex = to_latex_summary(sweeps)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"Wrote {output_path}")
    return tex


def main() -> None:
    p = argparse.ArgumentParser(description="Build interpolation summary table.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--activations", nargs="+", default=DEFAULT_ACTIVATIONS)
    p.add_argument(
        "--output",
        default=os.path.join("..", "..", "thesis", "tables",
                             "interpolation_summary.tex"),
    )
    args = p.parse_args()
    build_summary(args.results_dir, args.activations, args.output)


if __name__ == "__main__":
    main()
