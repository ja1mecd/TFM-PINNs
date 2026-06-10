"""Re-render the four activation heatmaps from existing result JSONs.

Use after changing the plot style in error_table_pinn.plot_heatmap (e.g.
larger fonts) to refresh figures/error_table_pinn_log_<activation>.png
without re-running the 2800-training sweep. All four panels are rendered
on a shared colour scale so they can be compared directly.

    python regenerate_heatmaps.py
"""
from __future__ import annotations

import math
import os

from error_table_pinn import plot_heatmap
from interpolation_stats import load_json, with_rell2_failures

ACTIVATIONS = ["Tanh", "Sigmoid", "ReLU", "Softmax"]
RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def render_all(results_dir: str, figures_dir: str,
               activations: list[str]) -> None:
    """Render every activation heatmap on a shared log10 colour scale."""
    sweeps = {}
    for act in activations:
        path = os.path.join(results_dir, f"error_table_pinn_{act}.json")
        if os.path.exists(path):
            sweeps[act] = with_rell2_failures(load_json(path))
        else:
            print(f"[skip] {path} not found")
    if not sweeps:
        print("No result JSONs found.")
        return

    # Global colour range over all finite log10(mean linf) cells.
    logs = [
        math.log10(v)
        for sw in sweeps.values()
        for row in sw.linf_mean
        for v in row
        if v > 0 and math.isfinite(v)
    ]
    vmin, vmax = (min(logs), max(logs)) if logs else (None, None)
    print(f"shared colour scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

    os.makedirs(figures_dir, exist_ok=True)
    for act, sw in sweeps.items():
        out = os.path.join(figures_dir, f"error_table_pinn_log_{act}.png")
        plot_heatmap(sw.linf_mean, list(sw.layers), list(sw.neurons),
                     act, out, vmin=vmin, vmax=vmax,
                     fail_mask=sw.n_failed)


def main() -> None:
    render_all(RESULTS_DIR, FIGURES_DIR, ACTIVATIONS)


if __name__ == "__main__":
    main()
