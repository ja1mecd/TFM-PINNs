"""Re-render the BVP activation heatmaps from existing result JSONs.

For each of the three metrics (solution L-infinity, relative solution L2,
PDE residual L2) the four activation panels are rendered on a shared log10
colour scale so they can be compared directly. Use after changing the plot
style, or after a sweep finishes, to refresh

    figures/activation_sweep_bvp_<metric>_<activation>.png

without re-running the training sweep.

    python regenerate_activation_heatmaps_bvp.py
"""
from __future__ import annotations

import math
import os

from activation_stats_bvp import METRICS, load_json
from activation_sweep_bvp import heatmap_path, plot_heatmap

ACTIVATIONS = ["Tanh", "Sigmoid", "ReLU", "Softmax"]
RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def _shared_range(sweeps: dict, mean_attr: str) -> tuple[float | None, float | None]:
    """Global (vmin, vmax) over finite log10(mean) cells for one metric."""
    logs = [
        math.log10(v)
        for sw in sweeps.values()
        for row in getattr(sw, mean_attr)
        for v in row
        if v > 0 and math.isfinite(v)
    ]
    return (min(logs), max(logs)) if logs else (None, None)


def render_all(results_dir: str, figures_dir: str,
               activations: list[str]) -> None:
    """Render every activation x metric heatmap on shared per-metric scales."""
    sweeps = {}
    for act in activations:
        path = os.path.join(results_dir, f"activation_sweep_bvp_{act}.json")
        if os.path.exists(path):
            sweeps[act] = load_json(path)
        else:
            print(f"[skip] {path} not found")
    if not sweeps:
        print("No result JSONs found.")
        return

    os.makedirs(figures_dir, exist_ok=True)
    for metric_key, meta in METRICS.items():
        vmin, vmax = _shared_range(sweeps, meta["mean_attr"])
        if vmin is None:
            print(f"[{metric_key}] no finite cells, skipping")
            continue
        print(f"[{metric_key}] shared colour scale: "
              f"vmin={vmin:.2f}, vmax={vmax:.2f}")
        for act, sw in sweeps.items():
            plot_heatmap(
                getattr(sw, meta["mean_attr"]),
                list(sw.layers), list(sw.neurons), act,
                heatmap_path(figures_dir, metric_key, act),
                cbar_label=meta["label"], vmin=vmin, vmax=vmax,
            )


def main() -> None:
    render_all(RESULTS_DIR, FIGURES_DIR, ACTIVATIONS)


if __name__ == "__main__":
    main()
