"""Regenerate the CFGS benchmark figure (Figure: fig:cfgs-ssbroyden) from the
already-saved run, without retraining.

The pointwise/relative-error heatmaps are recomputed deterministically from the
saved model checkpoint (exact + predicted fields), and the loss/error curves are
taken verbatim from the run's ``logs.npz``. Only the plotting code changed (the
bottom-left relative-error panel now uses a log color scale), so the rendered
figure is identical to the original run except for that panel.

Usage:
    python regenerate_cfgs_figure.py
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from pinn_ssbroyden_2d import NeuralNetwork  # noqa: E402
from pinn_ssbroyden_2d_urban import PINN_CFGS_Solver_Urban  # noqa: E402

RUN_DIR = os.path.join(
    "..", "results", "cfgs_urban_SSBroyden2_identity_20260521_185722"
)
MODEL_PATH = os.path.join("..", "models", "pinn_cfgs_urban_SSBroyden2_identity.pth")
# Match the original run (metadata.json: variant SSBroyden2, identity, lambda 0.5)
VARIANT = "SSBroyden2"
LOSS_TRANSFORM = "identity"
LOSS_LAMBDA = 0.5
GRID_N = 80  # same n as the original plot_results call


def main() -> None:
    model = NeuralNetwork(hidden_layers=(32, 32, 32), activation=nn.Tanh())
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    pinn = PINN_CFGS_Solver_Urban(
        model,
        variant=VARIANT,
        loss_transform=LOSS_TRANSFORM,
        loss_lambda=LOSS_LAMBDA,
    )

    # Inject the saved training/validation curves so the loss and L2-error
    # panels reproduce the original run exactly.
    logs = np.load(os.path.join(RUN_DIR, "logs.npz"))
    for key in (
        "obj_train", "obj_val", "J_train", "J_val",
        "pde_l2", "sol_l2", "sol_rel_l2",
    ):
        setattr(pinn, key, logs[key])

    save_path = os.path.join(RUN_DIR, "results.png")
    pinn.plot_results(n=GRID_N, save_path=save_path)
    print(f"Regenerated: {save_path}")


if __name__ == "__main__":
    main()
