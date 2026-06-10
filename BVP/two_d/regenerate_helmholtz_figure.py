"""Regenerate the Helmholtz benchmark figure (Figure: fig:nlp-ssbroyden) from
the already-saved run, without retraining.

Everything the figure needs is on disk: ``history.npz`` holds the per-epoch
curves (J, solution L2, residual/PDE L2) and ``fields.npz`` holds the final
exact/predicted fields. So this script is pure numpy + matplotlib — no torch,
no CUDA — and runs anywhere.

The only change versus the original ``plot_results`` is the bottom-left panel,
which now shows the solution L2 error and the residual L2 error over epochs
(alongside the validation residual J) instead of the raw loss-curve bundle.

Usage:
    python regenerate_helmholtz_figure.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN_DIR = os.path.join(
    "..", "results", "helmholtz_low_ssbroyden_identity_20260608_211851"
)
ADAM_EPOCHS = 2000  # Adam -> SSBroyden handover, for the phase-boundary marker


def main() -> None:
    hist = np.load(os.path.join(RUN_DIR, "history.npz"))
    fields = np.load(os.path.join(RUN_DIR, "fields.npz"))

    x = fields["x"]
    y = fields["y"]
    XX, YY = np.meshgrid(x, y, indexing="xy")
    phi_exact = fields["phi_exact"]
    phi_pred = fields["phi_pred"]
    abs_err = fields["abs_err"]

    J_val = hist["J_val"]
    sol_l2 = hist["sol_l2"]
    pde_l2 = hist["pde_l2"]
    epochs = np.arange(1, len(sol_l2) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    im0 = axes[0, 0].contourf(XX, YY, phi_exact, levels=30, cmap="viridis")
    fig.colorbar(im0, ax=axes[0, 0])
    axes[0, 0].set_title(r"$\phi_{\mathrm{exact}}(x, y)$")

    im1 = axes[0, 1].contourf(XX, YY, phi_pred, levels=30, cmap="viridis")
    fig.colorbar(im1, ax=axes[0, 1])
    axes[0, 1].set_title(r"$\phi_{\mathrm{PINN}}(x, y)$")

    axes[1, 0].semilogy(
        np.arange(1, len(J_val) + 1), J_val,
        color="0.6", linewidth=1.0, label=r"$\mathcal{J}_{\mathrm{val}}$",
    )
    axes[1, 0].semilogy(
        epochs, sol_l2, color="C0", linewidth=1.8,
        label=r"solution $\|\phi_{\mathrm{PINN}} - \phi_{\mathrm{exact}}\|_{L^2}$",
    )
    axes[1, 0].semilogy(
        np.arange(1, len(pde_l2) + 1), pde_l2, color="C1", linewidth=1.8,
        label=r"residual $\|\Delta\phi + k^2\phi - f\|_{L^2}$",
    )
    axes[1, 0].axvline(ADAM_EPOCHS, color="k", linestyle="--", linewidth=0.8)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel(r"$L^2$ error / $\mathcal{J}$")
    axes[1, 0].set_title(r"Solution and residual $L^2$ error")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    im3 = axes[1, 1].contourf(XX, YY, abs_err, levels=30, cmap="magma")
    fig.colorbar(im3, ax=axes[1, 1])
    axes[1, 1].set_title(r"$|\phi_{\mathrm{PINN}} - \phi_{\mathrm{exact}}|$")

    for i, j in [(0, 0), (0, 1), (1, 1)]:
        axes[i, j].set_xlabel("x")
        axes[i, j].set_ylabel("y")

    plt.tight_layout()
    save_path = os.path.join(RUN_DIR, "results.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Regenerated: {save_path}")


if __name__ == "__main__":
    main()
