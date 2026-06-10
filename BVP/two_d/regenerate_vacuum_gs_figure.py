"""Regenerate the vacuum Grad-Shafranov figure (Figure: fig:vacuum-gs-results)
from the already-saved run, without retraining.

Everything needed is in ``raw_histories.npz``: the per-seed predicted fields
and the per-epoch J / solution-L2 histories. So this is pure numpy + matplotlib.

Two rendering changes versus the original ``plot_results``, both aimed at the
pointwise-error panel, whose content is the converged correction
psi_hat - psi_exact = bubble * N at the ~1e-7 noise floor (near-Nyquist in R):

  * interpolation='nearest' — faithful per-pixel rendering, so the antialiased
    resampling no longer adds a moire texture on top of the real grid-scale
    striping;
  * the log color range is clipped to the top two decades (vmin = vmax/100)
    instead of bottoming out at ~1e-14 at every zero crossing, so the map reads
    as amplitude rather than as a forest of zero-crossing spikes.

Usage:
    python regenerate_vacuum_gs_figure.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN_DIR = os.path.join(
    "..", "results", "vacuum_gs_ssbroyden_identity_20260610_180827"
)
R_LO, R_HI, Z_LO, Z_HI = 1.0, 2.0, -1.0, 1.0
AREA = (R_HI - R_LO) * (Z_HI - Z_LO)


def _pad_and_stack(seq):
    if not seq:
        return np.empty((0, 0), dtype=np.float64)
    max_len = max(len(a) for a in seq)
    out = np.full((len(seq), max_len), np.nan, dtype=np.float64)
    for i, a in enumerate(seq):
        if len(a) == 0:
            continue
        out[i, : len(a)] = a
        out[i, len(a):] = a[-1]
    return out


def main() -> None:
    d = np.load(os.path.join(RUN_DIR, "raw_histories.npz"))
    seeds = [int(s) for s in d["seeds"]]
    fields = [d[f"field_seed{s}"] for s in seeds]
    J_vals = [np.asarray(d[f"J_val_seed{s}"], dtype=np.float64) for s in seeds]
    sol_l2s = [np.asarray(d[f"sol_l2_seed{s}"], dtype=np.float64) for s in seeds]

    n = fields[0].shape[0]
    rs = np.linspace(R_LO, R_HI, n)
    zs = np.linspace(Z_LO, Z_HI, n)
    RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
    psi_true = RR ** 2
    psi_pred = fields[-1]  # last seed
    err = np.abs(psi_pred - psi_true)
    extent = (R_LO, R_HI, Z_LO, Z_HI)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    im0 = ax[0, 0].imshow(psi_true.T, origin="lower", extent=extent,
                          cmap="viridis", aspect="auto", interpolation="nearest")
    ax[0, 0].set_title(r"$\psi_{\mathrm{exact}}(R, Z) = R^2$")
    fig.colorbar(im0, ax=ax[0, 0], shrink=0.8)

    im1 = ax[0, 1].imshow(psi_pred.T, origin="lower", extent=extent,
                          cmap="viridis", aspect="auto", interpolation="nearest")
    ax[0, 1].set_title(r"$\widehat{\psi}_\theta(R, Z)$ (last seed)")
    fig.colorbar(im1, ax=ax[0, 1], shrink=0.8)

    vmax = float(err.max()) + 1e-14
    vmin = vmax / 100.0  # top two decades; below this is converged noise floor
    im2 = ax[1, 0].imshow(
        err.T, origin="lower", extent=extent, cmap="inferno", aspect="auto",
        interpolation="nearest",
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    ax[1, 0].set_title(r"$|\widehat{\psi}_\theta - \psi_{\mathrm{exact}}|$ (log scale)")
    ax[1, 0].set_xlabel("R")
    ax[1, 0].set_ylabel("Z")
    fig.colorbar(im2, ax=ax[1, 0], shrink=0.8)

    sol_seeds = [s for s in sol_l2s]
    res_seeds = [np.sqrt(AREA * J) for J in J_vals]
    for sol, res in zip(sol_seeds, res_seeds):
        ax[1, 1].semilogy(sol, color="C0", alpha=0.25)
        ax[1, 1].semilogy(res, color="C1", alpha=0.25)
    sol_H = _pad_and_stack(sol_seeds)
    res_H = _pad_and_stack(res_seeds)
    ax[1, 1].semilogy(
        np.nanmean(sol_H, axis=0), color="C0", linewidth=1.8,
        label=r"solution $\|\widehat{\psi}_\theta - \psi_{\mathrm{exact}}\|_{L^2}$",
    )
    ax[1, 1].semilogy(
        np.nanmean(res_H, axis=0), color="C1", linewidth=1.8,
        label=r"residual $\|\Delta^\ast\widehat{\psi}_\theta\|_{L^2}$",
    )
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel(r"$L^2$ error")
    ax[1, 1].set_title(r"Solution and residual $L^2$ error")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(RUN_DIR, "vacuum_gs_results.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Regenerated: {save_path}")


if __name__ == "__main__":
    main()
