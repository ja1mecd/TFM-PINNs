"""
Box-Cox lambda sweep for the 1D PINN.

Runs `pinn_ssbroyden_1d.py` for a sequence of Box-Cox exponents and aggregates
the final L2 errors and convergence histories into a single comparison plot.
The boundaries lambda = 1.0 and lambda = 0.0 of the Box-Cox family correspond
to the identity and the logarithmic loss transforms, with sqrt at lambda = 0.5;
the intermediate values quantify how the curvature amplification g_lambda'(s)
affects convergence depth on the small-loss regime that the PINN reaches after
its Adam warm start.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# pinn_ssbroyden_1d lives in this directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from pinn_ssbroyden_1d import (  # noqa: E402
    DEFAULT_K,
    NeuralNetwork,
    PINN_BVP_SSBroyden,
)


def run_one_lambda(
    lam: float,
    k: float,
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    resample_every: int,
    hidden: tuple[int, ...],
    lr: float,
    seed: int,
    qn_variant: str,
) -> dict:
    """Train one PINN at a given Box-Cox lambda and return final-state metrics + curves."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = NeuralNetwork(hidden_layers=tuple(hidden), activation=nn.Tanh())
    pinn = PINN_BVP_SSBroyden(
        model=model,
        k=k,
        lr=lr,
        loss_transform="boxcox",
        loss_lambda=lam,
        qn_variant=qn_variant,
    )
    pinn.train(
        n_epochs=n_epochs,
        n_collocation=n_collocation,
        train_split=0.8,
        resample_every=resample_every,
        adam_epochs=adam_epochs,
        verbose_freq=max(1, n_epochs // 10),
        diag_grid_n=400,
        patience=n_epochs,
        min_delta=1e-12,
        moving_avg_window=20,
    )

    return {
        "lambda": float(lam),
        "J_train": np.asarray(pinn.J_train, dtype=np.float64),
        "J_val": np.asarray(pinn.J_val, dtype=np.float64),
        "pde_l2": np.asarray(pinn.pde_l2, dtype=np.float64),
        "sol_l2": np.asarray(pinn.sol_l2, dtype=np.float64),
        "sol_rel_l2": np.asarray(pinn.sol_rel_l2, dtype=np.float64),
        "final_J_train": float(pinn.J_train[-1]),
        "final_J_val": float(pinn.J_val[-1]),
        "final_pde_l2": float(pinn.pde_l2[-1]),
        "final_sol_l2": float(pinn.sol_l2[-1]),
        "final_sol_rel_l2": float(pinn.sol_rel_l2[-1]),
    }


def plot_sweep(results: list[dict], out_path: str, k: float, adam_epochs: int) -> None:
    """Three-panel comparison: J(val), solution L2, and final-error bar chart."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    cmap = plt.get_cmap("viridis")
    n = len(results)

    for i, r in enumerate(results):
        c = cmap(i / max(n - 1, 1))
        label = rf"$\lambda = {r['lambda']:g}$"
        epochs = np.arange(1, len(r["J_val"]) + 1)
        ax[0].semilogy(epochs, r["J_val"], color=c, label=label, linewidth=1.4)
        ax[1].semilogy(epochs, r["sol_l2"], color=c, label=label, linewidth=1.4)

    ax[0].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel(r"raw validation $\mathcal{J}$")
    ax[0].set_title("Validation residual MSE")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=9)

    ax[1].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5, label="Adam $\\to$ SSBroyden")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(r"$\|\widehat{u}_\theta - u_{\mathrm{exact}}\|_{L^2}$")
    ax[1].set_title("Solution $L^2$ error")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=9)

    lams = np.asarray([r["lambda"] for r in results])
    finals_J = np.asarray([r["final_J_val"] for r in results])
    finals_sol = np.asarray([r["final_sol_l2"] for r in results])
    width = 0.35
    pos = np.arange(n)
    ax[2].bar(pos - width / 2, finals_J, width, label=r"final val $\mathcal{J}$", color="C3")
    ax[2].bar(pos + width / 2, finals_sol, width, label=r"final $\|u-u^*\|_{L^2}$", color="C0")
    ax[2].set_yscale("log")
    ax[2].set_xticks(pos)
    ax[2].set_xticklabels([f"{lam:g}" for lam in lams])
    ax[2].set_xlabel(r"Box-Cox $\lambda$")
    ax[2].set_title(f"Final metrics (k = {k:g})")
    ax[2].grid(True, alpha=0.3, axis="y")
    ax[2].legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved sweep figure to: {out_path}")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Box-Cox lambda sweep for the 1D PINN.")
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
        help="Box-Cox exponents to scan (each is a separate training run).",
    )
    p.add_argument("--wavenumber", type=float, default=DEFAULT_K)
    p.add_argument(
        "--qn-variant",
        type=str,
        default="ssbroyden",
        choices=["bfgs", "ssbfgs", "ssbroyden"],
    )
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--adam-epochs", type=int, default=2000)
    p.add_argument("--n-collocation", type=int, default=400)
    p.add_argument("--resample-every", type=int, default=500)
    p.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[32, 32, 32],
        help=(
            "Hidden-layer widths. Default 3x32 matches the architecture "
            "used by optimiser_comparison_1d.py and documented in section "
            "4.2 of the thesis."
        ),
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join("..", "results"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        f"\nBox-Cox lambda sweep on the 1D BVP "
        f"(k = {args.wavenumber:g}, qn = {args.qn_variant}, "
        f"epochs = {args.epochs}, adam = {args.adam_epochs}).\n"
        f"Lambdas to scan: {args.lambdas}\n"
    )

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(
        args.results_dir,
        f"bvp1d_k{args.wavenumber:g}_boxcox_sweep_{args.qn_variant}_{run_tag}",
    )
    os.makedirs(sweep_dir, exist_ok=True)

    results: list[dict] = []
    for lam in args.lambdas:
        print("\n" + "=" * 72)
        print(f"  RUN  lambda = {lam:g}")
        print("=" * 72)
        r = run_one_lambda(
            lam=lam,
            k=args.wavenumber,
            n_epochs=args.epochs,
            adam_epochs=args.adam_epochs,
            n_collocation=args.n_collocation,
            resample_every=args.resample_every,
            hidden=tuple(args.hidden),
            lr=args.lr,
            seed=args.seed,
            qn_variant=args.qn_variant,
        )
        results.append(r)

    # Save aggregated NPZ + per-lambda summary table.
    np.savez(
        os.path.join(sweep_dir, "sweep_history.npz"),
        lambdas=np.asarray([r["lambda"] for r in results], dtype=np.float64),
        **{
            f"J_val_{r['lambda']:g}".replace(".", "p"): r["J_val"] for r in results
        },
        **{
            f"sol_l2_{r['lambda']:g}".replace(".", "p"): r["sol_l2"] for r in results
        },
        **{
            f"pde_l2_{r['lambda']:g}".replace(".", "p"): r["pde_l2"] for r in results
        },
    )
    table_path = os.path.join(sweep_dir, "summary_table.txt")
    with open(table_path, "w") as fh:
        fh.write(
            f"Box-Cox sweep — k={args.wavenumber:g}, qn={args.qn_variant}, "
            f"epochs={args.epochs} (adam={args.adam_epochs})\n"
        )
        fh.write(
            f"{'lambda':>10} {'final J(val)':>16} {'final pdeL2':>16} "
            f"{'final solL2':>16} {'final relL2':>16}\n"
        )
        for r in results:
            fh.write(
                f"{r['lambda']:>10.4g} {r['final_J_val']:>16.4e} "
                f"{r['final_pde_l2']:>16.4e} {r['final_sol_l2']:>16.4e} "
                f"{r['final_sol_rel_l2']:>16.4e}\n"
            )
    with open(table_path) as fh:
        print("\n" + fh.read())
    print(f"Saved summary table to: {table_path}")

    plot_sweep(
        results,
        out_path=os.path.join(sweep_dir, "boxcox_sweep.png"),
        k=args.wavenumber,
        adam_epochs=args.adam_epochs,
    )

    print(f"\nAll sweep artefacts written to: {os.path.abspath(sweep_dir)}")


if __name__ == "__main__":
    main()
