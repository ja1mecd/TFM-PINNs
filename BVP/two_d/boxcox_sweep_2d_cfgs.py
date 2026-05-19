"""
Fine-grained Box-Cox sweep on the current-free Grad-Shafranov (CFGS)
benchmark of Urban et al. 2025.

This is the *exact* problem on which the paper runs its loss-function
transformations: section 4.1.2 ("Impact of the loss function") trains three
identical networks with J, sqrt(J) and log(J) on the current-free
Grad-Shafranov equation, and Table 2 reports the resulting relative errors.
The square root is tested *only* on this problem in the paper, so it is the
proper place to reproduce the J_1/2 (lambda=0.5) and J_log (lambda=0) cases
and to sweep the full Box-Cox family that interpolates between them.

This script is the CFGS counterpart of ``boxcox_sweep_2d_helmholtz.py``. It
reuses the existing ``PINN_CFGS_Solver_Urban`` (which already implements the
same Box-Cox transformation with the expm1/log stable form, applied only
during the quasi-Newton phase exactly as the paper does — Adam still
optimises the raw MSE) and sweeps lambda on a finer grid with multi-seed
averaging.

Run with default settings to reproduce the paper's regime:

    python boxcox_sweep_2d_cfgs.py

For a quicker smoke test:

    python boxcox_sweep_2d_cfgs.py --epochs 2000 --adam-epochs 800 \
        --n-collocation 500 --seeds 42 43

Outputs go to `../results/cfgs_urban_<variant>_boxcox_finesweep_<timestamp>/`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from pinn_ssbroyden_2d_urban import (  # noqa: E402
    NeuralNetwork,
    PINN_CFGS_Solver_Urban,
)


@dataclass(frozen=True)
class SeedResult:
    seed: int
    final_J_val: float
    final_sol_l2: float
    final_sol_rel_l2: float
    J_val: np.ndarray
    sol_l2: np.ndarray


@dataclass(frozen=True)
class LambdaResult:
    lambda_: float
    seeds: tuple[SeedResult, ...]


def run_one(
    lam: float,
    seed: int,
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    hidden: tuple[int, ...],
    lr: float,
    variant: str,
    train_split: float,
    rad_resample_every: int,
    rad_pool_size: int,
    rad_k1: float,
    rad_k2: float,
    initial_scale: bool,
    h_on_cpu: bool,
    diag_every: int,
) -> SeedResult:
    # Seed before model construction so weight init is reproducible per seed;
    # train() re-seeds the data sampling with the same value internally.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = NeuralNetwork(hidden_layers=tuple(hidden), activation=nn.Tanh())
    pinn = PINN_CFGS_Solver_Urban(
        model,
        lr_adam=lr,
        variant=variant,
        loss_transform="boxcox",
        loss_lambda=lam,
        initial_scale=initial_scale,
        H_on_cpu=h_on_cpu,
    )
    pinn.train(
        n_epochs=n_epochs,
        adam_epochs=adam_epochs,
        n_collocation=n_collocation,
        train_split=train_split,
        rad_resample_every=rad_resample_every,
        rad_pool_size=rad_pool_size,
        rad_k1=rad_k1,
        rad_k2=rad_k2,
        verbose_freq=max(1, n_epochs // 5),
        diag_grid_n=60,
        diag_every=diag_every,
        seed=seed,
    )
    return SeedResult(
        seed=seed,
        final_J_val=float(pinn.J_val[-1]),
        final_sol_l2=float(pinn.sol_l2[-1]) if pinn.sol_l2 else float("nan"),
        final_sol_rel_l2=float(pinn.sol_rel_l2[-1]) if pinn.sol_rel_l2 else float("nan"),
        J_val=np.asarray(pinn.J_val, dtype=np.float64),
        sol_l2=np.asarray(pinn.sol_l2, dtype=np.float64),
    )


def run_sweep(
    lambdas: tuple[float, ...],
    seeds: tuple[int, ...],
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    hidden: tuple[int, ...],
    lr: float,
    variant: str,
    train_split: float,
    rad_resample_every: int,
    rad_pool_size: int,
    rad_k1: float,
    rad_k2: float,
    initial_scale: bool,
    h_on_cpu: bool,
    diag_every: int,
) -> tuple[LambdaResult, ...]:
    out: list[LambdaResult] = []
    for lam in lambdas:
        seed_runs: list[SeedResult] = []
        for s in seeds:
            print(f"\n[CFGS lambda={lam:g}, seed={s}]  variant={variant}")
            seed_runs.append(run_one(
                lam=lam, seed=s,
                n_epochs=n_epochs, adam_epochs=adam_epochs,
                n_collocation=n_collocation,
                hidden=hidden, lr=lr,
                variant=variant,
                train_split=train_split,
                rad_resample_every=rad_resample_every,
                rad_pool_size=rad_pool_size,
                rad_k1=rad_k1, rad_k2=rad_k2,
                initial_scale=initial_scale,
                h_on_cpu=h_on_cpu,
                diag_every=diag_every,
            ))
        out.append(LambdaResult(lambda_=lam, seeds=tuple(seed_runs)))
    return tuple(out)


def _pad_and_stack(seq: list[np.ndarray]) -> np.ndarray:
    """Stack possibly variable-length histories into (n, max_len), padding
    each shorter run with its last value so early-stopped trajectories
    coexist cleanly with full-budget ones in the mean curve."""
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


def plot_sweep(
    results: tuple[LambdaResult, ...],
    out_path: str,
    variant: str,
    adam_epochs: int,
) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    cmap = plt.get_cmap("viridis")
    n = len(results)

    for i, lr in enumerate(results):
        c = cmap(i / max(n - 1, 1))
        H = _pad_and_stack([s.J_val for s in lr.seeds])
        mean = np.nanmean(H, axis=0)
        ax[0, 0].semilogy(mean, color=c, linewidth=1.4, label=rf"$\lambda={lr.lambda_:g}$")

        S = _pad_and_stack([s.sol_l2 for s in lr.seeds])
        smean = np.nanmean(S, axis=0)
        ax[0, 1].semilogy(smean, color=c, linewidth=1.4, label=rf"$\lambda={lr.lambda_:g}$")

    ax[0, 0].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5, label="Adam$\\to$QN")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel(r"$\mathcal{J}_{\mathrm{val}}$ (mean over seeds)")
    ax[0, 0].set_title(f"CFGS (current-free Grad-Shafranov), {variant}")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend(fontsize=8, ncol=2)

    ax[0, 1].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5)
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel(r"$\|P_{\mathrm{NN}} - P^\star\|_{L^2}$")
    ax[0, 1].set_title("Solution error trajectory")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend(fontsize=8, ncol=2)

    lams = np.asarray([r.lambda_ for r in results])
    means_J = np.asarray([float(np.mean([s.final_J_val for s in r.seeds])) for r in results])
    stds_J = np.asarray([float(np.std([s.final_J_val for s in r.seeds])) for r in results])
    means_S = np.asarray([float(np.mean([s.final_sol_l2 for s in r.seeds])) for r in results])
    stds_S = np.asarray([float(np.std([s.final_sol_l2 for s in r.seeds])) for r in results])

    ax[1, 0].errorbar(lams, means_J, yerr=stds_J, fmt="o-", color="C3",
                      label=r"final $\mathcal{J}_{\mathrm{val}}$ mean $\pm$ std")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel(r"Box-Cox $\lambda$")
    ax[1, 0].set_ylabel(r"final $\mathcal{J}_{\mathrm{val}}$")
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].legend(fontsize=9)
    ax[1, 0].set_title("Final residual vs lambda")

    ax[1, 1].errorbar(lams, means_S, yerr=stds_S, fmt="s-", color="C0",
                      label=r"final $\|P-P^*\|_{L^2}$ mean $\pm$ std")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel(r"Box-Cox $\lambda$")
    ax[1, 1].set_ylabel(r"final solution $L^2$ error")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].legend(fontsize=9)
    ax[1, 1].set_title("Solution error vs lambda")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sweep figure to: {out_path}")


def write_summary(
    results: tuple[LambdaResult, ...],
    out_path: str,
    variant: str,
    n_epochs: int,
    adam_epochs: int,
    seeds: tuple[int, ...],
) -> None:
    lines: list[str] = []
    lines.append(
        f"CFGS (current-free Grad-Shafranov) Box-Cox sweep, "
        f"variant={variant}, epochs={n_epochs} (adam={adam_epochs}), "
        f"seeds={list(seeds)}\n\n"
    )
    lines.append(
        f"{'lambda':>8}   "
        f"{'mean J':>14}  {'std J':>14}    "
        f"{'mean solL2':>14}  {'std solL2':>14}\n"
    )
    for r in results:
        mJ = float(np.mean([s.final_J_val for s in r.seeds]))
        sJ = float(np.std([s.final_J_val for s in r.seeds]))
        mS = float(np.mean([s.final_sol_l2 for s in r.seeds]))
        sS = float(np.std([s.final_sol_l2 for s in r.seeds]))
        lines.append(
            f"{r.lambda_:>8.4g}   {mJ:>14.4e}  {sJ:>14.4e}    "
            f"{mS:>14.4e}  {sS:>14.4e}\n"
        )
    text = "".join(lines)
    with open(out_path, "w") as fh:
        fh.write(text)
    print("\n" + text)
    print(f"Saved summary to: {out_path}")


DEFAULT_LAMBDAS: tuple[float, ...] = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-grained Box-Cox sweep on the current-free "
                    "Grad-Shafranov benchmark (Urban et al. 2025, sec. 4.1.2)."
    )
    p.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    p.add_argument(
        "--lambdas-linspace",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "N"),
        default=None,
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument(
        "--variant",
        type=str,
        default="SSBroyden2",
        choices=[
            "BFGS", "BFGS_scipy", "SSBFGS_OL", "SSBFGS_AB",
            "SSBroyden1", "SSBroyden2", "SSBroyden3",
        ],
    )
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--adam-epochs", type=int, default=2000)
    p.add_argument("--n-collocation", type=int, default=1000)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--rad-resample-every", type=int, default=500)
    p.add_argument("--rad-pool-size", type=int, default=10000)
    p.add_argument("--rad-k1", type=float, default=1.0)
    p.add_argument("--rad-k2", type=float, default=1.0)
    p.add_argument("--hidden", type=int, nargs="+", default=[30])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--initial-scale", action="store_true")
    p.add_argument(
        "--diag-every", type=int, default=100,
        help="Grid-diagnostic recompute cadence in epochs (pde/sol L2). "
             "Keeps trajectory curves smooth while skipping most steps. "
             "0 -> fall back to the (coarse) verbose cadence.",
    )
    p.add_argument("--H-on-cpu", action="store_true")
    p.add_argument("--results-dir", type=str, default=os.path.join("..", "results"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.lambdas_linspace is not None:
        start, stop, n = args.lambdas_linspace
        lambdas = tuple(float(v) for v in np.linspace(start, stop, int(n)))
    else:
        lambdas = tuple(args.lambdas)

    seeds = tuple(args.seeds)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.results_dir,
        f"cfgs_urban_{args.variant}_boxcox_finesweep_{run_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"\nCFGS (current-free Grad-Shafranov) Box-Cox sweep "
        f"(variant={args.variant}, epochs={args.epochs}, "
        f"adam={args.adam_epochs})\n"
        f"  lambdas: {lambdas}\n"
        f"  seeds:   {seeds}\n"
    )

    results = run_sweep(
        lambdas=lambdas, seeds=seeds,
        n_epochs=args.epochs, adam_epochs=args.adam_epochs,
        n_collocation=args.n_collocation,
        hidden=tuple(args.hidden), lr=args.lr,
        variant=args.variant,
        train_split=args.train_split,
        rad_resample_every=args.rad_resample_every,
        rad_pool_size=args.rad_pool_size,
        rad_k1=args.rad_k1, rad_k2=args.rad_k2,
        initial_scale=args.initial_scale,
        h_on_cpu=args.H_on_cpu,
        diag_every=args.diag_every,
    )

    write_summary(
        results=results,
        out_path=os.path.join(out_dir, "summary_table.txt"),
        variant=args.variant,
        n_epochs=args.epochs, adam_epochs=args.adam_epochs,
        seeds=seeds,
    )
    plot_sweep(
        results=results,
        out_path=os.path.join(out_dir, "boxcox_sweep_2d_cfgs.png"),
        variant=args.variant,
        adam_epochs=args.adam_epochs,
    )

    np.savez(
        os.path.join(out_dir, "raw_histories.npz"),
        lambdas=np.asarray(lambdas, dtype=np.float64),
        seeds=np.asarray(seeds, dtype=np.int64),
        **{
            f"J_val_lam{lr.lambda_:g}_seed{s.seed}".replace(".", "p").replace("-", "m"):
                s.J_val for lr in results for s in lr.seeds
        },
        **{
            f"sol_l2_lam{lr.lambda_:g}_seed{s.seed}".replace(".", "p").replace("-", "m"):
                s.sol_l2 for lr in results for s in lr.seeds
        },
    )
    print(f"All sweep artefacts written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
