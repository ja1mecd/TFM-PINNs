"""
Fine-grained Box-Cox sweep on the 2D Helmholtz benchmark of Urban et al. 2025.

The 1D BVP at k = 4 forces J ~ 10^4 throughout the SSBroyden phase, so
g''_lambda(J) is numerically negligible regardless of lambda — Box-Cox simply
cannot deliver curvature amplification in that regime. The 2D Helmholtz
benchmark with (a1, a2) = (1, 4), k = 1 is the configuration the paper uses
to argue for log/sqrt transformations: SSBroyden does drive J into the
small-loss regime where g''_lambda actually fires. This script reuses the
existing `PINN_Helmholtz_Solver` (which already implements the same Box-Cox
transformation with the expm1/log stable form) and sweeps lambda on a finer
grid with multi-seed averaging.

Run with default settings to reproduce the paper's regime:

    python boxcox_sweep_2d_helmholtz.py

For a quicker smoke test:

    python boxcox_sweep_2d_helmholtz.py --epochs 5000 --adam-epochs 2000 \
        --n-collocation 2000 --seeds 42 43

Outputs go to `../results/helmholtz2d_boxcox_finesweep_<timestamp>/`.
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
from pinn_helmholtz_2d import (  # noqa: E402
    A1_WAVENUMBER,
    A2_WAVENUMBER,
    K_WAVENUMBER,
    NeuralNetwork,
    PINN_Helmholtz_Solver,
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
    a1: int,
    a2: int,
    k: float,
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    hidden: tuple[int, ...],
    lr: float,
    qn_variant: str,
    handover_strategy: str,
    handover_max_adam_epochs: int,
    plateau_patience: int,
    plateau_min_delta: float,
    patience: int,
) -> SeedResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = NeuralNetwork(hidden_layers=tuple(hidden), activation=nn.Tanh())
    pinn = PINN_Helmholtz_Solver(
        model=model,
        lr=lr,
        a1=a1, a2=a2, k=k,
        loss_transform="boxcox",
        loss_lambda=lam,
        qn_variant=qn_variant,
    )
    pinn.train(
        n_epochs=n_epochs,
        n_collocation=n_collocation,
        adam_epochs=adam_epochs,
        verbose_freq=max(1, n_epochs // 5),
        diag_grid_n=60,
        handover_strategy=handover_strategy,
        handover_max_adam_epochs=handover_max_adam_epochs,
        plateau_patience=plateau_patience,
        plateau_min_delta=plateau_min_delta,
        patience=patience,
        min_delta=1e-12,
        moving_avg_window=20,
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
    a1: int,
    a2: int,
    k: float,
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    hidden: tuple[int, ...],
    lr: float,
    qn_variant: str,
    handover_strategy: str,
    handover_max_adam_epochs: int,
    plateau_patience: int,
    plateau_min_delta: float,
    patience: int,
) -> tuple[LambdaResult, ...]:
    out: list[LambdaResult] = []
    for lam in lambdas:
        seed_runs: list[SeedResult] = []
        for s in seeds:
            print(
                f"\n[2DH lambda={lam:g}, seed={s}]  "
                f"handover={handover_strategy}, qn_patience={patience}"
            )
            seed_runs.append(run_one(
                lam=lam, seed=s,
                a1=a1, a2=a2, k=k,
                n_epochs=n_epochs, adam_epochs=adam_epochs,
                n_collocation=n_collocation,
                hidden=hidden, lr=lr,
                qn_variant=qn_variant,
                handover_strategy=handover_strategy,
                handover_max_adam_epochs=handover_max_adam_epochs,
                plateau_patience=plateau_patience,
                plateau_min_delta=plateau_min_delta,
                patience=patience,
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
    a1: int, a2: int, k: float,
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

    ax[0, 0].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5, label="Adam$\\to$SSBroyden")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel(r"$\mathcal{J}_{\mathrm{val}}$ (mean over seeds)")
    ax[0, 0].set_title(f"2D Helmholtz, (a1, a2)=({a1}, {a2}), k={k:g}")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend(fontsize=8, ncol=2)

    ax[0, 1].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5)
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel(r"$\|\widehat{u} - u^\star\|_{L^2}$")
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
                      label=r"final $\|u-u^*\|_{L^2}$ mean $\pm$ std")
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
    a1: int, a2: int, k: float,
    qn_variant: str,
    n_epochs: int,
    adam_epochs: int,
    seeds: tuple[int, ...],
    hidden: tuple[int, ...] = (),
) -> None:
    arch = "x".join(str(h) for h in hidden) if hidden else "?"
    lines: list[str] = []
    lines.append(
        f"2D Helmholtz Box-Cox sweep, (a1, a2)=({a1}, {a2}), k={k:g}, "
        f"net={arch}, qn={qn_variant}, epochs={n_epochs} (adam={adam_epochs}), "
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
        description="Fine-grained Box-Cox sweep on the 2D Helmholtz benchmark."
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
    p.add_argument("--a1", type=int, default=A1_WAVENUMBER)
    p.add_argument("--a2", type=int, default=A2_WAVENUMBER)
    p.add_argument("--k", type=float, default=K_WAVENUMBER)
    p.add_argument(
        "--qn-variant",
        type=str,
        default="ssbroyden",
        choices=["bfgs", "ssbfgs", "ssbroyden"],
    )
    p.add_argument("--epochs", type=int, default=10000,
                   help="Total budget cap (2000 Adam + up to 8000 QN); QN-phase "
                        "early stopping ends most runs well before this.")
    p.add_argument("--adam-epochs", type=int, default=2000,
                   help="Standardised fixed Adam warm-up before handover.")
    p.add_argument("--n-collocation", type=int, default=10000)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32, 32])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--handover-strategy",
        type=str,
        default="fixed",
        choices=["fixed", "plateau", "loss_threshold", "gradnorm"],
        help="Adam -> SSBroyden trigger. Default 'fixed': switch at exactly "
             "--adam-epochs (the standardised 2000-epoch convention) so every "
             "seed shares an identical Adam phase.",
    )
    p.add_argument("--handover-max-adam-epochs", type=int, default=10000)
    p.add_argument("--plateau-patience", type=int, default=200)
    p.add_argument("--plateau-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--patience",
        type=int,
        default=500,
        help="QN-phase early-stop patience. Counted only AFTER handover, so "
             "a stalled Adam phase cannot terminate a sweep pair before "
             "SSBroyden engages. Set --patience >= --epochs to disable.",
    )
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
        f"helmholtz2d_a{args.a1}_{args.a2}_k{args.k:g}_boxcox_finesweep_{args.qn_variant}_{run_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"\n2D Helmholtz Box-Cox sweep "
        f"((a1,a2)=({args.a1}, {args.a2}), k={args.k:g}, "
        f"qn={args.qn_variant}, epochs={args.epochs}, adam={args.adam_epochs})\n"
        f"  lambdas: {lambdas}\n"
        f"  seeds:   {seeds}\n"
    )

    results = run_sweep(
        lambdas=lambdas, seeds=seeds,
        a1=args.a1, a2=args.a2, k=args.k,
        n_epochs=args.epochs, adam_epochs=args.adam_epochs,
        n_collocation=args.n_collocation,
        hidden=tuple(args.hidden), lr=args.lr,
        qn_variant=args.qn_variant,
        handover_strategy=args.handover_strategy,
        handover_max_adam_epochs=args.handover_max_adam_epochs,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        patience=args.patience,
    )

    write_summary(
        results=results,
        out_path=os.path.join(out_dir, "summary_table.txt"),
        a1=args.a1, a2=args.a2, k=args.k,
        qn_variant=args.qn_variant,
        n_epochs=args.epochs, adam_epochs=args.adam_epochs,
        seeds=seeds,
        hidden=tuple(args.hidden),
    )
    plot_sweep(
        results=results,
        out_path=os.path.join(out_dir, "boxcox_sweep_2d_helmholtz.png"),
        a1=args.a1, a2=args.a2, k=args.k,
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
