"""
Fine-grained Box-Cox sweep on the vacuum Grad-Shafranov psi = R^2 validation
problem (section 4.3.2 of the thesis).

This is the counterpart of ``boxcox_sweep_2d_cfgs.py`` and
``boxcox_sweep_2d_helmholtz.py``, built on the same standardised protocol
(fixed 2000-epoch Adam warm-up -> SSBroyden, Box-Cox engaged only from the
start of the QN phase, QN-phase early stopping, three seeds).

Purpose: a probe. The vacuum GS run reaches J ~ 1e-11 in a clean two-stage
descent, so it should be *stagnation-limited* like CFGS and small lambda should
help. The interesting angle is conditioning: the box R in [1,2] keeps the 1/R
coefficient regular, so this operator is *well-conditioned*, unlike CFGS
(kappa ~ 1e12). If small lambda helps here too, the regime is set by basin
regularity, not raw conditioning.

Run:
    python boxcox_sweep_2d_vacuum_gs.py

Quick smoke test:
    python boxcox_sweep_2d_vacuum_gs.py --epochs 2000 --adam-epochs 800 \
        --n-collocation 500 --seeds 42 43

Outputs go to `../results/vacuum_gs_ssbroyden_boxcox_finesweep_<timestamp>/`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from pinn_vacuum_gs_2d import Net, VacuumGSPINN  # noqa: E402


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
    diag_grid_n: int,
    es_patience: int,
    es_min_delta: float,
    es_window: int,
) -> SeedResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = Net(hidden=tuple(hidden))
    pinn = VacuumGSPINN(
        model,
        lr=lr,
        loss_transform="boxcox",
        loss_lambda=lam,
        qn_variant="ssbroyden",
    )
    pinn.train(
        n_epochs=n_epochs,
        adam_epochs=adam_epochs,
        n_collocation=n_collocation,
        verbose_freq=max(1, n_epochs // 5),
        diag_grid_n=diag_grid_n,
        handover_strategy="fixed",
        early_stop=True,
        es_patience=es_patience,
        es_min_delta=es_min_delta,
        es_window=es_window,
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
    diag_grid_n: int,
    es_patience: int,
    es_min_delta: float,
    es_window: int,
) -> tuple[LambdaResult, ...]:
    out: list[LambdaResult] = []
    for lam in lambdas:
        seed_runs: list[SeedResult] = []
        for s in seeds:
            print(f"\n[vacuum-GS lambda={lam:g}, seed={s}]")
            seed_runs.append(run_one(
                lam=lam, seed=s,
                n_epochs=n_epochs, adam_epochs=adam_epochs,
                n_collocation=n_collocation,
                hidden=hidden, lr=lr,
                diag_grid_n=diag_grid_n,
                es_patience=es_patience,
                es_min_delta=es_min_delta,
                es_window=es_window,
            ))
        out.append(LambdaResult(lambda_=lam, seeds=tuple(seed_runs)))
    return tuple(out)


def _pad_and_stack(seq: list[np.ndarray]) -> np.ndarray:
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
    adam_epochs: int,
) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    cmap = plt.get_cmap("viridis")
    n = len(results)

    for i, lr in enumerate(results):
        c = cmap(i / max(n - 1, 1))
        H = _pad_and_stack([s.J_val for s in lr.seeds])
        ax[0, 0].semilogy(np.nanmean(H, axis=0), color=c, linewidth=1.4,
                          label=rf"$\lambda={lr.lambda_:g}$")
        S = _pad_and_stack([s.sol_l2 for s in lr.seeds])
        ax[0, 1].semilogy(np.nanmean(S, axis=0), color=c, linewidth=1.4,
                          label=rf"$\lambda={lr.lambda_:g}$")

    ax[0, 0].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5, label="Adam$\\to$QN")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel(r"$\mathcal{J}_{\mathrm{val}}$ (mean over seeds)")
    ax[0, 0].set_title(r"Vacuum Grad-Shafranov ($\psi=R^2$), SSBroyden")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend(fontsize=8, ncol=2)

    ax[0, 1].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5)
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel(r"$\|\widehat{\psi}_\theta - \psi_{\mathrm{exact}}\|_{L^2}$")
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
                      label=r"final $\|\widehat{\psi}_\theta-\psi_{\mathrm{exact}}\|_{L^2}$ mean $\pm$ std")
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
    n_epochs: int,
    adam_epochs: int,
    seeds: tuple[int, ...],
    hidden: tuple[int, ...] = (),
) -> None:
    arch = "x".join(str(h) for h in hidden) if hidden else "?"
    lines: list[str] = []
    lines.append(
        f"Vacuum Grad-Shafranov (psi=R^2) Box-Cox sweep, "
        f"net={arch}, variant=ssbroyden, epochs={n_epochs} (adam={adam_epochs}), "
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
        description="Fine-grained Box-Cox sweep on the vacuum Grad-Shafranov "
                    "psi=R^2 validation (thesis sec. 4.3.2)."
    )
    p.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    p.add_argument(
        "--lambdas-linspace", type=float, nargs=3,
        metavar=("START", "STOP", "N"), default=None,
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--adam-epochs", type=int, default=2000)
    p.add_argument("--n-collocation", type=int, default=2000)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32, 32])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--diag-grid-n", type=int, default=100,
        help="Grid size for the per-epoch solution-L2 diagnostic.",
    )
    p.add_argument("--es-patience", type=int, default=300)
    p.add_argument("--es-min-delta", type=float, default=1e-4)
    p.add_argument("--es-window", type=int, default=20)
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
        f"vacuum_gs_ssbroyden_boxcox_finesweep_{run_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"\nVacuum Grad-Shafranov (psi=R^2) Box-Cox sweep "
        f"(epochs={args.epochs}, adam={args.adam_epochs})\n"
        f"  lambdas: {lambdas}\n"
        f"  seeds:   {seeds}\n"
    )

    results = run_sweep(
        lambdas=lambdas, seeds=seeds,
        n_epochs=args.epochs, adam_epochs=args.adam_epochs,
        n_collocation=args.n_collocation,
        hidden=tuple(args.hidden), lr=args.lr,
        diag_grid_n=args.diag_grid_n,
        es_patience=args.es_patience,
        es_min_delta=args.es_min_delta,
        es_window=args.es_window,
    )

    write_summary(
        results=results,
        out_path=os.path.join(out_dir, "summary_table.txt"),
        n_epochs=args.epochs, adam_epochs=args.adam_epochs,
        seeds=seeds,
        hidden=tuple(args.hidden),
    )
    plot_sweep(
        results=results,
        out_path=os.path.join(out_dir, "boxcox_sweep_2d_vacuum_gs.png"),
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
