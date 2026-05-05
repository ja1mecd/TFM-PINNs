"""
Optimiser comparison on the 1D BVP -u'' = (k pi)^2 sin(k pi x).

Trains four optimisation pipelines on the *same* network architecture, the
*same* collocation seeds, and the *same* total iteration budget, and reports
mean +/- std of the final L^2 residual and L^2 solution error across seeds:

    - "adam"             pure Adam for the full budget;
    - "adam_bfgs"        Adam warm start, then standard BFGS;
    - "adam_ssbfgs"      Adam warm start, then self-scaled BFGS;
    - "adam_ssbroyden"   Adam warm start, then self-scaled Broyden.

This is the head-to-head comparison flagged as bullet 1 of section 5.2 of the
thesis, and is the experiment that lets the conditioning argument of chapter
3 land empirically. The hard Dirichlet ansatz of `pinn_ssbroyden_1d.py` is
used throughout, so the boundary residual is identically zero by construction
and any visible difference in the final accuracy is attributable to the
optimiser (not to the boundary term, which would otherwise dominate the
ill-conditioning analysis).

A single multi-panel figure compares the four pipelines:

    - validation J(epoch), mean across seeds
    - solution L^2 error, mean across seeds
    - bar plot of final residual / solution error with error bars
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
from pinn_ssbroyden_1d import (  # noqa: E402
    DEFAULT_K,
    NeuralNetwork,
    PINN_BVP_SSBroyden,
)
from pinn_adaptive_handover import (  # noqa: E402
    HANDOVER_STRATEGIES,
    PINN_BVP_AdaptiveHandover,
)


PIPELINES: tuple[str, ...] = ("adam", "adam_bfgs", "adam_ssbfgs", "adam_ssbroyden")
PIPELINE_LABEL: dict[str, str] = {
    "adam":           "Adam (full budget)",
    "adam_bfgs":      "Adam $\\to$ BFGS",
    "adam_ssbfgs":    "Adam $\\to$ SSBFGS",
    "adam_ssbroyden": "Adam $\\to$ SSBroyden",
}
PIPELINE_COLOR: dict[str, str] = {
    "adam":           "C7",
    "adam_bfgs":      "C2",
    "adam_ssbfgs":    "C1",
    "adam_ssbroyden": "C0",
}


@dataclass(frozen=True)
class SeedRun:
    seed: int
    J_val_history: np.ndarray
    sol_l2_history: np.ndarray
    final_J_val: float
    final_pde_l2: float
    final_sol_l2: float
    final_sol_rel_l2: float


@dataclass(frozen=True)
class PipelineResult:
    pipeline: str
    seeds: tuple[SeedRun, ...]


# =============================================================================
# Single seed run for a pipeline
# =============================================================================
def run_pipeline_once(
    pipeline: str,
    seed: int,
    k: float,
    total_epochs: int,
    adam_warmup: int,
    n_collocation: int,
    hidden: tuple[int, ...],
    lr: float,
    handover_strategy: str,
    handover_max_adam_epochs: int,
    plateau_patience: int,
    plateau_min_delta: float,
    patience: int,
) -> SeedRun:
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}; valid: {PIPELINES}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = NeuralNetwork(hidden_layers=tuple(hidden), activation=nn.Tanh())

    if pipeline == "adam":
        # Pure Adam: adam_epochs == n_epochs so the QN phase is empty.
        # The base class now accepts adam_epochs in [0, n_epochs] and prints
        # "[pure Adam]" instead of a phantom "(1 iters)" QN phase.
        pinn = PINN_BVP_SSBroyden(
            model=model, k=k, lr=lr,
            loss_transform="identity",
            qn_variant="ssbroyden",  # never reached
        )
        pinn.train(
            n_epochs=total_epochs,
            n_collocation=n_collocation,
            train_split=0.8,
            resample_every=500,
            adam_epochs=total_epochs,
            verbose_freq=max(1, total_epochs // 5),
            diag_grid_n=400,
            patience=patience,
            min_delta=1e-12,
            moving_avg_window=20,
        )
    else:
        qn = {
            "adam_bfgs": "bfgs",
            "adam_ssbfgs": "ssbfgs",
            "adam_ssbroyden": "ssbroyden",
        }[pipeline]
        pinn = PINN_BVP_AdaptiveHandover(
            model=model, k=k, lr=lr,
            loss_transform="identity",
            qn_variant=qn,
        )
        pinn.train(
            n_epochs=total_epochs,
            n_collocation=n_collocation,
            train_split=0.8,
            resample_every=500,
            adam_epochs=adam_warmup,
            verbose_freq=max(1, total_epochs // 5),
            diag_grid_n=400,
            patience=patience,
            min_delta=1e-12,
            moving_avg_window=20,
            handover_strategy=handover_strategy,
            handover_max_adam_epochs=handover_max_adam_epochs,
            plateau_patience=plateau_patience,
            plateau_min_delta=plateau_min_delta,
        )

    return SeedRun(
        seed=seed,
        J_val_history=np.asarray(pinn.J_val, dtype=np.float64),
        sol_l2_history=np.asarray(pinn.sol_l2, dtype=np.float64),
        final_J_val=float(pinn.J_val[-1]),
        final_pde_l2=float(pinn.pde_l2[-1]),
        final_sol_l2=float(pinn.sol_l2[-1]),
        final_sol_rel_l2=float(pinn.sol_rel_l2[-1]),
    )


# =============================================================================
# Sweep
# =============================================================================
def run_comparison(
    pipelines: tuple[str, ...],
    seeds: tuple[int, ...],
    k: float,
    total_epochs: int,
    adam_warmup: int,
    n_collocation: int,
    hidden: tuple[int, ...],
    lr: float,
    handover_strategy: str,
    handover_max_adam_epochs: int,
    plateau_patience: int,
    plateau_min_delta: float,
    patience: int,
) -> tuple[PipelineResult, ...]:
    results: list[PipelineResult] = []
    for p in pipelines:
        runs: list[SeedRun] = []
        for s in seeds:
            print(f"\n[pipeline={p}, seed={s}]")
            run = run_pipeline_once(
                pipeline=p,
                seed=s,
                k=k,
                total_epochs=total_epochs,
                adam_warmup=adam_warmup,
                n_collocation=n_collocation,
                hidden=hidden,
                lr=lr,
                handover_strategy=handover_strategy,
                handover_max_adam_epochs=handover_max_adam_epochs,
                plateau_patience=plateau_patience,
                plateau_min_delta=plateau_min_delta,
                patience=patience,
            )
            runs.append(run)
        results.append(PipelineResult(pipeline=p, seeds=tuple(runs)))
    return tuple(results)


# =============================================================================
# Plotting helpers
# =============================================================================
def _pad_and_stack(seq: list[np.ndarray]) -> np.ndarray:
    """Stack possibly variable-length histories into (n, max_len), padding
    each shorter run with its last value. Lets early-stopped runs coexist
    with full-budget ones in the same mean / quantile band."""
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


# =============================================================================
# Plotting
# =============================================================================
def plot_comparison(
    results: tuple[PipelineResult, ...], out_path: str, k: float, adam_warmup: int
) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    for r in results:
        c = PIPELINE_COLOR[r.pipeline]
        H = _pad_and_stack([s.J_val_history for s in r.seeds])
        mean = np.nanmean(H, axis=0)
        lo = np.nanquantile(H, 0.25, axis=0)
        hi = np.nanquantile(H, 0.75, axis=0)
        epochs = np.arange(1, mean.size + 1)
        ax[0, 0].semilogy(epochs, mean, color=c, linewidth=1.5,
                          label=PIPELINE_LABEL[r.pipeline])
        ax[0, 0].fill_between(epochs, lo, hi, color=c, alpha=0.15)

        S = _pad_and_stack([s.sol_l2_history for s in r.seeds])
        smean = np.nanmean(S, axis=0)
        slo = np.nanquantile(S, 0.25, axis=0)
        shi = np.nanquantile(S, 0.75, axis=0)
        ax[0, 1].semilogy(epochs, smean, color=c, linewidth=1.5,
                          label=PIPELINE_LABEL[r.pipeline])
        ax[0, 1].fill_between(epochs, slo, shi, color=c, alpha=0.15)

    ax[0, 0].axvline(adam_warmup, color="k", linestyle=":", alpha=0.5, label="warm-up boundary")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel(r"$\mathcal{J}_{\mathrm{val}}$ (mean, IQR shaded)")
    ax[0, 0].set_title(f"Validation residual MSE (k={k:g})")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend(fontsize=9)

    ax[0, 1].axvline(adam_warmup, color="k", linestyle=":", alpha=0.5)
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel(r"$\|\widehat{u} - u^\star\|_{L^2}$ (mean, IQR shaded)")
    ax[0, 1].set_title("Solution L^2 error")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend(fontsize=9)

    # Bar plot of final residual.
    pipeline_names = [r.pipeline for r in results]
    means_J = [float(np.mean([s.final_J_val for s in r.seeds])) for r in results]
    stds_J = [float(np.std([s.final_J_val for s in r.seeds])) for r in results]
    means_sol = [float(np.mean([s.final_sol_l2 for s in r.seeds])) for r in results]
    stds_sol = [float(np.std([s.final_sol_l2 for s in r.seeds])) for r in results]

    x = np.arange(len(pipeline_names))
    cols = [PIPELINE_COLOR[p] for p in pipeline_names]

    ax[1, 0].bar(x, means_J, yerr=stds_J, color=cols, alpha=0.85, capsize=4)
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels([PIPELINE_LABEL[p] for p in pipeline_names], rotation=20, ha="right")
    ax[1, 0].set_ylabel(r"final $\mathcal{J}_{\mathrm{val}}$")
    ax[1, 0].set_title("Final residual (mean $\\pm$ std across seeds)")
    ax[1, 0].grid(True, alpha=0.3, axis="y")

    ax[1, 1].bar(x, means_sol, yerr=stds_sol, color=cols, alpha=0.85, capsize=4)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels([PIPELINE_LABEL[p] for p in pipeline_names], rotation=20, ha="right")
    ax[1, 1].set_ylabel(r"final $\|u-u^*\|_{L^2}$")
    ax[1, 1].set_title("Final solution error (mean $\\pm$ std across seeds)")
    ax[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to: {out_path}")


def write_summary(
    results: tuple[PipelineResult, ...],
    out_path: str,
    k: float,
    total_epochs: int,
    adam_warmup: int,
    seeds: tuple[int, ...],
) -> None:
    lines: list[str] = []
    lines.append(
        f"Optimiser comparison (1D BVP, k={k:g}, total_epochs={total_epochs}, "
        f"adam_warmup={adam_warmup}, seeds={list(seeds)})\n\n"
    )
    lines.append(
        f"{'pipeline':>20}  "
        f"{'mean J':>14}  {'std J':>14}    "
        f"{'mean solL2':>14}  {'std solL2':>14}    "
        f"{'mean relL2':>14}\n"
    )
    for r in results:
        mJ = float(np.mean([s.final_J_val for s in r.seeds]))
        sJ = float(np.std([s.final_J_val for s in r.seeds]))
        mS = float(np.mean([s.final_sol_l2 for s in r.seeds]))
        sS = float(np.std([s.final_sol_l2 for s in r.seeds]))
        mR = float(np.mean([s.final_sol_rel_l2 for s in r.seeds]))
        lines.append(
            f"{r.pipeline:>20}  "
            f"{mJ:>14.4e}  {sJ:>14.4e}    "
            f"{mS:>14.4e}  {sS:>14.4e}    "
            f"{mR:>14.4e}\n"
        )
    text = "".join(lines)
    with open(out_path, "w") as fh:
        fh.write(text)
    print("\n" + text)
    print(f"Saved summary to: {out_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adam / Adam->BFGS / Adam->SSBFGS / Adam->SSBroyden comparison on the 1D BVP."
    )
    p.add_argument(
        "--pipelines",
        type=str,
        nargs="+",
        default=list(PIPELINES),
        choices=list(PIPELINES),
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--wavenumber", type=float, default=DEFAULT_K)
    p.add_argument("--total-epochs", type=int, default=5000,
                   help="Matches the thesis 4.2 budget; SSBroyden corrupts H "
                        "on much longer horizons.")
    p.add_argument("--adam-warmup", type=int, default=2000,
                   help="Used only when --handover-strategy=fixed.")
    p.add_argument(
        "--handover-strategy",
        type=str,
        default="plateau",
        choices=list(HANDOVER_STRATEGIES),
    )
    p.add_argument("--handover-max-adam-epochs", type=int, default=10000)
    p.add_argument("--plateau-patience", type=int, default=200)
    p.add_argument("--plateau-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Early-stopping patience on the validation MA. Default 200 "
             "(stagnant runs terminate quickly). Set --patience >= "
             "--total-epochs to disable.",
    )
    p.add_argument("--n-collocation", type=int, default=400)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join("..", "results"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.results_dir,
        f"bvp1d_k{args.wavenumber:g}_optim_compare_{run_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    patience = args.patience
    print(
        f"\nOptimiser comparison on the 1D BVP "
        f"(k={args.wavenumber:g}, total_epochs={args.total_epochs}, "
        f"adam_warmup={args.adam_warmup}).\n"
        f"  pipelines:           {args.pipelines}\n"
        f"  seeds:               {args.seeds}\n"
        f"  handover_strategy:   {args.handover_strategy}\n"
        f"  early-stop patience: {patience}"
        f"{' (>= total_epochs => disabled)' if patience >= args.total_epochs else ''}\n"
    )

    results = run_comparison(
        pipelines=tuple(args.pipelines),
        seeds=tuple(args.seeds),
        k=args.wavenumber,
        total_epochs=args.total_epochs,
        adam_warmup=args.adam_warmup,
        n_collocation=args.n_collocation,
        hidden=tuple(args.hidden),
        lr=args.lr,
        handover_strategy=args.handover_strategy,
        handover_max_adam_epochs=args.handover_max_adam_epochs,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        patience=patience,
    )

    write_summary(
        results=results,
        out_path=os.path.join(out_dir, "summary_table.txt"),
        k=args.wavenumber,
        total_epochs=args.total_epochs,
        adam_warmup=args.adam_warmup,
        seeds=tuple(args.seeds),
    )
    plot_comparison(
        results=results,
        out_path=os.path.join(out_dir, "optimiser_comparison.png"),
        k=args.wavenumber,
        adam_warmup=args.adam_warmup,
    )

    np.savez(
        os.path.join(out_dir, "raw_histories.npz"),
        pipelines=np.asarray(args.pipelines),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        **{
            f"J_val_{r.pipeline}_seed{s.seed}": s.J_val_history
            for r in results for s in r.seeds
        },
        **{
            f"sol_l2_{r.pipeline}_seed{s.seed}": s.sol_l2_history
            for r in results for s in r.seeds
        },
    )
    print(f"All comparison artefacts written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
