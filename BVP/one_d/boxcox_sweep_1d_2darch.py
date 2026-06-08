"""
Box-Cox sweep on the 1D BVP using the *same protocol as the 2D sweeps*.

This is the 1D counterpart of ``boxcox_sweep_2d_helmholtz.py`` /
``boxcox_sweep_2d_cfgs.py``. All BVP benchmarks share the same standard
network -- 3 hidden layers x 32 units, Tanh -- so this script differs from
``boxcox_sweep_1d_finegrained.py`` not in architecture but in *protocol*: it
runs the from-the-start (non-delayed) Box-Cox engagement on the larger 2D
budget (10 000 collocation points, 10 000 epochs, 3 seeds), so the 1D and 2D
Box-Cox results can be compared on equal footing:

    * network          : 3 hidden layers x 32 units, Tanh   (BVP standard)
    * collocation      : 10 000 points
    * budget           : 10 000 epochs (2000 fixed Adam -> SSBroyden)
    * optimiser        : Adam (lr 1e-3) -> SSBroyden
    * handover         : fixed at 2000 epochs
    * QN early stop     : patience 500 (counted only after handover)
    * seeds            : 42, 43, 44 (mean +/- std)
    * lambda grid      : 0.0 .. 1.0 in steps of 0.1

The Adam warm-up runs in identity; the Box-Cox transform is applied throughout
exactly as the base ``PINN_BVP_SSBroyden`` does (the transform multiplies the
gradient/Hessian seen by every optimiser step). The known regime caveat still
holds: at k = 4 the 1D residual J plateaus around 10^4 after Adam, so
g''_lambda(J) is numerically negligible and Box-Cox cannot amplify curvature
in this regime (see ``BOXCOX_INVESTIGATION.md`` and ``boxcox_diagnostic.py``).
Running it under the 2D protocol makes that statement directly comparable to
the 2D figure rather than confounded by a different network/budget.

Run with default settings to reproduce the 2D protocol on the 1D BVP:

    python boxcox_sweep_1d_2darch.py

For a quicker smoke test:

    python boxcox_sweep_1d_2darch.py --epochs 5000 --adam-epochs 2000 \
        --n-collocation 2000 --seeds 42 43

Outputs go to `../results/bvp1d_k<k>_boxcox_2darch_<variant>_<timestamp>/`.
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
)
from pinn_adaptive_handover import (  # noqa: E402
    HANDOVER_STRATEGIES,
    PINN_BVP_AdaptiveHandover,
)


@dataclass(frozen=True)
class SeedResult:
    seed: int
    final_J_val: float
    final_sol_l2: float
    final_sol_rel_l2: float
    J_val: np.ndarray
    sol_l2: np.ndarray


# A seed counts as successful iff its final relative L^2 error sits below this
# threshold. At k = 4 the failure mode is binary: a seed either escapes the
# Adam plateau (relL2 ~ 1e-5) or stays on it (relL2 ~ 1, the trivial zero
# predictor, since ||u_exact||_{L^2} = sqrt(1/2) for u = sin(4 pi x) on [0,1]).
# 1.0 puts the cut-off just below that stuck value, so the conditional
# aggregates and trajectories report only the seeds that actually trained.
# This is what keeps a single escaping seed at, e.g., lambda = 0.7 from
# producing a misleading "partial improvement" in the unconditional mean.
SUCCESS_REL_L2_DEFAULT: float = 1.0


def is_successful(s: SeedResult, rel_l2_threshold: float) -> bool:
    """True iff the final iterate beats the configured rel. L^2 threshold."""
    return bool(np.isfinite(s.final_sol_rel_l2)) and (
        s.final_sol_rel_l2 < rel_l2_threshold
    )


@dataclass(frozen=True)
class LambdaResult:
    lambda_: float
    seeds: tuple[SeedResult, ...]

    def successful(self, rel_l2_threshold: float) -> tuple[SeedResult, ...]:
        return tuple(s for s in self.seeds if is_successful(s, rel_l2_threshold))

    def cond_stats(
        self, attr: str, rel_l2_threshold: float
    ) -> tuple[int, float, float]:
        """Return (n_success, mean, std) over the successful seeds.

        If no seed succeeded, returns (0, NaN, NaN). NaN-not-zero keeps
        matplotlib's log axes from auto-extending to absurd decades when a
        lambda has no successful run.
        """
        succ = self.successful(rel_l2_threshold)
        if not succ:
            return 0, float("nan"), float("nan")
        vals = [getattr(s, attr) for s in succ]
        return len(succ), float(np.mean(vals)), float(np.std(vals))


def run_one(
    lam: float,
    seed: int,
    k: float,
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    resample_every: int,
    train_split: float,
    hidden: tuple[int, ...],
    lr: float,
    qn_variant: str,
    handover_strategy: str,
    handover_max_adam_epochs: int,
    plateau_patience: int,
    plateau_min_delta: float,
    diag_grid_n: int,
    patience: int,
) -> SeedResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = NeuralNetwork(hidden_layers=tuple(hidden), activation=nn.Tanh())
    pinn = PINN_BVP_AdaptiveHandover(
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
        train_split=train_split,
        resample_every=resample_every,
        adam_epochs=adam_epochs,
        verbose_freq=max(1, n_epochs // 5),
        diag_grid_n=diag_grid_n,
        min_delta=1e-12,
        moving_avg_window=20,
        handover_strategy=handover_strategy,
        handover_max_adam_epochs=handover_max_adam_epochs,
        plateau_patience=plateau_patience,
        plateau_min_delta=plateau_min_delta,
        # The 2D protocol's --patience is a QN-phase early-stop, counted only
        # after handover. In the 1D adaptive-handover solver that is the
        # urban-style relative-MA criterion gated on handover_done.
        early_stop=True,
        es_patience=patience,
        es_window=20,
        es_min_delta=1e-4,
        es_stop_loss=0.0,
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
    k: float,
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    resample_every: int,
    train_split: float,
    hidden: tuple[int, ...],
    lr: float,
    qn_variant: str,
    handover_strategy: str,
    handover_max_adam_epochs: int,
    plateau_patience: int,
    plateau_min_delta: float,
    diag_grid_n: int,
    patience: int,
) -> tuple[LambdaResult, ...]:
    out: list[LambdaResult] = []
    for lam in lambdas:
        seed_runs: list[SeedResult] = []
        for s in seeds:
            print(
                f"\n[1D(2D-arch) lambda={lam:g}, seed={s}]  "
                f"handover={handover_strategy}, qn_patience={patience}"
            )
            seed_runs.append(run_one(
                lam=lam, seed=s, k=k,
                n_epochs=n_epochs, adam_epochs=adam_epochs,
                n_collocation=n_collocation,
                resample_every=resample_every,
                train_split=train_split,
                hidden=hidden, lr=lr,
                qn_variant=qn_variant,
                handover_strategy=handover_strategy,
                handover_max_adam_epochs=handover_max_adam_epochs,
                plateau_patience=plateau_patience,
                plateau_min_delta=plateau_min_delta,
                diag_grid_n=diag_grid_n,
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
    k: float,
    hidden: tuple[int, ...],
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

    arch = "x".join(str(h) for h in hidden)
    ax[0, 0].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5, label="Adam$\\to$SSBroyden")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel(r"$\mathcal{J}_{\mathrm{val}}$ (mean over seeds)")
    ax[0, 0].set_title(f"1D BVP, k={k:g}, net {arch} (2D protocol)")
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


def _log_safe_yerr(means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Convert symmetric ``mean +- std`` into asymmetric whiskers that never
    reach <=0 on a log axis. Returns a 2xN array suitable for ``yerr=``.

    When std > mean the lower whisker would otherwise be non-positive;
    matplotlib silently clips it and the axis auto-limits blow up. Capping
    the lower whisker at 0.95*mean keeps it strictly positive. NaN cells
    (lambda with zero successful seeds) get yerr=0; the accompanying NaN
    ``means`` value makes matplotlib skip drawing.
    """
    m = np.asarray(means, dtype=float)
    s = np.asarray(stds, dtype=float)
    lo = np.where(np.isfinite(m), np.minimum(s, 0.95 * np.abs(m)), 0.0)
    hi = np.where(np.isfinite(m), s, 0.0)
    return np.vstack([lo, hi])


def plot_sweep_conditioned(
    results: tuple[LambdaResult, ...],
    out_path: str,
    k: float,
    hidden: tuple[int, ...],
    adam_epochs: int,
    rel_l2_threshold: float = SUCCESS_REL_L2_DEFAULT,
) -> None:
    """Success-conditioned counterpart of ``plot_sweep``.

    Failed seeds (final rel.L^2 >= ``rel_l2_threshold``) are excluded from
    every aggregate; lambdas with zero successful seeds get a "k/N" annotation
    in the bar panels and no trajectory in the line panels. This is the honest
    representation of the binary escape-the-plateau outcome at k = 4, where the
    unconditional mean averages two distinct populations.
    """
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    cmap = plt.get_cmap("viridis")
    n = len(results)
    success_counts: list[tuple[int, int]] = []

    for i, lr in enumerate(results):
        c = cmap(i / max(n - 1, 1))
        succ = lr.successful(rel_l2_threshold)
        n_succ, n_total = len(succ), len(lr.seeds)
        success_counts.append((n_succ, n_total))
        succ_tag = f" [{n_succ}/{n_total}]"
        if n_succ == 0:
            ax[0, 0].plot([], [], color=c, linewidth=1.4,
                          label=rf"$\lambda={lr.lambda_:g}${succ_tag}")
            ax[0, 1].plot([], [], color=c, linewidth=1.4,
                          label=rf"$\lambda={lr.lambda_:g}${succ_tag}")
            continue
        H = _pad_and_stack([s.J_val for s in succ])
        ax[0, 0].semilogy(np.nanmean(H, axis=0), color=c, linewidth=1.4,
                          label=rf"$\lambda={lr.lambda_:g}${succ_tag}")
        S = _pad_and_stack([s.sol_l2 for s in succ])
        ax[0, 1].semilogy(np.nanmean(S, axis=0), color=c, linewidth=1.4,
                          label=rf"$\lambda={lr.lambda_:g}${succ_tag}")

    arch = "x".join(str(h) for h in hidden)
    ax[0, 0].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5,
                     label="Adam$\\to$SSBroyden")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel(r"$\mathcal{J}_{\mathrm{val}}$ (mean over successful seeds)")
    ax[0, 0].set_title(
        f"1D BVP, k={k:g}, net {arch} (2D protocol); successful seeds only "
        f"(rel.$L^2<{rel_l2_threshold:g}$)"
    )
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend(fontsize=8, ncol=2)

    ax[0, 1].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5)
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel(r"$\|\widehat{u} - u^\star\|_{L^2}$")
    ax[0, 1].set_title("Solution error trajectory (successful seeds)")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend(fontsize=8, ncol=2)

    lams = np.asarray([r.lambda_ for r in results])
    cond_J = [r.cond_stats("final_J_val", rel_l2_threshold) for r in results]
    means_J = np.asarray([m for _, m, _ in cond_J], dtype=float)
    stds_J = np.asarray([s for _, _, s in cond_J], dtype=float)
    cond_S = [r.cond_stats("final_sol_l2", rel_l2_threshold) for r in results]
    means_S = np.asarray([m for _, m, _ in cond_S], dtype=float)
    stds_S = np.asarray([s for _, _, s in cond_S], dtype=float)

    def _set_log_ylim(axis: plt.Axes, means: np.ndarray, stds: np.ndarray) -> None:
        m = np.asarray(means, dtype=float)
        s = np.asarray(stds, dtype=float)
        mask = np.isfinite(m)
        if not mask.any():
            return
        lo_cand = np.maximum(m[mask] - np.minimum(s[mask], 0.95 * m[mask]),
                             m[mask] * 0.05)
        hi_cand = m[mask] + s[mask]
        lo, hi = float(np.min(lo_cand)), float(np.max(hi_cand))
        if lo <= 0 or not np.isfinite(lo) or not np.isfinite(hi):
            return
        axis.set_ylim(lo / np.sqrt(10.0), hi * np.sqrt(10.0))

    def _annotate_empty(axis: plt.Axes, means: np.ndarray) -> None:
        y_lo, _ = axis.get_ylim()
        for lam_val, m, (n_succ, n_total) in zip(lams, means, success_counts):
            if np.isfinite(m):
                continue
            axis.text(lam_val, y_lo * 3, f"{n_succ}/{n_total}",
                      ha="center", va="bottom", fontsize=8, color="dimgray")

    ax[1, 0].errorbar(lams, means_J, yerr=_log_safe_yerr(means_J, stds_J),
                      fmt="o-", color="C3", capsize=3,
                      label=r"final $\mathcal{J}_{\mathrm{val}}$ mean $\pm$ std (successful seeds)")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel(r"Box-Cox $\lambda$")
    ax[1, 0].set_ylabel(r"final $\mathcal{J}_{\mathrm{val}}$")
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].legend(fontsize=9)
    ax[1, 0].set_title("Final residual vs lambda (successful seeds)")
    _set_log_ylim(ax[1, 0], means_J, stds_J)
    _annotate_empty(ax[1, 0], means_J)

    ax[1, 1].errorbar(lams, means_S, yerr=_log_safe_yerr(means_S, stds_S),
                      fmt="s-", color="C0", capsize=3,
                      label=r"final $\|u-u^*\|_{L^2}$ mean $\pm$ std (successful seeds)")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel(r"Box-Cox $\lambda$")
    ax[1, 1].set_ylabel(r"final solution $L^2$ error")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].legend(fontsize=9)
    ax[1, 1].set_title("Solution error vs lambda (successful seeds)")
    _set_log_ylim(ax[1, 1], means_S, stds_S)
    _annotate_empty(ax[1, 1], means_S)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved success-conditioned figure to: {out_path}")


def write_summary(
    results: tuple[LambdaResult, ...],
    out_path: str,
    k: float,
    hidden: tuple[int, ...],
    qn_variant: str,
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    seeds: tuple[int, ...],
    rel_l2_threshold: float = SUCCESS_REL_L2_DEFAULT,
) -> None:
    arch = "x".join(str(h) for h in hidden)
    lines: list[str] = []
    lines.append(
        f"1D BVP Box-Cox sweep (2D protocol), k={k:g}, net={arch}, "
        f"n_collocation={n_collocation}, qn={qn_variant}, "
        f"epochs={n_epochs} (adam={adam_epochs}), seeds={list(seeds)}\n"
    )
    lines.append(
        f"Success criterion: final relative L^2 error < {rel_l2_threshold:g}.\n\n"
    )

    # --- Unconditional aggregates (mean over ALL seeds; 2D-comparable) -------
    lines.append("== Unconditional aggregates (all seeds) ==\n")
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

    # --- Conditional aggregates (successful seeds only) ----------------------
    lines.append("\n== Aggregates over successful seeds only ==\n")
    lines.append(
        f"{'lambda':>8}  {'n_succ':>8}  "
        f"{'mean J':>14}  {'std J':>14}    "
        f"{'mean solL2':>14}  {'std solL2':>14}\n"
    )
    for r in results:
        nJ, mJ, sJ = r.cond_stats("final_J_val", rel_l2_threshold)
        _, mS, sS = r.cond_stats("final_sol_l2", rel_l2_threshold)
        n_total = len(r.seeds)
        mJ_s = "n/a" if not np.isfinite(mJ) else f"{mJ:.4e}"
        sJ_s = "n/a" if not np.isfinite(sJ) else f"{sJ:.4e}"
        mS_s = "n/a" if not np.isfinite(mS) else f"{mS:.4e}"
        sS_s = "n/a" if not np.isfinite(sS) else f"{sS:.4e}"
        lines.append(
            f"{r.lambda_:>8.4g}  {nJ:>3d}/{n_total:<3d}  "
            f"{mJ_s:>14}  {sJ_s:>14}    {mS_s:>14}  {sS_s:>14}\n"
        )

    # --- Per-seed final metrics (the escape table) ---------------------------
    lines.append("\n== Per-seed final metrics (all runs, including failures) ==\n")
    lines.append(
        f"{'lambda':>8}  {'seed':>6}  "
        f"{'final J':>14}  {'final solL2':>14}  {'final relL2':>14}  status\n"
    )
    for r in results:
        for s in r.seeds:
            status = "ok" if is_successful(s, rel_l2_threshold) else "FAIL"
            lines.append(
                f"{r.lambda_:>8.4g}  {s.seed:>6d}  "
                f"{s.final_J_val:>14.4e}  {s.final_sol_l2:>14.4e}  "
                f"{s.final_sol_rel_l2:>14.4e}  {status}\n"
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
        description="Box-Cox sweep on the 1D BVP using the 2D Helmholtz "
                    "experimental protocol (2x20 net, 10k collocation, "
                    "2000 Adam -> SSBroyden)."
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
    p.add_argument("--wavenumber", type=float, default=DEFAULT_K)
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
    p.add_argument("--n-collocation", type=int, default=10000,
                   help="Matches the 2D Helmholtz protocol (10 000 points).")
    p.add_argument("--resample-every", type=int, default=500)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32, 32],
                   help="Hidden-layer widths. Default 3x32 is the standard BVP "
                        "architecture used by every 1D and 2D benchmark, so all "
                        "Box-Cox sweeps share one network.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--diag-grid-n", type=int, default=400,
                   help="Grid resolution for the 1D solution-error diagnostics.")
    p.add_argument(
        "--handover-strategy",
        type=str,
        default="fixed",
        choices=list(HANDOVER_STRATEGIES),
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
    p.add_argument(
        "--success-rel-l2-threshold",
        type=float,
        default=SUCCESS_REL_L2_DEFAULT,
        help=(
            "Final relative L^2 error below which a seed counts as successful "
            "(having escaped the Adam plateau). Default 1.0 (= the trivial "
            "zero predictor). Drives the success-conditioned figure and the "
            "conditional aggregates in the summary table."
        ),
    )
    p.add_argument(
        "--no-conditioned-plot",
        dest="conditioned_plot",
        action="store_false",
        help="Skip the success-conditioned figure (emit only the "
             "unconditional 2D-comparable one).",
    )
    p.set_defaults(conditioned_plot=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.lambdas_linspace is not None:
        start, stop, n = args.lambdas_linspace
        lambdas = tuple(float(v) for v in np.linspace(start, stop, int(n)))
    else:
        lambdas = tuple(args.lambdas)

    seeds = tuple(args.seeds)
    hidden = tuple(args.hidden)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.results_dir,
        f"bvp1d_k{args.wavenumber:g}_boxcox_2darch_{args.qn_variant}_{run_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    arch = "x".join(str(h) for h in hidden)
    print(
        f"\n1D BVP Box-Cox sweep (2D protocol) "
        f"(k={args.wavenumber:g}, net={arch}, "
        f"n_collocation={args.n_collocation}, qn={args.qn_variant}, "
        f"epochs={args.epochs}, adam={args.adam_epochs})\n"
        f"  lambdas: {lambdas}\n"
        f"  seeds:   {seeds}\n"
    )

    results = run_sweep(
        lambdas=lambdas, seeds=seeds,
        k=args.wavenumber,
        n_epochs=args.epochs, adam_epochs=args.adam_epochs,
        n_collocation=args.n_collocation,
        resample_every=args.resample_every,
        train_split=args.train_split,
        hidden=hidden, lr=args.lr,
        qn_variant=args.qn_variant,
        handover_strategy=args.handover_strategy,
        handover_max_adam_epochs=args.handover_max_adam_epochs,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        diag_grid_n=args.diag_grid_n,
        patience=args.patience,
    )

    write_summary(
        results=results,
        out_path=os.path.join(out_dir, "summary_table.txt"),
        k=args.wavenumber,
        hidden=hidden,
        qn_variant=args.qn_variant,
        n_epochs=args.epochs, adam_epochs=args.adam_epochs,
        n_collocation=args.n_collocation,
        seeds=seeds,
        rel_l2_threshold=args.success_rel_l2_threshold,
    )
    plot_sweep(
        results=results,
        out_path=os.path.join(out_dir, "boxcox_sweep_1d_2darch.png"),
        k=args.wavenumber,
        hidden=hidden,
        adam_epochs=args.adam_epochs,
    )
    if args.conditioned_plot:
        plot_sweep_conditioned(
            results=results,
            out_path=os.path.join(out_dir, "boxcox_sweep_1d_2darch_conditioned.png"),
            k=args.wavenumber,
            hidden=hidden,
            adam_epochs=args.adam_epochs,
            rel_l2_threshold=args.success_rel_l2_threshold,
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
