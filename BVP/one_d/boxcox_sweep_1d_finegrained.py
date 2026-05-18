"""
Fine-grained Box-Cox lambda sweep for the 1D PINN.

Three improvements over `boxcox_sweep_1d.py`:

    (1) Finer lambda grid (default 11 points: lambda in {-0.5, -0.25, 0, 0.125,
        0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0}). The user can also pass an
        explicit `--lambdas-linspace start stop n` triple.

    (2) Multi-seed averaging. By default each lambda is repeated 5 times with
        independent seeds, and the table reports mean +/- std rather than a
        single seed.

    (3) Optional *delayed engagement*: with `--engage-when J<thresh`, the
        transformation is left at identity until the validation J first crosses
        a configurable threshold. This places SSBroyden inside the small-loss
        regime where g''_lambda(J) is genuinely large — the regime that
        Urban et al. (2025) target.

The Adam phase always runs in identity. The schedule for the SSBroyden phase
is parametrised by:

    transform = identity   if J_val > engage_threshold else boxcox(lambda)

This isolates the curvature-amplification claim of section 4.2.3 of the thesis
from the Adam warm-up regime, in which (k pi)^4 forces J ~ 10^4 and g''_lambda
is numerically negligible. See `boxcox_diagnostic.py` for the underlying
analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

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


# =============================================================================
# Subclass with optional delayed Box-Cox engagement
# =============================================================================
class PINN_BVP_SSBroyden_Schedulable(PINN_BVP_SSBroyden):
    """Same as the base PINN, but the transformation switches from `identity`
    to `boxcox(lambda)` after the *training* J first dips below a threshold.

    Pass ``engage_threshold=None`` to recover the unconditional behaviour.
    """

    def __init__(
        self,
        *args,
        engage_threshold: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._engage_threshold = engage_threshold
        self._engaged = engage_threshold is None
        self._target_lambda = float(self.loss_lambda)
        if engage_threshold is not None:
            # Start with identity; the actual lambda is restored once we engage.
            self.loss_transform = "identity"
            self.engagement_epoch: Optional[int] = None
        else:
            self.engagement_epoch = 1

    def _maybe_engage(self, current_J: float, epoch: int) -> None:
        if self._engaged:
            return
        if self._engage_threshold is None:
            return
        if current_J < self._engage_threshold:
            self.loss_transform = "boxcox"
            self.loss_lambda = self._target_lambda
            self._engaged = True
            self.engagement_epoch = epoch
            print(
                f"  [engage] epoch {epoch}: J={current_J:.3e} < threshold "
                f"{self._engage_threshold:.3e}; switching transform to "
                f"boxcox(lambda={self._target_lambda:g})"
            )

    def compute_loss(self, x_interior, create_graph_second):
        # Hook the engagement check before computing the (possibly transformed)
        # loss for the next training step. We use the most recent J_train value
        # as the trigger so that the change happens between iterations.
        if not self._engaged and self.J_train:
            self._maybe_engage(float(self.J_train[-1]), len(self.J_train))
        return super().compute_loss(x_interior, create_graph_second)


# =============================================================================
# Data classes
# =============================================================================
@dataclass(frozen=True)
class SeedResult:
    seed: int
    final_J_val: float
    final_pde_l2: float
    final_sol_l2: float
    final_sol_rel_l2: float
    engagement_epoch: Optional[int]
    J_val_history: np.ndarray
    sol_l2_history: np.ndarray


# A seed counts as successful iff its final relative L^2 error sits below
# this threshold. 1.0 puts the cut-off at the trivial zero predictor; with
# u_exact = sin(4 pi x) on [0,1] that's at ||u_exact||_{L^2} = sqrt(1/2),
# so a seed that fails to escape the Adam plateau saturates near rel.L^2 ~= 1.
# All conditional statistics in plot_sweep / write_summary are taken over the
# success subset to avoid having the unconditional mean dominated by 1-2 seeds
# that never enter the small-loss regime (which silently wipes out the
# lambda dependence on networks that are near the edge of trainability).
SUCCESS_REL_L2_DEFAULT: float = 1.0


def is_successful(s: "SeedResult", rel_l2_threshold: float) -> bool:
    """True iff the final iterate beats the configured rel. L^2 threshold."""
    return bool(np.isfinite(s.final_sol_rel_l2)) and (
        s.final_sol_rel_l2 < rel_l2_threshold
    )


@dataclass(frozen=True)
class LambdaResult:
    lambda_: float
    seeds: tuple[SeedResult, ...]

    @property
    def final_J_val_mean(self) -> float:
        return float(np.mean([s.final_J_val for s in self.seeds]))

    @property
    def final_J_val_std(self) -> float:
        return float(np.std([s.final_J_val for s in self.seeds]))

    @property
    def final_sol_l2_mean(self) -> float:
        return float(np.mean([s.final_sol_l2 for s in self.seeds]))

    @property
    def final_sol_l2_std(self) -> float:
        return float(np.std([s.final_sol_l2 for s in self.seeds]))

    def successful(self, rel_l2_threshold: float) -> tuple[SeedResult, ...]:
        return tuple(s for s in self.seeds if is_successful(s, rel_l2_threshold))

    def cond_stats(
        self, attr: str, rel_l2_threshold: float
    ) -> tuple[int, float, float]:
        """Return (n_success, mean, std) over the successful seeds.

        If no seed succeeded, returns (0, NaN, NaN). NaN-not-zero is what
        keeps matplotlib's log axes from auto-extending to absurd negative
        decades when a lambda has no successful run.
        """
        succ = self.successful(rel_l2_threshold)
        if not succ:
            return 0, float("nan"), float("nan")
        vals = [getattr(s, attr) for s in succ]
        return len(succ), float(np.mean(vals)), float(np.std(vals))


# =============================================================================
# Single run
# =============================================================================
def run_single(
    lam: float,
    seed: int,
    k: float,
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    resample_every: int,
    hidden: tuple[int, ...],
    lr: float,
    qn_variant: str,
    engage_threshold: Optional[float],
    handover_strategy: str,
    handover_max_adam_epochs: int,
    plateau_patience: int,
    plateau_min_delta: float,
    loss_threshold_handover: float,
    gradnorm_threshold: float,
    patience: int,
) -> SeedResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = NeuralNetwork(hidden_layers=tuple(hidden), activation=nn.Tanh())

    if engage_threshold is not None:
        # Delayed Box-Cox engagement: keep the legacy fixed-handover schedule.
        pinn = PINN_BVP_SSBroyden_Schedulable(
            model=model,
            k=k,
            lr=lr,
            loss_transform="boxcox",
            loss_lambda=lam,
            qn_variant=qn_variant,
            engage_threshold=engage_threshold,
        )
        pinn.train(
            n_epochs=n_epochs,
            n_collocation=n_collocation,
            train_split=0.8,
            resample_every=resample_every,
            adam_epochs=adam_epochs,
            verbose_freq=max(1, n_epochs // 5),
            diag_grid_n=400,
            patience=patience,
            min_delta=1e-12,
            moving_avg_window=20,
        )
        engagement_epoch = pinn.engagement_epoch
    else:
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
            train_split=0.8,
            resample_every=resample_every,
            adam_epochs=adam_epochs,
            verbose_freq=max(1, n_epochs // 5),
            diag_grid_n=400,
            patience=patience,
            min_delta=1e-12,
            moving_avg_window=20,
            handover_strategy=handover_strategy,
            handover_max_adam_epochs=handover_max_adam_epochs,
            plateau_patience=plateau_patience,
            plateau_min_delta=plateau_min_delta,
            loss_threshold=loss_threshold_handover,
            gradnorm_threshold=gradnorm_threshold,
        )
        engagement_epoch = getattr(pinn, "handover_epoch", None)

    return SeedResult(
        seed=seed,
        final_J_val=float(pinn.J_val[-1]),
        final_pde_l2=float(pinn.pde_l2[-1]),
        final_sol_l2=float(pinn.sol_l2[-1]),
        final_sol_rel_l2=float(pinn.sol_rel_l2[-1]),
        engagement_epoch=engagement_epoch,
        J_val_history=np.asarray(pinn.J_val, dtype=np.float64),
        sol_l2_history=np.asarray(pinn.sol_l2, dtype=np.float64),
    )


# =============================================================================
# Sweep driver
# =============================================================================
def run_sweep(
    lambdas: tuple[float, ...],
    seeds: tuple[int, ...],
    k: float,
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    resample_every: int,
    hidden: tuple[int, ...],
    lr: float,
    qn_variant: str,
    engage_threshold: Optional[float],
    handover_strategy: str,
    handover_max_adam_epochs: int,
    plateau_patience: int,
    plateau_min_delta: float,
    loss_threshold_handover: float,
    gradnorm_threshold: float,
    patience: int,
) -> tuple[LambdaResult, ...]:
    out: list[LambdaResult] = []
    for lam in lambdas:
        seed_results: list[SeedResult] = []
        for seed in seeds:
            print(
                f"\n[lambda={lam:g}, seed={seed}] "
                f"engage_threshold={engage_threshold}, "
                f"handover={handover_strategy}, patience={patience}"
            )
            res = run_single(
                lam=lam,
                seed=seed,
                k=k,
                n_epochs=n_epochs,
                adam_epochs=adam_epochs,
                n_collocation=n_collocation,
                resample_every=resample_every,
                hidden=hidden,
                lr=lr,
                qn_variant=qn_variant,
                engage_threshold=engage_threshold,
                handover_strategy=handover_strategy,
                handover_max_adam_epochs=handover_max_adam_epochs,
                plateau_patience=plateau_patience,
                plateau_min_delta=plateau_min_delta,
                loss_threshold_handover=loss_threshold_handover,
                gradnorm_threshold=gradnorm_threshold,
                patience=patience,
            )
            seed_results.append(res)
        out.append(LambdaResult(lambda_=lam, seeds=tuple(seed_results)))
    return tuple(out)


# =============================================================================
# Plotting
# =============================================================================
def _log_safe_yerr(
    means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    """Convert symmetric ``mean +- std`` into asymmetric whiskers that never
    reach <=0 on a log axis. Returns a 2xN array suitable for ``yerr=``.

    When std > mean, the lower whisker would otherwise be non-positive;
    matplotlib silently clips it to a tiny value and the axis auto-limits
    blow up. Capping the lower whisker at 0.95*mean keeps it strictly
    positive while still showing the seedwise spread when std < mean.
    NaN cells (lambda with zero successful seeds) get yerr=0; the
    accompanying NaN ``means`` value makes matplotlib skip drawing.
    """
    m = np.asarray(means, dtype=float)
    s = np.asarray(stds, dtype=float)
    lo = np.where(np.isfinite(m), np.minimum(s, 0.95 * np.abs(m)), 0.0)
    hi = np.where(np.isfinite(m), s, 0.0)
    return np.vstack([lo, hi])


def plot_sweep(
    results: tuple[LambdaResult, ...],
    out_path: str,
    k: float,
    adam_epochs: int,
    engage_threshold: Optional[float],
    rel_l2_threshold: float = SUCCESS_REL_L2_DEFAULT,
) -> None:
    """Compose the four-panel sweep figure using only successful seeds.

    Failed seeds (final rel.L^2 >= ``rel_l2_threshold``) are excluded from
    every aggregate statistic; lambdas with zero successful seeds get a
    "0/N succeeded" annotation in the bar panels and no trajectory in the
    line panels.
    """
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    cmap = plt.get_cmap("viridis")
    n = len(results)
    success_counts: list[tuple[int, int]] = []

    # (0,0) and (0,1) -- mean trajectories per lambda, conditional on success.
    for i, lr in enumerate(results):
        c = cmap(i / max(n - 1, 1))
        succ = lr.successful(rel_l2_threshold)
        n_succ, n_total = len(succ), len(lr.seeds)
        success_counts.append((n_succ, n_total))
        succ_tag = f" [{n_succ}/{n_total}]"
        if n_succ == 0:
            # Reserve the legend slot but draw nothing.
            ax[0, 0].plot([], [], color=c, linewidth=1.3,
                          label=rf"$\lambda={lr.lambda_:g}${succ_tag}")
            ax[0, 1].plot([], [], color=c, linewidth=1.3,
                          label=rf"$\lambda={lr.lambda_:g}${succ_tag}")
            continue
        H = np.stack([s.J_val_history for s in succ], axis=0)
        ax[0, 0].semilogy(np.mean(H, axis=0), color=c, linewidth=1.3,
                          label=rf"$\lambda={lr.lambda_:g}${succ_tag}")
        S = np.stack([s.sol_l2_history for s in succ], axis=0)
        ax[0, 1].semilogy(np.mean(S, axis=0), color=c, linewidth=1.3,
                          label=rf"$\lambda={lr.lambda_:g}${succ_tag}")

    ax[0, 0].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5,
                     label="Adam$\\to$SSBroyden")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel(r"$\mathcal{J}_{\mathrm{val}}$ (mean over successful seeds)")
    ax[0, 0].set_title(
        f"Validation residual MSE; successful seeds only "
        f"(rel.$L^2<{rel_l2_threshold:g}$)"
    )
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend(fontsize=8, ncol=2)

    ax[0, 1].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5)
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel(r"$\|\widehat{u} - u^\star\|_{L^2}$")
    ax[0, 1].set_title(
        f"Solution $L^2$ error; successful seeds only "
        f"(rel.$L^2<{rel_l2_threshold:g}$)"
    )
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend(fontsize=8, ncol=2)

    # (1,0) and (1,1) -- final values per lambda, conditional on success.
    lams = np.asarray([r.lambda_ for r in results])
    cond = [r.cond_stats("final_J_val", rel_l2_threshold) for r in results]
    means_J = np.asarray([m for _, m, _ in cond], dtype=float)
    stds_J = np.asarray([s for _, _, s in cond], dtype=float)
    cond_s = [r.cond_stats("final_sol_l2", rel_l2_threshold) for r in results]
    means_sol = np.asarray([m for _, m, _ in cond_s], dtype=float)
    stds_sol = np.asarray([s for _, _, s in cond_s], dtype=float)

    def _set_log_ylim(
        axis: plt.Axes, means: np.ndarray, stds: np.ndarray
    ) -> None:
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
            axis.text(lam_val, y_lo * 3,
                      f"{n_succ}/{n_total}",
                      ha="center", va="bottom", fontsize=8, color="dimgray")

    ax[1, 0].errorbar(lams, means_J, yerr=_log_safe_yerr(means_J, stds_J),
                      fmt="o-", color="C3", capsize=3,
                      label=r"final $\mathcal{J}_{\mathrm{val}}$ mean $\pm$ std (successful seeds)")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel(r"Box-Cox $\lambda$")
    ax[1, 0].set_ylabel(r"final $\mathcal{J}_{\mathrm{val}}$")
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].legend(fontsize=9)
    title = f"Final residual (k={k:g}"
    if engage_threshold is not None:
        title += f", delayed engagement at J<{engage_threshold:g}"
    title += ")"
    ax[1, 0].set_title(title)
    _set_log_ylim(ax[1, 0], means_J, stds_J)
    _annotate_empty(ax[1, 0], means_J)

    ax[1, 1].errorbar(lams, means_sol, yerr=_log_safe_yerr(means_sol, stds_sol),
                      fmt="s-", color="C0", capsize=3,
                      label=r"final $\|u-u^*\|_{L^2}$ mean $\pm$ std (successful seeds)")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel(r"Box-Cox $\lambda$")
    ax[1, 1].set_ylabel(r"final solution $L^2$ error")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].legend(fontsize=9)
    ax[1, 1].set_title("Solution error vs lambda (successful seeds)")
    _set_log_ylim(ax[1, 1], means_sol, stds_sol)
    _annotate_empty(ax[1, 1], means_sol)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved sweep figure to: {out_path}")
    plt.close(fig)


def _fmt_stat(values: list[float]) -> tuple[str, str]:
    """Return ``(mean, std)`` as strings, or ``("n/a", "n/a")`` when empty."""
    if not values:
        return "n/a", "n/a"
    return f"{float(np.mean(values)):.4e}", f"{float(np.std(values)):.4e}"


def write_summary(
    results: tuple[LambdaResult, ...],
    out_path: str,
    k: float,
    qn_variant: str,
    n_epochs: int,
    adam_epochs: int,
    engage_threshold: Optional[float],
    seeds: tuple[int, ...],
    rel_l2_threshold: float = SUCCESS_REL_L2_DEFAULT,
) -> None:
    """Write a two-section summary: success counts + conditional aggregates,
    then per-seed final metrics with ok/FAIL flags so failed runs stay
    auditable in the artefact."""
    lines: list[str] = []
    lines.append(
        f"Fine-grained Box-Cox sweep (1D BVP, k={k:g}, qn={qn_variant}, "
        f"epochs={n_epochs}, adam={adam_epochs}, seeds={list(seeds)})\n"
    )
    if engage_threshold is not None:
        lines.append(
            f"Delayed engagement: transformation engages once J < "
            f"{engage_threshold:g}.\n"
        )
    lines.append(
        f"Success criterion: final relative L^2 error < {rel_l2_threshold:g}.\n\n"
    )

    lines.append("== Aggregate statistics over successful seeds ==\n")
    lines.append(
        f"{'lambda':>8}  {'n_succ':>8}  "
        f"{'mean J':>14}  {'std J':>14}    "
        f"{'mean solL2':>14}  {'std solL2':>14}    "
        f"{'engaged':>16}\n"
    )
    for lr in results:
        succ = lr.successful(rel_l2_threshold)
        n_succ, n_total = len(succ), len(lr.seeds)
        mJ, sJ = _fmt_stat([s.final_J_val for s in succ])
        mS, sS = _fmt_stat([s.final_sol_l2 for s in succ])
        eng = [s.engagement_epoch for s in succ if s.engagement_epoch is not None]
        eng_str = f"epoch~{int(np.mean(eng))}" if eng else "never/from-start"
        lines.append(
            f"{lr.lambda_:>8.4g}  {n_succ:>3d}/{n_total:<3d}  "
            f"{mJ:>14}  {sJ:>14}    "
            f"{mS:>14}  {sS:>14}    "
            f"{eng_str:>16}\n"
        )

    lines.append("\n== Per-seed final metrics (all runs, including failures) ==\n")
    lines.append(
        f"{'lambda':>8}  {'seed':>6}  "
        f"{'final J':>14}  {'final solL2':>14}  {'final relL2':>14}  status\n"
    )
    for lr in results:
        for s in lr.seeds:
            status = "ok" if is_successful(s, rel_l2_threshold) else "FAIL"
            lines.append(
                f"{lr.lambda_:>8.4g}  {s.seed:>6d}  "
                f"{s.final_J_val:>14.4e}  {s.final_sol_l2:>14.4e}  "
                f"{s.final_sol_rel_l2:>14.4e}  {status}\n"
            )

    text = "".join(lines)
    with open(out_path, "w") as fh:
        fh.write(text)
    print("\n" + text)
    print(f"Saved summary to: {out_path}")


# =============================================================================
# CLI
# =============================================================================
DEFAULT_LAMBDAS: tuple[float, ...] = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-grained Box-Cox sweep with multi-seed averaging "
                    "and optional delayed engagement."
    )
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help=f"Box-Cox lambda grid. Default: {list(DEFAULT_LAMBDAS)}",
    )
    p.add_argument(
        "--lambdas-linspace",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "N"),
        default=None,
        help="Override --lambdas with np.linspace(start, stop, n).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Random seeds to average over. Default: {list(DEFAULT_SEEDS)}",
    )
    p.add_argument(
        "--engage-threshold",
        type=float,
        default=None,
        help="If set, leave transform=identity until validation J first dips "
             "below this threshold; only then switch to boxcox(lambda).",
    )
    p.add_argument("--wavenumber", type=float, default=DEFAULT_K)
    p.add_argument(
        "--qn-variant",
        type=str,
        default="ssbroyden",
        choices=["bfgs", "ssbfgs", "ssbroyden"],
    )
    p.add_argument("--epochs", type=int, default=5000,
                   help="Total epochs (matches the thesis 4.2 budget).")
    p.add_argument("--adam-epochs", type=int, default=2000,
                   help="Used only when --handover-strategy=fixed.")
    p.add_argument(
        "--handover-strategy",
        type=str,
        default="plateau",
        choices=list(HANDOVER_STRATEGIES),
        help="When Adam hands over to the QN optimiser. plateau (default) "
             "switches once val J stops improving by --plateau-min-delta over "
             "the last --plateau-patience epochs, capped at --handover-max-adam-epochs.",
    )
    p.add_argument("--handover-max-adam-epochs", type=int, default=10000)
    p.add_argument("--plateau-patience", type=int, default=200)
    p.add_argument("--plateau-min-delta", type=float, default=1e-4)
    p.add_argument("--loss-threshold-handover", type=float, default=1.0,
                   help="Used only when --handover-strategy=loss_threshold.")
    p.add_argument("--gradnorm-threshold", type=float, default=1e-3,
                   help="Used only when --handover-strategy=gradnorm.")
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stopping patience on the validation MA. Default: disabled "
             "(equal to --epochs) so all (lambda, seed) pairs run for the same "
             "fixed budget. Set to e.g. 500 to enable.",
    )
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
            "4.2 of the thesis, so the optimiser-comparison and Box-Cox "
            "sweep share the same network."
        ),
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join("..", "results"),
    )
    p.add_argument(
        "--success-rel-l2-threshold",
        type=float,
        default=SUCCESS_REL_L2_DEFAULT,
        help=(
            "Final relative L^2 error below which a seed is considered "
            "successful. Default 1.0 (= the trivial zero predictor). "
            "Aggregates in the figure and summary table are taken over "
            "the successful subset; tighten this if you only want clearly "
            "trained runs."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.lambdas_linspace is not None:
        start, stop, n = args.lambdas_linspace
        lambdas = tuple(float(v) for v in np.linspace(start, stop, int(n)))
    else:
        lambdas = tuple(args.lambdas)

    seeds = tuple(args.seeds)
    patience = args.epochs if args.patience is None else args.patience

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    suffix = "_delayed" if args.engage_threshold is not None else ""
    sweep_dir = os.path.join(
        args.results_dir,
        f"bvp1d_k{args.wavenumber:g}_boxcox_finesweep{suffix}_{args.qn_variant}_{run_tag}",
    )
    os.makedirs(sweep_dir, exist_ok=True)

    print(
        f"\nFine-grained Box-Cox sweep on the 1D BVP "
        f"(k={args.wavenumber:g}, qn={args.qn_variant}, "
        f"epochs={args.epochs}, adam={args.adam_epochs})\n"
        f"  lambdas:           {lambdas}\n"
        f"  seeds:             {seeds}\n"
        f"  engage_threshold:  {args.engage_threshold}\n"
        f"  handover_strategy: {args.handover_strategy}\n"
        f"  early-stop patience: {patience} (== epochs => disabled)\n"
    )

    results = run_sweep(
        lambdas=lambdas,
        seeds=seeds,
        k=args.wavenumber,
        n_epochs=args.epochs,
        adam_epochs=args.adam_epochs,
        n_collocation=args.n_collocation,
        resample_every=args.resample_every,
        hidden=tuple(args.hidden),
        lr=args.lr,
        qn_variant=args.qn_variant,
        engage_threshold=args.engage_threshold,
        handover_strategy=args.handover_strategy,
        handover_max_adam_epochs=args.handover_max_adam_epochs,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        loss_threshold_handover=args.loss_threshold_handover,
        gradnorm_threshold=args.gradnorm_threshold,
        patience=patience,
    )

    write_summary(
        results=results,
        out_path=os.path.join(sweep_dir, "summary_table.txt"),
        k=args.wavenumber,
        qn_variant=args.qn_variant,
        n_epochs=args.epochs,
        adam_epochs=args.adam_epochs,
        engage_threshold=args.engage_threshold,
        seeds=seeds,
        rel_l2_threshold=args.success_rel_l2_threshold,
    )
    plot_sweep(
        results=results,
        out_path=os.path.join(sweep_dir, "boxcox_sweep_finegrained.png"),
        k=args.wavenumber,
        adam_epochs=args.adam_epochs,
        engage_threshold=args.engage_threshold,
        rel_l2_threshold=args.success_rel_l2_threshold,
    )

    # Persist raw histories for later replotting.
    np.savez(
        os.path.join(sweep_dir, "raw_histories.npz"),
        lambdas=np.asarray(lambdas, dtype=np.float64),
        seeds=np.asarray(seeds, dtype=np.int64),
        **{
            f"J_val_lam{lr.lambda_:g}_seed{s.seed}".replace(".", "p").replace("-", "m"):
                s.J_val_history for lr in results for s in lr.seeds
        },
        **{
            f"sol_l2_lam{lr.lambda_:g}_seed{s.seed}".replace(".", "p").replace("-", "m"):
                s.sol_l2_history for lr in results for s in lr.seeds
        },
    )
    print(f"All sweep artefacts written to: {os.path.abspath(sweep_dir)}")


if __name__ == "__main__":
    main()
