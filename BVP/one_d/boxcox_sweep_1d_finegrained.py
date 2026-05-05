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


# =============================================================================
# Per-(lambda, seed) checkpointing — survives mid-sweep interruptions.
# =============================================================================
def _pair_filename(lam: float, seed: int) -> str:
    """Filesystem-safe checkpoint filename for a single (lambda, seed) pair."""
    safe_lam = f"{lam:g}".replace(".", "p").replace("-", "m")
    return f"lam{safe_lam}_seed{seed}.npz"


def save_pair(pairs_dir: str, lam: float, result: SeedResult) -> None:
    """Atomically write the SeedResult for one (lambda, seed) pair so that a
    later run with the same --resume-dir can skip it. Pass np.savez a file
    handle (not a path string) to avoid its silent ``.npz`` auto-suffix."""
    os.makedirs(pairs_dir, exist_ok=True)
    path = os.path.join(pairs_dir, _pair_filename(lam, result.seed))
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            lam=np.float64(lam),
            seed=np.int64(result.seed),
            final_J_val=np.float64(result.final_J_val),
            final_pde_l2=np.float64(result.final_pde_l2),
            final_sol_l2=np.float64(result.final_sol_l2),
            final_sol_rel_l2=np.float64(result.final_sol_rel_l2),
            engagement_epoch=np.int64(
                -1 if result.engagement_epoch is None else result.engagement_epoch
            ),
            J_val_history=result.J_val_history,
            sol_l2_history=result.sol_l2_history,
        )
    os.replace(tmp, path)


def load_pair(pairs_dir: str, lam: float, seed: int) -> Optional[SeedResult]:
    """Return a cached SeedResult, or None if the pair has not run yet."""
    path = os.path.join(pairs_dir, _pair_filename(lam, seed))
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as data:
            eng = int(data["engagement_epoch"])
            return SeedResult(
                seed=int(data["seed"]),
                final_J_val=float(data["final_J_val"]),
                final_pde_l2=float(data["final_pde_l2"]),
                final_sol_l2=float(data["final_sol_l2"]),
                final_sol_rel_l2=float(data["final_sol_rel_l2"]),
                engagement_epoch=None if eng < 0 else eng,
                J_val_history=np.asarray(data["J_val_history"], dtype=np.float64),
                sol_l2_history=np.asarray(data["sol_l2_history"], dtype=np.float64),
            )
    except Exception as exc:  # noqa: BLE001 — corrupt checkpoint, fall back to rerun.
        print(f"  [warn] failed to load checkpoint {path}: {exc}; rerunning.")
        return None


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
    pairs_dir: str,
) -> tuple[LambdaResult, ...]:
    """Iterate the (lambda, seed) grid. Each pair is checkpointed under
    `pairs_dir` immediately on completion, so a later run with the same
    `--resume-dir` skips already-finished pairs."""
    out: list[LambdaResult] = []
    n_done = 0
    n_skipped = 0
    n_total = len(lambdas) * len(seeds)
    for lam in lambdas:
        seed_results: list[SeedResult] = []
        for seed in seeds:
            cached = load_pair(pairs_dir, lam, seed)
            if cached is not None:
                print(
                    f"\n[lambda={lam:g}, seed={seed}] "
                    f"[resume] cached -> skip "
                    f"(final J_val={cached.final_J_val:.3e}, "
                    f"solL2={cached.final_sol_l2:.3e})"
                )
                seed_results.append(cached)
                n_skipped += 1
                continue
            print(
                f"\n[lambda={lam:g}, seed={seed}] "
                f"({n_done + n_skipped + 1}/{n_total})  "
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
            save_pair(pairs_dir, lam, res)
            seed_results.append(res)
            n_done += 1
        out.append(LambdaResult(lambda_=lam, seeds=tuple(seed_results)))
    print(f"\n[sweep] {n_done} new pair(s) run, {n_skipped} resumed from disk.")
    return tuple(out)


# =============================================================================
# Plotting helpers
# =============================================================================
def _pad_and_stack(seq: list[np.ndarray]) -> np.ndarray:
    """Stack histories of possibly-different lengths into (n, max_len) by
    holding the last value of each shorter run. Early-stopped trajectories
    therefore appear as flat lines past their stop epoch in the mean curve,
    which is the right semantics for "this run converged at this value"."""
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
def plot_sweep(results: tuple[LambdaResult, ...], out_path: str, k: float,
               adam_epochs: int, engage_threshold: Optional[float]) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    cmap = plt.get_cmap("viridis")
    n = len(results)

    # (0,0) Mean J(val) trajectory per lambda (averaged across seeds).
    for i, lr in enumerate(results):
        c = cmap(i / max(n - 1, 1))
        # Pad histories to common length (early stopping is disabled).
        H = _pad_and_stack([s.J_val_history for s in lr.seeds])
        mean = np.mean(H, axis=0)
        ax[0, 0].semilogy(mean, color=c, linewidth=1.3, label=rf"$\lambda={lr.lambda_:g}$")
    ax[0, 0].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5, label="Adam$\\to$SSBroyden")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel(r"$\mathcal{J}_{\mathrm{val}}$ (mean over seeds)")
    ax[0, 0].set_title("Validation residual MSE")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend(fontsize=8, ncol=2)

    # (0,1) Mean solution L2 trajectory per lambda.
    for i, lr in enumerate(results):
        c = cmap(i / max(n - 1, 1))
        H = _pad_and_stack([s.sol_l2_history for s in lr.seeds])
        mean = np.mean(H, axis=0)
        ax[0, 1].semilogy(mean, color=c, linewidth=1.3, label=rf"$\lambda={lr.lambda_:g}$")
    ax[0, 1].axvline(adam_epochs, color="k", linestyle=":", alpha=0.5)
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel(r"$\|\widehat{u} - u^\star\|_{L^2}$")
    ax[0, 1].set_title("Solution L2 error")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend(fontsize=8, ncol=2)

    # (1,0) Final J(val) mean +/- std as bar / errorbar plot.
    lams = np.asarray([r.lambda_ for r in results])
    means_J = np.asarray([r.final_J_val_mean for r in results])
    stds_J = np.asarray([r.final_J_val_std for r in results])
    means_sol = np.asarray([r.final_sol_l2_mean for r in results])
    stds_sol = np.asarray([r.final_sol_l2_std for r in results])

    ax[1, 0].errorbar(lams, means_J, yerr=stds_J, fmt="o-", color="C3",
                      label=r"final $\mathcal{J}_{\mathrm{val}}$ mean $\pm$ std")
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

    ax[1, 1].errorbar(lams, means_sol, yerr=stds_sol, fmt="s-", color="C0",
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
    print(f"Saved sweep figure to: {out_path}")
    plt.close(fig)


def write_summary(
    results: tuple[LambdaResult, ...],
    out_path: str,
    k: float,
    qn_variant: str,
    n_epochs: int,
    adam_epochs: int,
    engage_threshold: Optional[float],
    seeds: tuple[int, ...],
) -> None:
    lines: list[str] = []
    lines.append(
        f"Fine-grained Box-Cox sweep (1D BVP, k={k:g}, qn={qn_variant}, "
        f"epochs={n_epochs}, adam={adam_epochs}, seeds={list(seeds)})\n"
    )
    if engage_threshold is not None:
        lines.append(f"Delayed engagement: transformation engages once J < {engage_threshold:g}.\n")
    lines.append("\n")
    lines.append(
        f"{'lambda':>8}   "
        f"{'mean J':>16}  {'std J':>14}    "
        f"{'mean solL2':>16}  {'std solL2':>14}    "
        f"{'engaged':>10}\n"
    )
    for lr in results:
        eng = [s.engagement_epoch for s in lr.seeds if s.engagement_epoch is not None]
        eng_str = f"epoch~{int(np.mean(eng))}" if eng else "never/from-start"
        lines.append(
            f"{lr.lambda_:>8.4g}   "
            f"{lr.final_J_val_mean:>16.4e}  {lr.final_J_val_std:>14.4e}    "
            f"{lr.final_sol_l2_mean:>16.4e}  {lr.final_sol_l2_std:>14.4e}    "
            f"{eng_str:>10}\n"
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
        default=200,
        help="Early-stopping patience on the validation MA. Default 200 "
             "(stagnant runs terminate ~200 epochs after the loss plateau). "
             "Set --patience equal to --epochs to disable early stopping "
             "and force every (lambda, seed) pair to run the full budget.",
    )
    p.add_argument("--n-collocation", type=int, default=400)
    p.add_argument("--resample-every", type=int, default=500)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join("..", "results"),
    )
    p.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Reuse this output directory instead of creating a fresh "
             "timestamped one. Per-(lambda, seed) checkpoints already in this "
             "directory are loaded; missing pairs are computed and "
             "checkpointed. Used to resume after a connection drop.",
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
    patience = args.patience

    if args.resume_dir is not None:
        sweep_dir = os.path.abspath(args.resume_dir)
        os.makedirs(sweep_dir, exist_ok=True)
        mode_msg = f"RESUME mode -> {sweep_dir}"
    else:
        run_tag = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_delayed" if args.engage_threshold is not None else ""
        sweep_dir = os.path.join(
            args.results_dir,
            f"bvp1d_k{args.wavenumber:g}_boxcox_finesweep{suffix}_{args.qn_variant}_{run_tag}",
        )
        os.makedirs(sweep_dir, exist_ok=True)
        mode_msg = f"FRESH sweep -> {sweep_dir}"

    pairs_dir = os.path.join(sweep_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    print(
        f"\nFine-grained Box-Cox sweep on the 1D BVP "
        f"(k={args.wavenumber:g}, qn={args.qn_variant}, "
        f"epochs={args.epochs}, adam={args.adam_epochs})\n"
        f"  {mode_msg}\n"
        f"  lambdas:           {lambdas}\n"
        f"  seeds:             {seeds}\n"
        f"  engage_threshold:  {args.engage_threshold}\n"
        f"  handover_strategy: {args.handover_strategy}\n"
        f"  early-stop patience: {patience}"
        f"{' (== epochs => disabled)' if patience >= args.epochs else ''}\n"
        f"  per-pair checkpoints under: {pairs_dir}\n"
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
        pairs_dir=pairs_dir,
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
    )
    plot_sweep(
        results=results,
        out_path=os.path.join(sweep_dir, "boxcox_sweep_finegrained.png"),
        k=args.wavenumber,
        adam_epochs=args.adam_epochs,
        engage_threshold=args.engage_threshold,
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
