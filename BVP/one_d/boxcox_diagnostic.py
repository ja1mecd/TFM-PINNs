"""
Box-Cox diagnostic for the 1D PINN.

Three independent checks:

    1. Numerical equivalence of `expm1(lambda log s)/lambda` and `(s^lambda - 1)/lambda`
       across a wide range of s and lambda. Also checks the lambda -> 0 limit.

    2. Autograd verification: ``torch.autograd`` of the implemented transformation
       matches the analytic derivative `g'_lambda(s) = s^{lambda-1}` (and the
       second derivative) to single/double precision tolerances.

    3. Regime analysis: plot `g'_lambda(J)` and `g''_lambda(J)` against the
       *typical* J trajectory of an Adam->SSBroyden run, to localise the
       window in which the transformation could amplify curvature. This is
       what makes the lambda=1 result of section 4.2.3 of the thesis
       interpretable.

Run it with:

    python boxcox_diagnostic.py

It produces a single figure (`boxcox_diagnostic.png`) and a short text
summary printed to stdout.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
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


# =============================================================================
# Reference implementations
# =============================================================================
def boxcox_naive(s: torch.Tensor, lam: float) -> torch.Tensor:
    """Naive `(s^lambda - 1) / lambda`; degenerates for small |lambda|."""
    if lam == 0.0:
        return torch.log(s)
    return (torch.pow(s, lam) - 1.0) / lam


def boxcox_stable(s: torch.Tensor, lam: float) -> torch.Tensor:
    """Numerically-stable `expm1(lambda * log s) / lambda`."""
    if lam == 0.0:
        return torch.log(s)
    return torch.expm1(lam * torch.log(s)) / lam


def boxcox_grad_analytic(s: torch.Tensor, lam: float) -> torch.Tensor:
    """g'_lambda(s) = s^{lambda - 1}."""
    return torch.pow(s, lam - 1.0)


def boxcox_hess_analytic(s: torch.Tensor, lam: float) -> torch.Tensor:
    """g''_lambda(s) = (lambda - 1) s^{lambda - 2}."""
    return (lam - 1.0) * torch.pow(s, lam - 2.0)


# =============================================================================
# Check 1: numerical equivalence (naive vs stable)
# =============================================================================
@dataclass(frozen=True)
class EquivalenceReport:
    lambdas: tuple[float, ...]
    s_grid: tuple[float, ...]
    max_rel_err_per_lambda: tuple[float, ...]
    limit_error_at_lambda_0: float


def check_numerical_equivalence(
    lambdas: tuple[float, ...] = (-0.5, -0.1, -1e-6, 0.0, 1e-6, 0.1, 0.5, 1.0),
    s_lo: float = 1e-12,
    s_hi: float = 1e6,
    n_points: int = 2001,
) -> EquivalenceReport:
    s = torch.logspace(
        float(np.log10(s_lo)), float(np.log10(s_hi)), n_points, dtype=torch.float64
    )
    max_errs: list[float] = []
    for lam in lambdas:
        # The naive form blows up for tiny |lambda| because of catastrophic
        # cancellation; we therefore compare it to the stable form only for
        # |lambda| >= 1e-4 and verify the small-lambda branch separately.
        if abs(lam) < 1e-4:
            max_errs.append(float("nan"))
            continue
        a = boxcox_naive(s, lam)
        b = boxcox_stable(s, lam)
        rel = (a - b).abs() / (a.abs() + 1e-300)
        max_errs.append(float(rel.max().item()))

    # lambda -> 0 limit: stable form should agree with log(s) at lambda = 1e-8.
    s_test = torch.logspace(-8.0, 6.0, 2001, dtype=torch.float64)
    g_small = boxcox_stable(s_test, 1e-8)
    g_log = torch.log(s_test)
    limit_err = float((g_small - g_log).abs().max().item())

    return EquivalenceReport(
        lambdas=tuple(lambdas),
        s_grid=(s_lo, s_hi),
        max_rel_err_per_lambda=tuple(max_errs),
        limit_error_at_lambda_0=limit_err,
    )


# =============================================================================
# Check 2: autograd against analytic derivatives
# =============================================================================
@dataclass(frozen=True)
class GradReport:
    lambdas: tuple[float, ...]
    s_test: tuple[float, ...]
    max_grad_err: tuple[float, ...]
    max_hess_err: tuple[float, ...]


def check_autograd_against_analytic(
    lambdas: tuple[float, ...] = (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0),
    s_test: tuple[float, ...] = (1e-6, 1e-3, 1.0, 1e2, 1e4),
) -> GradReport:
    grad_errs: list[float] = []
    hess_errs: list[float] = []

    for lam in lambdas:
        ge_max = 0.0
        he_max = 0.0
        for s_val in s_test:
            s = torch.tensor([s_val], dtype=torch.float64, requires_grad=True)
            y = boxcox_stable(s, lam)
            g = torch.autograd.grad(y, s, create_graph=True)[0]
            h = torch.autograd.grad(g, s, create_graph=False)[0]

            g_ref = boxcox_grad_analytic(s.detach(), lam)
            h_ref = boxcox_hess_analytic(s.detach(), lam)

            ge_max = max(
                ge_max,
                float((g.detach() - g_ref).abs().item() / (g_ref.abs().item() + 1e-300)),
            )
            he_max = max(
                he_max,
                float((h.detach() - h_ref).abs().item() / (h_ref.abs().item() + 1e-300)),
            )

        grad_errs.append(ge_max)
        hess_errs.append(he_max)

    return GradReport(
        lambdas=tuple(lambdas),
        s_test=tuple(s_test),
        max_grad_err=tuple(grad_errs),
        max_hess_err=tuple(hess_errs),
    )


# =============================================================================
# Check 3: regime analysis on a typical J trajectory
# =============================================================================
@dataclass(frozen=True)
class RegimeData:
    epochs: np.ndarray
    J_train: np.ndarray
    J_val: np.ndarray
    adam_epochs: int


def collect_typical_trajectory(
    k: float,
    n_epochs: int,
    adam_epochs: int,
    n_collocation: int,
    seed: int,
    hidden: tuple[int, ...] = (64, 64, 64),
    lr: float = 1e-3,
) -> RegimeData:
    """Run a short identity-loss Adam->SSBroyden experiment to obtain the
    natural J(theta_k) range that the optimiser visits."""
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = NeuralNetwork(hidden_layers=hidden, activation=nn.Tanh())
    pinn = PINN_BVP_SSBroyden(
        model=model,
        k=k,
        lr=lr,
        loss_transform="identity",
        qn_variant="ssbroyden",
    )
    pinn.train(
        n_epochs=n_epochs,
        n_collocation=n_collocation,
        train_split=0.8,
        resample_every=500,
        adam_epochs=adam_epochs,
        verbose_freq=max(1, n_epochs // 10),
        diag_grid_n=200,
        patience=n_epochs,
        min_delta=1e-12,
        moving_avg_window=20,
    )
    epochs = np.arange(1, len(pinn.J_train) + 1)
    return RegimeData(
        epochs=epochs,
        J_train=np.asarray(pinn.J_train, dtype=np.float64),
        J_val=np.asarray(pinn.J_val, dtype=np.float64),
        adam_epochs=adam_epochs,
    )


# =============================================================================
# Plot
# =============================================================================
def plot_diagnostic(
    eq: EquivalenceReport,
    grad_rep: GradReport,
    regime: RegimeData,
    out_path: str,
    lambdas_for_regime: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0, -0.25),
) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) g_lambda(s) vs s for several lambdas — illustrative.
    s_plot = torch.logspace(-6.0, 6.0, 401, dtype=torch.float64)
    cmap = plt.get_cmap("viridis")
    n = len(lambdas_for_regime)
    for i, lam in enumerate(lambdas_for_regime):
        c = cmap(i / max(n - 1, 1))
        y = boxcox_stable(s_plot, lam).cpu().numpy()
        ax[0, 0].plot(s_plot.cpu().numpy(), y, color=c, label=rf"$\lambda={lam:g}$")
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel(r"$s = J + \varepsilon$")
    ax[0, 0].set_ylabel(r"$g_\lambda(s)$")
    ax[0, 0].set_title(r"Box-Cox transformations $g_\lambda$")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend(fontsize=9)

    # (0,1) g'_lambda(s) — the gradient scaling factor.
    for i, lam in enumerate(lambdas_for_regime):
        c = cmap(i / max(n - 1, 1))
        y = boxcox_grad_analytic(s_plot, lam).cpu().numpy()
        ax[0, 1].loglog(s_plot.cpu().numpy(), y, color=c, label=rf"$\lambda={lam:g}$")
    ax[0, 1].set_xlabel(r"$s = J + \varepsilon$")
    ax[0, 1].set_ylabel(r"$g'_\lambda(s)$  (gradient scale)")
    ax[0, 1].set_title("Gradient amplification factor")
    ax[0, 1].grid(True, which="both", alpha=0.3)
    ax[0, 1].legend(fontsize=9)

    # (1,0) |g''_lambda(s)| — curvature amplification factor.
    for i, lam in enumerate(lambdas_for_regime):
        c = cmap(i / max(n - 1, 1))
        y = np.abs(boxcox_hess_analytic(s_plot, lam).cpu().numpy())
        ax[1, 0].loglog(s_plot.cpu().numpy(), y + 1e-30, color=c, label=rf"$\lambda={lam:g}$")
    ax[1, 0].set_xlabel(r"$s = J + \varepsilon$")
    ax[1, 0].set_ylabel(r"$|g''_\lambda(s)|$  (curvature scale)")
    ax[1, 0].set_title("Curvature amplification factor")
    ax[1, 0].grid(True, which="both", alpha=0.3)
    ax[1, 0].legend(fontsize=9)

    # (1,1) Empirical J trajectory + shaded "amplification windows".
    ax[1, 1].semilogy(regime.epochs, regime.J_val, "C0-", linewidth=1.4, label=r"$\mathcal{J}_{\mathrm{val}}$")
    ax[1, 1].semilogy(regime.epochs, regime.J_train, "C3--", linewidth=1.0, label=r"$\mathcal{J}_{\mathrm{train}}$", alpha=0.7)
    ax[1, 1].axvline(regime.adam_epochs, color="k", linestyle=":", alpha=0.5, label="Adam$\\to$SSBroyden")
    j_min = float(np.min(regime.J_val))
    j_max = float(np.max(regime.J_val))
    ax[1, 1].axhspan(1e-3, max(1e-3, j_min), color="green", alpha=0.10, label="curvature-amplification window")
    ax[1, 1].axhspan(1.0, j_max, color="red", alpha=0.10, label="contraction-only window")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel(r"$\mathcal{J}$")
    ax[1, 1].set_title("Where the optimiser actually lives (empirical J trajectory)")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic figure to: {out_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Box-Cox diagnostic for the 1D PINN.")
    p.add_argument("--wavenumber", type=float, default=DEFAULT_K)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--adam-epochs", type=int, default=2000)
    p.add_argument("--n-collocation", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-trajectory", action="store_true",
                   help="Skip the (slow) trajectory collection and use synthetic J range.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("..", "results", "boxcox_diagnostic"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 72)
    print("  BOX-COX DIAGNOSTIC")
    print("=" * 72)

    # --- Check 1: equivalence ---
    eq = check_numerical_equivalence()
    print("\n[1] Naive vs stable Box-Cox: max relative error per lambda")
    for lam, err in zip(eq.lambdas, eq.max_rel_err_per_lambda):
        marker = "  (skipped, |lam| < 1e-4)" if not np.isfinite(err) else ""
        print(f"    lambda = {lam:>8.4g}  max-rel-err = {err:.3e}{marker}")
    print(f"    lambda -> 0 limit error vs log(s) at lam=1e-8: {eq.limit_error_at_lambda_0:.3e}")

    # --- Check 2: autograd ---
    grad_rep = check_autograd_against_analytic()
    print("\n[2] Autograd of g_lambda(s) vs analytic derivatives")
    print(f"    s test points: {grad_rep.s_test}")
    for lam, ge, he in zip(grad_rep.lambdas, grad_rep.max_grad_err, grad_rep.max_hess_err):
        print(f"    lambda = {lam:>5.2f}  |g' err|/|g'| <= {ge:.2e}   |g'' err|/|g''| <= {he:.2e}")

    # --- Check 3: regime ---
    if args.skip_trajectory:
        print("\n[3] Using synthetic J range (10^-6 to 10^4) — no training run.")
        epochs = np.arange(1, 5001)
        synth = np.geomspace(1e4, 1e-3, 5000)
        regime = RegimeData(epochs=epochs, J_train=synth, J_val=synth, adam_epochs=2000)
    else:
        print("\n[3] Running short Adam->SSBroyden trajectory to localise J range...")
        regime = collect_typical_trajectory(
            k=args.wavenumber,
            n_epochs=args.epochs,
            adam_epochs=args.adam_epochs,
            n_collocation=args.n_collocation,
            seed=args.seed,
        )

    j_typical = float(np.median(regime.J_val[regime.adam_epochs:]))
    print(f"    median J on the SSBroyden phase: {j_typical:.3e}")
    print("    curvature factor |g''(J_typical)| for several lambdas:")
    for lam in (1.0, 0.5, 0.25, 0.0, -0.25):
        if lam == 1.0:
            print(f"      lam = 1.00  (identity)            |g''| = 0")
        else:
            v = abs((lam - 1.0) * j_typical ** (lam - 2.0))
            print(f"      lam = {lam:>5.2f}                          |g''| = {v:.3e}")

    plot_path = os.path.join(args.out_dir, "boxcox_diagnostic.png")
    plot_diagnostic(eq, grad_rep, regime, plot_path)

    # Pass/fail summary.
    eq_ok = all(
        (not np.isfinite(e)) or e < 1e-12
        for e in eq.max_rel_err_per_lambda
    ) and eq.limit_error_at_lambda_0 < 1e-7
    ag_ok = all(e < 1e-9 for e in grad_rep.max_grad_err) and all(
        e < 1e-7 for e in grad_rep.max_hess_err
    )
    print("\nSummary:")
    print(f"  numerical equivalence : {'PASS' if eq_ok else 'CHECK'}")
    print(f"  autograd vs analytic  : {'PASS' if ag_ok else 'CHECK'}")
    print(
        f"  regime alignment      : {'OK' if 1e-3 <= j_typical <= 1.0 else 'OUT-OF-WINDOW'}\n"
        f"     (Box-Cox curvature amplification only kicks in when J lives in [1e-6, 1])"
    )

    if not (1e-3 <= j_typical <= 1.0):
        print(
            "\n  --> The empirical J range above contains values where g'' is "
            "negligible. This is the most likely reason that lambda=1 "
            "outperforms lambda<1 on this benchmark: there is no curvature "
            "to amplify in the first place.\n"
            "     Recommended remedies: longer Adam warm-up, loss "
            "normalisation by (k*pi)^4, or delayed Box-Cox engagement after "
            "J crosses a threshold (see boxcox_sweep_1d_finegrained.py)."
        )


if __name__ == "__main__":
    main()
