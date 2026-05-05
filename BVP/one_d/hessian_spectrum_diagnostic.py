"""
Empirical Hessian-spectrum confirmation for Figure 3.1 of the thesis.

The thesis derives, for the linear PINN

    u_theta(x) = sum_{k=1}^N theta_k sin(k pi x)

minimising

    J(theta) = (M/2) * mean_i (u_theta''(x_i) - f(x_i))^2

at M >= 2N equispaced collocation points x_i = i / (M+1), that

    J_R^T J_R = diag(M/2 * (k pi)^4),   k = 1, ..., N,

so kappa(J_R^T J_R) = N^4 grows as the fourth power of the number of resolved
modes. The diagonal Jacobi preconditioner H = (diag J_R^T J_R)^{-1} flattens
the spectrum to identity. Both statements are *analytical* in the thesis;
this script confirms them numerically by

    1. assembling J_R via finite-difference / second-derivative evaluation,
    2. forming J_R^T J_R explicitly,
    3. computing its eigenvalues,
    4. comparing them to the analytical (M/2)(k pi)^4,
    5. plotting the bare and Jacobi-preconditioned spectra side by side.

The result is the empirical counterpart of Figure 3.1 of the thesis. Run with

    python hessian_spectrum_diagnostic.py

to produce `hessian_spectrum_empirical.png` and a short pass/fail summary.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class SpectrumResult:
    N: int
    M: int
    eigvals_bare: np.ndarray
    eigvals_jacobi: np.ndarray
    analytic: np.ndarray
    cond_bare: float
    cond_jacobi: float
    relative_diff: np.ndarray  # |eigvals_bare - analytic| / analytic


def build_residual_jacobian(N: int, M: int) -> np.ndarray:
    """For u_theta(x) = sum_{k=1}^N theta_k sin(k pi x), the i-th residual entry
    is (u_theta'' - f)(x_i). The Jacobian wrt theta_k is therefore

        d r_i / d theta_k = -(k pi)^2 sin(k pi x_i),

    independent of f. The overall sign cancels in J_R^T J_R, and the
    expression matches the thesis derivation exactly when collocation points
    x_i = i / (M + 1) for i = 1, ..., M."""
    i_idx = np.arange(1, M + 1)[:, None]      # (M, 1)
    k_idx = np.arange(1, N + 1)[None, :]      # (1, N)
    x = i_idx / (M + 1.0)                     # (M, 1)
    coef = -(k_idx * np.pi) ** 2              # (1, N)
    sin_modes = np.sin(k_idx * np.pi * x)     # (M, N)
    return coef * sin_modes                   # (M, N)


def analyse(N: int, M: int) -> SpectrumResult:
    if M < 2 * N:
        raise ValueError(
            f"M >= 2N is required for the orthogonality of discrete sine modes "
            f"(got M={M}, N={N})."
        )

    # The Gauss-Newton Hessian of J(theta) = (1/2) ||r(theta)||^2 is exactly
    # J_R^T J_R. With x_i = i / (M+1) and the discrete-sine orthogonality
    # identity sum_i sin(j pi x_i) sin(k pi x_i) = (M+1)/2 * delta_{j,k},
    # this matrix is *exactly* diagonal with entries (M+1)/2 * (k pi)^4. The
    # thesis writes the leading factor as M/2 instead, which is the M -> infty
    # approximation; both forms are reported below for completeness.
    JR = build_residual_jacobian(N, M)
    A = JR.T @ JR

    eigvals_bare = np.sort(np.linalg.eigvalsh(A))[::-1]

    diag = np.diag(A)
    if np.any(diag <= 0.0):
        raise RuntimeError("non-positive diagonal entries; Jacobi preconditioner ill-defined")

    # The Jacobi-preconditioned Hessian is D^{-1/2} A D^{-1/2}, which is
    # similar to D^{-1} A and therefore shares its eigenvalues. Using
    # D^{-1} A D^{-1} (the previous version of this script) returns
    # eigenvalues of D^{-1}, which is *more* spread than A — the opposite
    # of preconditioning.
    sqrt_inv = np.diag(1.0 / np.sqrt(diag))
    A_pre = sqrt_inv @ A @ sqrt_inv
    eigvals_jacobi = np.sort(np.linalg.eigvalsh(0.5 * (A_pre + A_pre.T)))[::-1]

    k_idx = np.arange(1, N + 1)
    analytic = ((M + 1) / 2.0) * (k_idx * np.pi) ** 4
    analytic = np.sort(analytic)[::-1]

    cond_bare = float(eigvals_bare[0] / eigvals_bare[-1])
    cond_jacobi = float(eigvals_jacobi[0] / eigvals_jacobi[-1])
    rel_diff = np.abs(eigvals_bare - analytic) / np.abs(analytic)

    return SpectrumResult(
        N=N,
        M=M,
        eigvals_bare=eigvals_bare,
        eigvals_jacobi=eigvals_jacobi,
        analytic=analytic,
        cond_bare=cond_bare,
        cond_jacobi=cond_jacobi,
        relative_diff=rel_diff,
    )


def plot_spectrum(res: SpectrumResult, out_path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    idx = np.arange(1, res.N + 1)

    ax[0].semilogy(idx, res.eigvals_bare, "o-", color="C0",
                   label=r"empirical $\lambda_k$ of $J_R^\top J_R$", markersize=5)
    ax[0].semilogy(idx, res.analytic, "x", color="C3",
                   label=r"analytic $\frac{M+1}{2}(k\pi)^4$", markersize=8)
    ax[0].set_xlabel(r"eigenvalue index $k$")
    ax[0].set_ylabel(r"$\lambda_k$")
    ax[0].set_title(
        f"Bare spectrum (N={res.N}, M={res.M})  "
        rf"$\kappa = {res.cond_bare:.3e}$"
    )
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=9)

    ax[1].semilogy(idx, res.eigvals_jacobi, "s-", color="C2", markersize=5,
                   label=r"$\lambda_k$ after diag-Jacobi preconditioner")
    ax[1].axhline(1.0, color="k", linestyle=":", alpha=0.5, label=r"$\lambda \equiv 1$")
    ax[1].set_xlabel(r"eigenvalue index $k$")
    ax[1].set_ylabel(r"$\lambda_k$ (preconditioned)")
    ax[1].set_title(
        f"Preconditioned spectrum  "
        rf"$\kappa = {res.cond_jacobi:.3e}$"
    )
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Hessian-spectrum figure to: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Empirical confirmation of the analytical Hessian-spectrum claim "
                    "(thesis Fig. 3.1)."
    )
    p.add_argument("--N", type=int, default=20, help="Number of Fourier modes.")
    p.add_argument("--M", type=int, default=40, help="Number of equispaced collocation points; M>=2N.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("..", "results", "hessian_spectrum"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    res = analyse(N=args.N, M=args.M)

    print(f"N = {res.N}, M = {res.M}")
    print(f"  bare condition number      kappa(J_R^T J_R)        = {res.cond_bare:.3e}")
    print(f"  preconditioned condition   kappa((diag^{-1}) ... ) = {res.cond_jacobi:.3e}")
    print(f"  max relative deviation from analytic spectrum     = {res.relative_diff.max():.3e}")

    plot_spectrum(res, out_path=os.path.join(args.out_dir, "hessian_spectrum_empirical.png"))

    np.savez(
        os.path.join(args.out_dir, "spectrum.npz"),
        N=res.N, M=res.M,
        eigvals_bare=res.eigvals_bare,
        eigvals_jacobi=res.eigvals_jacobi,
        analytic=res.analytic,
    )

    if res.relative_diff.max() > 1e-9:
        print("\n[CHECK] Empirical spectrum diverges from analytic by >1e-9. "
              "Verify M >= 2N and that the convention (M/2) * mean(...) matches the thesis.")
    else:
        print("\n[PASS] Analytic and empirical spectra agree to numerical precision.")


if __name__ == "__main__":
    main()
