"""
Non-Linear Grad-Shafranov (NLGS) PINN — replication of section 4.2 of

    Urbán, Stefanou & Pons, "Unveiling the optimization process of
    physics informed neural networks: How accurate and competitive can
    PINNs be?", J. Comp. Phys. 523, 113656 (2025).

Equation (eq. 28 of the paper, with a non-vanishing toroidal function T(P)):

    Delta_GS P + T(P) * T'(P) = 0,

where (eq. 29, in compactified spherical coordinates q = 1/r, mu = cos(theta))

    Delta_GS = q^2 ( q^2 d^2/dq^2 + 2 q d/dq ) + q^2 (1 - mu^2) d^2/dmu^2
             = q^4 P_qq + 2 q^3 P_q + q^2 (1 - mu^2) P_mumu,

and the toroidal function (eq. 36, generalised to negative P) is

    T(P) = s * (|P| - P_c)^sigma     if |P| > P_c,
           0                          otherwise.

Hence T'(P) = s * sigma * (|P| - P_c)^(sigma - 1) * sign(P) above the
threshold, and T(P) T'(P) is C^1 smooth across |P| = P_c provided sigma >= 2.

Domain (Table 1, NLGS row): (q, mu) in [0, 1] x [-1, 1].

Boundary conditions (paper §4.2): a richer surface field with eight
multipoles, b_l != 0 for 1 <= l <= 8, with the same hard-enforcement ansatz
as CFGS,

    P(q, mu) = f_b(q, mu) + h_b(q, mu) * N(q, mu; theta),
    f_b     = q * (1 - mu^2) * sum_{l=1}^{8} b_l * P'_l(mu),
    h_b     = q * (q - 1) * (1 - mu^2).

The non-linearity precludes a closed-form solution, so the primary validation
metric is the L^2 norm of the discretised PDE residual on a fine grid (paper
fig. 4 right panels). The ``compute_pde_l2`` diagnostic returns this metric.
A reference linear (CFGS) solution P_an built from the same b_l coefficients
is also tracked as a sanity-check baseline; it is *not* the exact solution of
the non-linear problem but provides a meaningful regression target during the
Adam warm-start phase, when the residual is still dominated by the linear
part.

Training pipeline (paper, Table 1 NLGS):
    - tanh activations
    - Layers: 2, Neurons: 30
    - Adam for 10 000 iterations, then quasi-Newton for 10 000 more (total 20 000)
    - Batch (collocation) size: 8000; training set refreshed every 500 iters
    - Loss: MSE of the PDE residual over the interior

This script exposes the same knobs as the CFGS solver:
    --variant       one of {"bfgs", "ssbfgs", "ssbroyden"}
    --loss_transform one of {"identity", "sqrt", "log", "boxcox"}

plus the Phase A / Phase B lambda schedule introduced for the Box-Cox
generalisation; see ``loss_lambda_schedule`` in train(). The ``boxcox``
option applies the Box-Cox transformation
g_lambda(J + eps) = (expm1(lambda * log(J + eps))) / lambda
(or log(J + eps) when lambda == 0), evaluated in a numerically stable form
that avoids catastrophic cancellation for small |lambda|.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")  # headless: no display needed on the remote server
import matplotlib.pyplot as plt  # noqa: E402

# Make the shared optimizer importable when running `python pinn_ssbroyden_2d.py`
# from this directory (BVP/two_d/).
_OPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimizers")
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)
from ssbroyden import SSBroydenOptimizer  # noqa: E402


# =============================================================================
# DEVICE
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    _ = torch.zeros(1, device=device)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# =============================================================================
# PROBLEM SETUP: NLGS in compactified spherical (q, mu)
# =============================================================================
q_min, q_max = 0.0, 1.0
mu_min, mu_max = -1.0, 1.0

# Eight-multipole surface coefficients b_l for l = 1, ..., 8.
# Paper §4.2: "the boundary condition consists of eight multipoles, so that
# b_{l<=8} != 0". The specific values are a free parameter that controls the
# richness of the surface field; we use a slowly decaying tail in l so the
# dipole still dominates and the higher harmonics add fine structure.
B_COEFFS = (1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08)

# Toroidal-function parameters (eq. 36 of the paper):
# T(P) = s * (|P| - P_c)^sigma  for |P| > P_c, else 0.
# We use sigma = 2 so that T(P) T'(P) is C^1 across |P| = P_c, which keeps
# second-order autodiff of the residual well behaved.
T_S = 1.0
T_PC = 0.05
T_SIGMA = 2.0


def legendre_derivatives(mu: torch.Tensor, lmax: int) -> list[torch.Tensor]:
    """Return [P'_1(mu), P'_2(mu), ..., P'_lmax(mu)] using the stable recursion

        P_0  = 1,  P_1 = mu,
        P_l  = ((2l-1) mu P_{l-1} - (l-1) P_{l-2}) / l,
        P'_l = (l / (mu^2 - 1)) * (mu P_l - P_{l-1})  for mu^2 != 1.

    To avoid the singularity at mu = +/-1 we use the equivalent form

        P'_l = l * (P_{l-1} - mu * P_l) / (1 - mu^2 + eps)

    with a small epsilon clamp. Returns a list of tensors of the same shape as mu.
    """
    eps = 1e-12
    P = [torch.ones_like(mu), mu.clone()]
    for l in range(2, lmax + 1):
        P_next = ((2 * l - 1) * mu * P[l - 1] - (l - 1) * P[l - 2]) / l
        P.append(P_next)

    one_minus_mu2 = torch.clamp(1.0 - mu**2, min=eps)
    derivs: list[torch.Tensor] = []
    for l in range(1, lmax + 1):
        dP = l * (P[l - 1] - mu * P[l]) / one_minus_mu2
        derivs.append(dP)
    return derivs


def _surface_sum(mu: torch.Tensor, b_coeffs=B_COEFFS) -> torch.Tensor:
    """Return sum_l b_l * P'_l(mu)."""
    derivs = legendre_derivatives(mu, lmax=len(b_coeffs))
    total = torch.zeros_like(mu)
    for b, dP in zip(b_coeffs, derivs):
        total = total + b * dP
    return total


def P_linear_reference(qmu: torch.Tensor, b_coeffs=B_COEFFS) -> torch.Tensor:
    """Linear (CFGS, T = 0) reference solution built from the same surface
    multipoles, paper eq. 32.

    Used as a sanity baseline only: it is the *exact* solution of the linear
    problem with these b_l coefficients but only an approximation of the
    non-linear NLGS solution. Useful to monitor early-training behaviour
    where the residual is still dominated by the linear Delta_GS part.
    """
    q = qmu[:, 0:1]
    mu = qmu[:, 1:2]
    derivs = legendre_derivatives(mu, lmax=len(b_coeffs))
    poly = torch.zeros_like(q)
    for l, (b, dP) in enumerate(zip(b_coeffs, derivs), start=1):
        poly = poly + (q**l) * b * dP
    return (1.0 - mu**2) * poly


# Backwards-compatible alias so any pre-existing code path that imports
# P_exact still resolves; for NLGS this is *not* the exact solution.
P_exact = P_linear_reference


def toroidal_TT_prime(
    P: torch.Tensor,
    s: float = T_S,
    P_c: float = T_PC,
    sigma: float = T_SIGMA,
) -> torch.Tensor:
    """Compute T(P) * T'(P) for the toroidal function of paper eq. 36,

        T(P) = s * (|P| - P_c)^sigma   for |P| > P_c, else 0,

    and its derivative

        T'(P) = s * sigma * (|P| - P_c)^(sigma - 1) * sign(P)
                  for |P| > P_c, else 0.

    Hence

        T(P) T'(P) = s^2 * sigma * (|P| - P_c)^(2 sigma - 1) * sign(P)
                       on |P| > P_c, else 0.

    The mask is implemented with a smooth softplus-like clamp so that the
    function is differentiable through autograd. We use the relu-clamp
    `torch.clamp(|P| - P_c, min=0)` inside a power, which is C^{2 sigma - 2}
    smooth at the threshold; for sigma >= 2 this yields C^{>=2} continuity,
    sufficient for second-order residual autodiff.
    """
    abs_P = torch.abs(P)
    excess = torch.clamp(abs_P - P_c, min=0.0)
    # T(P) T'(P) = s^2 * sigma * excess^(2*sigma - 1) * sign(P)
    return (s * s) * sigma * excess ** (2.0 * sigma - 1.0) * torch.sign(P)


def f_b(qmu: torch.Tensor, b_coeffs=B_COEFFS) -> torch.Tensor:
    """Smooth function satisfying the Dirichlet BCs (paper eq. 30)."""
    q = qmu[:, 0:1]
    mu = qmu[:, 1:2]
    return q * (1.0 - mu**2) * _surface_sum(mu, b_coeffs)


def h_b(qmu: torch.Tensor) -> torch.Tensor:
    """Bubble that vanishes on the boundary of the (q, mu) rectangle (paper eq. 31)."""
    q = qmu[:, 0:1]
    mu = qmu[:, 1:2]
    return q * (q - 1.0) * (1.0 - mu**2)


# =============================================================================
# Neural network P_theta(q, mu)
# =============================================================================
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers=(32, 32, 32), activation=None) -> None:
        super().__init__()
        activation = activation if activation is not None else nn.Tanh()
        layers: list[nn.Module] = []
        in_dim = 2
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# PINN solver for the NLGS equation
# =============================================================================
class PINN_NLGS_Solver:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lambda_pde: float = 1.0,
        loss_transform: str = "identity",  # "identity" | "sqrt" | "log" | "boxcox"
        loss_lambda: float = 0.5,
        loss_eps: float = 1e-12,
        rel_err_eps: float = 1e-12,
        qn_variant: str = "ssbroyden",  # "bfgs" | "ssbfgs" | "ssbroyden"
        qn_H_on_cpu: bool = False,
        b_coeffs: tuple[float, ...] = B_COEFFS,
        toroidal_s: float = T_S,
        toroidal_Pc: float = T_PC,
        toroidal_sigma: float = T_SIGMA,
    ) -> None:
        self.model = model.to(device)
        self.lambda_pde = float(lambda_pde)
        self.loss_transform = str(loss_transform)
        self.loss_lambda = float(loss_lambda)
        self.loss_eps = float(loss_eps)
        self.rel_err_eps = float(rel_err_eps)
        self.b_coeffs = tuple(b_coeffs)
        self.toroidal_s = float(toroidal_s)
        self.toroidal_Pc = float(toroidal_Pc)
        self.toroidal_sigma = float(toroidal_sigma)

        self.adam = optim.Adam(self.model.parameters(), lr=lr)
        self.quasi_newton = SSBroydenOptimizer(
            self.model.parameters(),
            variant=qn_variant,
            lr=1.0,
            line_search=True,
            c1=1e-4,
            backtrack=0.5,
            max_ls=20,
            damping=1e-12,
            tau_min=1e-6,
            tau_max=1.0,
            reset_on_fail=True,
            H_on_cpu=qn_H_on_cpu,
        )

        # Logs
        self.obj_train: list[float] = []
        self.obj_val: list[float] = []
        self.J_train: list[float] = []
        self.J_val: list[float] = []
        self.pde_l2: list[float] = []
        self.sol_l2: list[float] = []
        self.sol_rel_l2: list[float] = []

        self.best_state: dict | None = None
        self.best_val_ma = float("inf")

        # (epoch, lambda) entries recorded each time the phase_ab schedule
        # commits to a new lambda. Empty when schedule is not used.
        self.lambda_history: list[tuple[int, float]] = []

    # ---- transform J -> objective ----
    def _transform_objective(self, J_raw: torch.Tensor) -> torch.Tensor:
        eps = self.loss_eps
        if self.loss_transform == "identity":
            return J_raw
        if self.loss_transform == "sqrt":
            return torch.sqrt(J_raw + eps)
        if self.loss_transform == "log":
            return torch.log(J_raw + eps)
        if self.loss_transform == "boxcox":
            # Box-Cox transformation g_lambda(J + eps) = (expm1(lam * log(J + eps))) / lam,
            # falling back to log(J + eps) at lam == 0. The expm1 form avoids the
            # catastrophic cancellation of the naive ((J + eps)^lam - 1) / lam expression
            # for small |lam|, which is exactly the regime where Box-Cox is most useful
            # as a continuous interpolation between sqrt (lam=0.5) and log (lam=0).
            lam = self.loss_lambda
            shifted = J_raw + eps
            if lam == 0.0:
                return torch.log(shifted)
            return torch.exp(lam * torch.log(shifted)) / lam
        raise ValueError(f"Unknown loss_transform={self.loss_transform!r}")

    # ---- hard Dirichlet BC on the (q, mu) rectangle ----
    def _P_hat(self, qmu: torch.Tensor) -> torch.Tensor:
        return f_b(qmu, self.b_coeffs) + h_b(qmu) * self.model(qmu)

    # ---- Grad-Shafranov linear part: Delta_GS P = q^4 P_qq + 2 q^3 P_q
    #      + q^2 (1-mu^2) P_mumu, plus the NLGS toroidal source T(P) T'(P).
    #      Returns (residual, P) so the caller can also use P for diagnostics. ----
    def _residual(
        self, qmu: torch.Tensor, create_graph_second: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qmu = qmu.to(device)
        if not qmu.requires_grad:
            qmu = qmu.requires_grad_(True)

        P = self._P_hat(qmu)

        grads = torch.autograd.grad(
            P, qmu, grad_outputs=torch.ones_like(P), create_graph=True
        )[0]
        P_q = grads[:, 0:1]
        P_mu = grads[:, 1:2]

        P_qq = torch.autograd.grad(
            P_q,
            qmu,
            grad_outputs=torch.ones_like(P_q),
            create_graph=create_graph_second,
            retain_graph=True,
        )[0][:, 0:1]

        P_mumu = torch.autograd.grad(
            P_mu,
            qmu,
            grad_outputs=torch.ones_like(P_mu),
            create_graph=create_graph_second,
        )[0][:, 1:2]

        q = qmu[:, 0:1]
        mu = qmu[:, 1:2]
        delta_gs = (
            q**4 * P_qq
            + 2.0 * q**3 * P_q
            + q**2 * (1.0 - mu**2) * P_mumu
        )
        # Non-linear toroidal source: residual = Delta_GS P + T(P) T'(P).
        TTp = toroidal_TT_prime(
            P,
            s=self.toroidal_s,
            P_c=self.toroidal_Pc,
            sigma=self.toroidal_sigma,
        )
        return delta_gs + TTp, P

    # Backwards-compatible alias for any helper that still expects only the
    # operator value; returns just the residual tensor.
    def _delta_gs(self, qmu: torch.Tensor, create_graph_second: bool) -> torch.Tensor:
        res, _ = self._residual(qmu, create_graph_second=create_graph_second)
        return res

    # ---- loss (objective + raw MSE residual) ----
    def compute_loss(self, qmu_interior: torch.Tensor, create_graph_second: bool):
        qmu = qmu_interior.detach().clone().requires_grad_(True)
        res, _ = self._residual(qmu, create_graph_second=create_graph_second)
        area = (q_max - q_min) * (mu_max - mu_min)
        J_raw = self.lambda_pde * (torch.mean(res**2) * area)
        J_obj = self._transform_objective(J_raw)
        return J_obj, J_raw.detach()

    # ---- diagnostics on a uniform (q, mu) grid ----
    def _grid(self, n: int):
        qs = np.linspace(q_min, q_max, n)
        mus = np.linspace(mu_min, mu_max, n)
        QQ, MM = np.meshgrid(qs, mus, indexing="xy")
        QM = np.stack([QQ.ravel(), MM.ravel()], axis=1).astype(np.float32)
        return qs, mus, QQ, MM, QM

    def compute_pde_l2(self, n: int = 60) -> float:
        qs, mus, _, _, QM = self._grid(n)
        QMt = torch.from_numpy(QM).to(device)
        res = (
            self._delta_gs(QMt, create_graph_second=False)
            .detach()
            .cpu()
            .numpy()
            .reshape(n, n)
        )
        intMu = np.trapz(res**2, mus, axis=0)
        return float(np.sqrt(np.trapz(intMu, qs, axis=0)))

    def compute_sol_l2(self, n: int = 60) -> float:
        qs, mus, _, _, QM = self._grid(n)
        QMt = torch.from_numpy(QM).to(device)
        u_true = P_exact(QMt).detach().cpu().numpy().reshape(n, n)
        with torch.no_grad():
            u_pred = self._P_hat(QMt).cpu().numpy().reshape(n, n)
        diff = u_pred - u_true
        intMu = np.trapz(diff**2, mus, axis=0)
        return float(np.sqrt(np.trapz(intMu, qs, axis=0)))

    def compute_sol_rel_l2(self, n: int = 60) -> float:
        qs, mus, _, _, QM = self._grid(n)
        QMt = torch.from_numpy(QM).to(device)
        u_true = P_exact(QMt).detach().cpu().numpy().reshape(n, n)
        with torch.no_grad():
            u_pred = self._P_hat(QMt).cpu().numpy().reshape(n, n)
        diff = u_pred - u_true
        num = np.trapz(np.trapz(diff**2, mus, axis=0), qs, axis=0)
        den = np.trapz(np.trapz(u_true**2, mus, axis=0), qs, axis=0)
        return float(np.sqrt(num) / (np.sqrt(den) + self.rel_err_eps))

    # ---- low-level step helpers (used by both the main loop and the
    #      Phase B trial scan in the boxcox/phase_ab schedule) ----
    def _adam_step(self, qmu_train: torch.Tensor) -> tuple[float, float]:
        self.adam.zero_grad()
        J_obj, J_raw = self.compute_loss(qmu_train, create_graph_second=True)
        J_obj.backward()
        self.adam.step()
        return float(J_obj.item()), float(J_raw.item())

    def _qn_step(self, qmu_train: torch.Tensor) -> tuple[float, float]:
        holder: dict = {}

        def closure():
            self.quasi_newton.zero_grad()
            J_obj_c, J_raw_c = self.compute_loss(qmu_train, create_graph_second=True)
            holder["J_raw"] = J_raw_c
            J_obj_c.backward()
            return J_obj_c

        def loss_eval():
            J_obj_e, _ = self.compute_loss(qmu_train, create_graph_second=False)
            return J_obj_e

        J_obj = self.quasi_newton.step(closure, loss_eval)
        return float(J_obj.item()), float(holder["J_raw"].item())

    # ---- snapshot / restore for the lambda trial scan ----
    def _save_qn_snapshot(self) -> dict:
        H = self.quasi_newton.H
        return {
            "model": {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            },
            "H": H.detach().clone() if H is not None else None,
        }

    def _restore_qn_snapshot(self, snap: dict) -> None:
        self.model.load_state_dict(
            {k: v.to(device) for k, v in snap["model"].items()}
        )
        self.quasi_newton.H = (
            snap["H"].clone() if snap["H"] is not None else None
        )

    # ---- Phase B trial scan: run K QN steps for each candidate lambda
    #      from a saved snapshot, pick the one with the lowest trailing
    #      raw validation residual, restore its end state. Resets H to
    #      identity for every candidate (including the incumbent) so that
    #      the comparison is fair under the changed objective geometry. ----
    def _phase_b_trial_block(
        self,
        qmu_train: torch.Tensor,
        qmu_val: torch.Tensor,
        K: int,
        candidates: list[float],
        diag_grid_n: int,
        verbose_freq: int,
        moving_avg_window: int,
    ) -> tuple[dict, float, float]:
        snap = self._save_qn_snapshot()
        trail_n = max(1, min(moving_avg_window, K))

        best = {
            "trail": float("inf"),
            "lambda": None,
            "log": None,
            "snap": None,
        }

        for c in candidates:
            self._restore_qn_snapshot(snap)
            self.loss_lambda = float(c)
            # Force a fresh identity H for fairness across candidates.
            self.quasi_newton.H = None

            log: dict = {
                "obj_train": [],
                "J_train": [],
                "obj_val": [],
                "J_val": [],
                "pde_l2": [],
                "sol_l2": [],
                "sol_rel_l2": [],
            }
            last_pde = float("nan")
            last_sol = float("nan")
            last_rel = float("nan")

            for k in range(K):
                J_obj_v, J_raw_v = self._qn_step(qmu_train)
                with torch.set_grad_enabled(True):
                    val_obj, val_raw = self.compute_loss(
                        qmu_val, create_graph_second=False
                    )
                log["obj_train"].append(J_obj_v)
                log["J_train"].append(J_raw_v)
                log["obj_val"].append(float(val_obj.item()))
                log["J_val"].append(float(val_raw.item()))

                if k == 0 or ((k + 1) % verbose_freq == 0):
                    last_pde = self.compute_pde_l2(n=diag_grid_n)
                    last_sol = self.compute_sol_l2(n=diag_grid_n)
                    last_rel = self.compute_sol_rel_l2(n=diag_grid_n)
                log["pde_l2"].append(last_pde)
                log["sol_l2"].append(last_sol)
                log["sol_rel_l2"].append(last_rel)

            trail = float(np.mean(log["J_val"][-trail_n:]))
            if trail < best["trail"]:
                best["trail"] = trail
                best["lambda"] = float(c)
                best["log"] = log
                best["snap"] = self._save_qn_snapshot()

        # Commit to the winner.
        self.loss_lambda = float(best["lambda"])
        self._restore_qn_snapshot(best["snap"])
        return best["log"], best["lambda"], best["trail"]

    # ---- training loop ----
    def train(
        self,
        n_epochs: int = 20000,
        n_collocation: int = 1000,
        train_split: float = 0.8,
        resample_every: int = 500,
        adam_epochs: int = 2000,
        verbose_freq: int = 200,
        diag_grid_n: int = 60,
        patience: int = 20000,
        min_delta: float = 1e-10,
        moving_avg_window: int = 20,
        # QN-phase early stopping (urban-style relative-MA criterion). Active
        # only after the fixed Adam warm-up (epoch > adam_epochs).
        early_stop: bool = True,
        es_patience: int = 300,
        es_window: int = 20,
        es_min_delta: float = 1e-4,
        es_stop_loss: float = 0.0,
        scheduler_patience: int = 300,
        scheduler_threshold: float = 1e-4,
        scheduler_gamma: float = 0.9,
        scheduler_min_lr: float = 1e-6,
        loss_lambda_schedule: str = "none",  # "none" | "phase_ab"
        lambda_phase_b_init: float = 0.5,
        lambda_block_size: int = 100,
        lambda_step: float = 0.1,
        lambda_min: float = -1.0,
        lambda_max: float = 1.0,
    ) -> None:
        print(
            "\nTraining NLGS PINN: Delta_GS P + T(P) T'(P) = 0  "
            "(Urban et al. 2025, sec. 4.2)"
        )
        print(
            f"  Toroidal:        s={self.toroidal_s:g}, "
            f"P_c={self.toroidal_Pc:g}, sigma={self.toroidal_sigma:g}"
        )
        print(f"  Domain:          q in [{q_min}, {q_max}], mu in [{mu_min}, {mu_max}]")
        print(f"  Surface coeffs:  b = {self.b_coeffs}")
        if self.loss_transform == "boxcox":
            print(
                f"  Loss transform:  {self.loss_transform}  "
                f"(lambda={self.loss_lambda:g}, eps={self.loss_eps:g})"
            )
        else:
            print(f"  Loss transform:  {self.loss_transform}  (eps={self.loss_eps:g})")
        print(
            f"  Optimizers:      Adam ({adam_epochs} iters)"
            f" then {self.quasi_newton.param_groups[0]['variant'].upper()}"
            f" ({n_epochs - adam_epochs} iters)"
        )
        print("-" * 80)

        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be in (0, 1).")
        if n_collocation < 2:
            raise ValueError("n_collocation must be >= 2.")
        if resample_every < 1:
            raise ValueError("resample_every must be >= 1.")
        if adam_epochs < 0 or adam_epochs >= n_epochs:
            raise ValueError("adam_epochs must be in [0, n_epochs - 1].")

        schedule_on = loss_lambda_schedule == "phase_ab"
        if schedule_on:
            if self.loss_transform != "boxcox":
                raise ValueError(
                    "loss_lambda_schedule='phase_ab' requires loss_transform='boxcox'."
                )
            if not (lambda_min <= lambda_phase_b_init <= lambda_max):
                raise ValueError(
                    "lambda_phase_b_init must lie in [lambda_min, lambda_max]."
                )
            if lambda_block_size < 1:
                raise ValueError("lambda_block_size must be >= 1.")
            if lambda_step <= 0:
                raise ValueError("lambda_step must be > 0.")
            # Phase A: identity transformation (Box-Cox at lambda=1).
            self.loss_lambda = 1.0
            print(
                f"  Lambda schedule: phase_ab  "
                f"(K={lambda_block_size}, delta={lambda_step:g}, "
                f"clip=[{lambda_min:g},{lambda_max:g}], "
                f"phase_b_init={lambda_phase_b_init:g})"
            )

        n_train = int(n_collocation * train_split)
        n_train = min(max(n_train, 1), n_collocation - 1)

        def resample_block():
            q = torch.empty(n_collocation, 1, device=device).uniform_(q_min, q_max)
            mu = torch.empty(n_collocation, 1, device=device).uniform_(mu_min, mu_max)
            qmu = torch.cat([q, mu], dim=1)
            perm = torch.randperm(n_collocation, device=device)
            qmu = qmu[perm]
            return qmu[:n_train].detach().clone(), qmu[n_train:].detach().clone()

        qmu_train, qmu_val = resample_block()

        def make_plateau(opt):
            try:
                return optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=scheduler_gamma,
                    patience=scheduler_patience,
                    threshold=scheduler_threshold,
                    min_lr=scheduler_min_lr,
                )
            except TypeError:
                return optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=scheduler_gamma,
                    patience=scheduler_patience,
                    threshold=scheduler_threshold,
                )

        sch_adam = make_plateau(self.adam)
        sch_qn = make_plateau(self.quasi_newton)

        self.best_state = None
        self.best_val_ma = float("inf")
        ma_buf: list[float] = []
        epochs_no_improve = 0

        last_pde_l2 = np.nan
        last_sol_l2 = np.nan
        last_sol_rel_l2 = np.nan

        skip_remaining = 0  # set by the trial scan to absorb the next K-1 iterations

        # QN-phase early-stop detector (urban-style relative-MA on raw val J).
        es_hist: "deque[float]" = deque(maxlen=es_window)
        es_best_ma = float("inf")
        es_bad = 0
        es_stopped_at = None
        es_reason = ""

        for epoch in range(1, n_epochs + 1):
            if skip_remaining > 0:
                skip_remaining -= 1
                continue

            if epoch != 1 and ((epoch - 1) % resample_every == 0):
                qmu_train, qmu_val = resample_block()

            # ---- Phase A -> Phase B transition (lambda schedule) ----
            if schedule_on and epoch == adam_epochs + 1:
                self.loss_lambda = float(lambda_phase_b_init)
                self.quasi_newton.H = None
                self.lambda_history.append((epoch, self.loss_lambda))
                print(
                    f"Epoch {epoch:6d} [PHASE_B_START] | "
                    f"lambda <- {self.loss_lambda:g} (H reset)"
                )

            # ---- Phase B: at every K-block boundary, run the lambda trial scan ----
            in_phase_b = epoch > adam_epochs
            phase_b_idx = epoch - adam_epochs - 1  # 0-based index within phase B
            at_block_boundary = (
                schedule_on
                and in_phase_b
                and phase_b_idx % lambda_block_size == 0
            )
            block_fits = epoch + lambda_block_size - 1 <= n_epochs
            if at_block_boundary and block_fits:
                cands_set = {
                    max(lambda_min, self.loss_lambda - lambda_step),
                    self.loss_lambda,
                    min(lambda_max, self.loss_lambda + lambda_step),
                }
                candidates = sorted(cands_set)
                trial_log, new_lambda, trail_val = self._phase_b_trial_block(
                    qmu_train,
                    qmu_val,
                    K=lambda_block_size,
                    candidates=candidates,
                    diag_grid_n=diag_grid_n,
                    verbose_freq=verbose_freq,
                    moving_avg_window=moving_avg_window,
                )

                self.obj_train.extend(trial_log["obj_train"])
                self.obj_val.extend(trial_log["obj_val"])
                self.J_train.extend(trial_log["J_train"])
                self.J_val.extend(trial_log["J_val"])
                self.pde_l2.extend(trial_log["pde_l2"])
                self.sol_l2.extend(trial_log["sol_l2"])
                self.sol_rel_l2.extend(trial_log["sol_rel_l2"])

                last_pde_l2 = trial_log["pde_l2"][-1]
                last_sol_l2 = trial_log["sol_l2"][-1]
                last_sol_rel_l2 = trial_log["sol_rel_l2"][-1]

                for v in trial_log["obj_val"]:
                    ma_buf.append(v)
                    if len(ma_buf) > moving_avg_window:
                        ma_buf.pop(0)
                val_ma = float(np.mean(ma_buf)) if ma_buf else float("inf")
                if val_ma + min_delta < self.best_val_ma:
                    self.best_val_ma = val_ma
                    self.best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += lambda_block_size

                self.lambda_history.append((epoch, float(new_lambda)))
                cand_str = ", ".join(f"{c:+.3f}" for c in candidates)
                print(
                    f"Epoch {epoch:6d} [LAMBDA_TRIAL] | candidates [{cand_str}]"
                    f" -> lambda = {new_lambda:+.3f}  "
                    f"(trail valJ = {trail_val:.3e})"
                )

                # The trial covered epochs [epoch, epoch + K - 1] inclusive.
                # The for-loop body will not run for this iteration; the next
                # K - 1 iterations are absorbed via skip_remaining.
                skip_remaining = lambda_block_size - 1
                continue

            use_adam = epoch <= adam_epochs
            opt = self.adam if use_adam else self.quasi_newton
            sch = sch_adam if use_adam else sch_qn

            if use_adam:
                opt.zero_grad()
                J_obj, J_raw = self.compute_loss(qmu_train, create_graph_second=True)
                J_obj.backward()
                opt.step()
            else:
                holder: dict = {}

                def closure():
                    opt.zero_grad()
                    J_obj_c, J_raw_c = self.compute_loss(
                        qmu_train, create_graph_second=True
                    )
                    holder["J_raw"] = J_raw_c
                    J_obj_c.backward()
                    return J_obj_c

                def loss_eval():
                    J_obj_e, _ = self.compute_loss(
                        qmu_train, create_graph_second=False
                    )
                    return J_obj_e

                J_obj = opt.step(closure, loss_eval)
                J_raw = holder["J_raw"]

            with torch.set_grad_enabled(True):
                val_obj, val_raw = self.compute_loss(
                    qmu_val, create_graph_second=False
                )

            self.obj_train.append(float(J_obj.item()))
            self.obj_val.append(float(val_obj.item()))
            self.J_train.append(float(J_raw.item()))
            self.J_val.append(float(val_raw.item()))

            ma_buf.append(float(val_obj.item()))
            if len(ma_buf) > moving_avg_window:
                ma_buf.pop(0)
            val_ma = float(np.mean(ma_buf))

            if val_ma + min_delta < self.best_val_ma:
                self.best_val_ma = val_ma
                self.best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Urban-style relative-MA early-stop counter (raw val J), QN-only.
            if early_stop and not use_adam:
                jv = float(val_raw.item())
                es_hist.append(jv)
                if es_stop_loss > 0.0 and math.isfinite(jv) and jv <= es_stop_loss:
                    es_reason = f"J_val={jv:.3e} <= stop_loss={es_stop_loss:.1e}"
                    es_stopped_at = epoch
                elif len(es_hist) == es_hist.maxlen:
                    ma = float(np.mean(es_hist))
                    if math.isfinite(ma) and ma < es_best_ma * (1.0 - es_min_delta):
                        es_best_ma = ma
                        es_bad = 0
                    else:
                        es_bad += 1
                        if es_bad >= es_patience:
                            es_reason = (
                                f"no >{es_min_delta:.1e} rel. improvement in "
                                f"MA(J_val, w={es_window}) for {es_patience} "
                                f"epochs (MA={ma:.3e}, best={es_best_ma:.3e})"
                            )
                            es_stopped_at = epoch

            sch.step(float(val_obj.item()))

            # Diagnostics every epoch (printing throttled by verbose_freq).
            last_pde_l2 = self.compute_pde_l2(n=diag_grid_n)
            last_sol_l2 = self.compute_sol_l2(n=diag_grid_n)
            last_sol_rel_l2 = self.compute_sol_rel_l2(n=diag_grid_n)
            self.pde_l2.append(last_pde_l2)
            self.sol_l2.append(last_sol_l2)
            self.sol_rel_l2.append(last_sol_rel_l2)

            if epoch == 1 or (epoch % verbose_freq == 0):
                lr_now = opt.param_groups[0]["lr"]
                phase = "ADAM" if use_adam else self.quasi_newton.param_groups[0][
                    "variant"
                ].upper()
                print(
                    f"Epoch {epoch:6d} [{phase}] | "
                    f"obj={self.obj_train[-1]:.3e}, val_obj={self.obj_val[-1]:.3e} | "
                    f"J={self.J_train[-1]:.3e}, val_J={self.J_val[-1]:.3e} | "
                    f"pdeL2={last_pde_l2:.3e}, solL2={last_sol_l2:.3e}, "
                    f"relSolL2={last_sol_rel_l2:.3e} | lr={lr_now:.2e}"
                )

            # QN-phase early stopping (urban-style; fires only after warm-up).
            if es_stopped_at is not None:
                print(
                    f"  [QN early stop] epoch {epoch} "
                    f"({epoch - adam_epochs} QN steps): {es_reason}"
                )
                break
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val-MA improvement for {patience} epochs)."
                )
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        print("-" * 80)
        print(f"Done. Best val objective moving average: {self.best_val_ma:.6e}")

    # ---- plotting ----
    def plot_results(
        self, n: int = 80, save_path: str | None = None, dpi: int = 150
    ) -> None:
        qs, mus, QQ, MM, QM = self._grid(n)
        QMt = torch.from_numpy(QM).to(device)

        u_true = P_exact(QMt).detach().cpu().numpy().reshape(n, n)
        with torch.no_grad():
            u_pred = self._P_hat(QMt).cpu().numpy().reshape(n, n)
        abs_err = np.abs(u_pred - u_true)
        rel_err = abs_err / (np.abs(u_true) + self.rel_err_eps)

        fig, ax = plt.subplots(2, 3, figsize=(16, 9))

        im0 = ax[0, 0].imshow(
            u_true, origin="lower", extent=[q_min, q_max, mu_min, mu_max], aspect="auto"
        )
        ax[0, 0].set_title(r"$P_{\mathrm{exact}}(q, \mu)$")
        plt.colorbar(im0, ax=ax[0, 0], fraction=0.046)

        im1 = ax[0, 1].imshow(
            u_pred, origin="lower", extent=[q_min, q_max, mu_min, mu_max], aspect="auto"
        )
        ax[0, 1].set_title(r"$P_{\mathrm{PINN}}(q, \mu)$")
        plt.colorbar(im1, ax=ax[0, 1], fraction=0.046)

        im2 = ax[0, 2].imshow(
            abs_err,
            origin="lower",
            extent=[q_min, q_max, mu_min, mu_max],
            aspect="auto",
        )
        ax[0, 2].set_title(r"$|P_{\mathrm{PINN}} - P_{\mathrm{exact}}|$")
        plt.colorbar(im2, ax=ax[0, 2], fraction=0.046)

        # Clip the colormap at the 99th percentile so isolated spikes
        # (where |P_exact| ~ 0 inflates the ratio) do not flatten the rest
        # of the field. The underlying values are unchanged; only the
        # visual range is bounded.
        rel_vmax = float(np.percentile(rel_err, 99))
        if not np.isfinite(rel_vmax) or rel_vmax <= 0.0:
            rel_vmax = float(np.nanmax(rel_err)) if np.isfinite(np.nanmax(rel_err)) else 1.0
        im3 = ax[1, 0].imshow(
            rel_err,
            origin="lower",
            extent=[q_min, q_max, mu_min, mu_max],
            aspect="auto",
            vmin=0.0,
            vmax=rel_vmax,
        )
        ax[1, 0].set_title(
            r"$|P_{\mathrm{PINN}} - P_{\mathrm{exact}}|/(|P_{\mathrm{exact}}| + \varepsilon)$"
            f"  (clipped at p99 = {rel_vmax:.2e})"
        )
        plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, extend="max")

        ax[1, 1].semilogy(self.obj_train, label="obj(train)")
        ax[1, 1].semilogy(self.obj_val, label="obj(val)")
        ax[1, 1].semilogy(self.J_train, "--", label="J(train)")
        ax[1, 1].semilogy(self.J_val, "--", label="J(val)")
        ax[1, 1].grid(True, alpha=0.3)
        ax[1, 1].legend()
        ax[1, 1].set_title("Objective / loss curves")
        ax[1, 1].set_xlabel("Epoch")

        ax[1, 2].semilogy(self.pde_l2, label=r"$\|\Delta_{GS} P\|_{L^2}$")
        ax[1, 2].semilogy(self.sol_l2, label=r"$\|P - P_{\mathrm{exact}}\|_{L^2}$")
        ax[1, 2].semilogy(self.sol_rel_l2, label="relative $L^2$")
        ax[1, 2].grid(True, alpha=0.3)
        ax[1, 2].legend()
        ax[1, 2].set_title("Errors over epochs")
        ax[1, 2].set_xlabel("Epoch")

        for i, j in [(0, 0), (0, 1), (0, 2), (1, 0)]:
            ax[i, j].set_xlabel("q")
            ax[i, j].set_ylabel(r"$\mu$")

        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure to: {save_path}")
        plt.close(fig)

    def save(self, path: str = "../models/pinn_nlgs_ssbroyden.pth") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to: {path}")

    # ---- numerical results ----
    def save_results(
        self,
        run_dir: str,
        n_eval: int = 80,
        extra_metadata: dict | None = None,
    ) -> None:
        """Dump training curves + final metrics + field snapshots to `run_dir`.

        Writes three files:
            history.npz    — per-epoch arrays (losses, L2 errors, ...)
            fields.npz     — final P_exact, P_pred, abs_err on the (q, mu) grid
            summary.json   — scalar metrics + hyperparameters
        """
        os.makedirs(run_dir, exist_ok=True)

        hist_path = os.path.join(run_dir, "history.npz")
        np.savez(
            hist_path,
            obj_train=np.asarray(self.obj_train, dtype=np.float64),
            obj_val=np.asarray(self.obj_val, dtype=np.float64),
            J_train=np.asarray(self.J_train, dtype=np.float64),
            J_val=np.asarray(self.J_val, dtype=np.float64),
            pde_l2=np.asarray(self.pde_l2, dtype=np.float64),
            sol_l2=np.asarray(self.sol_l2, dtype=np.float64),
            sol_rel_l2=np.asarray(self.sol_rel_l2, dtype=np.float64),
        )
        print(f"Saved training history to: {hist_path}")

        qs, mus, QQ, MM, QM = self._grid(n_eval)
        QMt = torch.from_numpy(QM).to(device)
        u_true = P_exact(QMt).detach().cpu().numpy().reshape(n_eval, n_eval)
        with torch.no_grad():
            u_pred = self._P_hat(QMt).cpu().numpy().reshape(n_eval, n_eval)
        abs_err = np.abs(u_pred - u_true)

        fields_path = os.path.join(run_dir, "fields.npz")
        np.savez(
            fields_path,
            q=qs.astype(np.float64),
            mu=mus.astype(np.float64),
            P_exact=u_true.astype(np.float64),
            P_pred=u_pred.astype(np.float64),
            abs_err=abs_err.astype(np.float64),
        )
        print(f"Saved field snapshots to: {fields_path}")

        summary = {
            "problem": "NLGS (Urban et al. 2025, sec. 4.2)",
            "qn_variant": self.quasi_newton.param_groups[0]["variant"],
            "loss_transform": self.loss_transform,
            "loss_lambda": self.loss_lambda,
            "loss_eps": self.loss_eps,
            "lambda_pde": self.lambda_pde,
            "lambda_history": [
                [int(e), float(lam)] for e, lam in self.lambda_history
            ],
            "b_coeffs": list(self.b_coeffs),
            "toroidal_s": self.toroidal_s,
            "toroidal_Pc": self.toroidal_Pc,
            "toroidal_sigma": self.toroidal_sigma,
            "domain": {
                "q": [q_min, q_max],
                "mu": [mu_min, mu_max],
            },
            "n_epochs_run": len(self.obj_train),
            "best_val_objective_ma": float(self.best_val_ma),
            "final_obj_train": float(self.obj_train[-1]) if self.obj_train else None,
            "final_obj_val": float(self.obj_val[-1]) if self.obj_val else None,
            "final_J_train": float(self.J_train[-1]) if self.J_train else None,
            "final_J_val": float(self.J_val[-1]) if self.J_val else None,
            "final_pde_l2": float(self.pde_l2[-1]) if self.pde_l2 else None,
            "final_sol_l2": float(self.sol_l2[-1]) if self.sol_l2 else None,
            "final_sol_rel_l2": (
                float(self.sol_rel_l2[-1]) if self.sol_rel_l2 else None
            ),
            "max_abs_err": float(np.max(abs_err)),
            "mean_abs_err": float(np.mean(abs_err)),
            "device": str(device),
            "torch_version": torch.__version__,
        }
        if extra_metadata is not None:
            summary.update(extra_metadata)

        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved summary to: {summary_path}")


# =============================================================================
# MAIN — Urban et al. (2025), Table 1 NLGS hyperparameters
# =============================================================================
def main() -> None:
    # --- user knobs reproducing Table 2 of the paper ---
    qn_variant = "ssbroyden"     # one of: "bfgs", "ssbfgs", "ssbroyden"
    loss_transform = "identity"  # one of: "identity", "sqrt", "log", "boxcox"
    loss_lambda = 0.5            # only used when loss_transform == "boxcox"
    # Phase A / Phase B lambda schedule (requires loss_transform == "boxcox").
    # When "phase_ab", Phase A trains with lambda=1 (identity) and Phase B
    # adapts lambda by a 3-candidate trial scan every `lambda_block_size` QN
    # epochs, starting from `lambda_phase_b_init`.
    loss_lambda_schedule = "none"  # "none" | "phase_ab"
    lambda_phase_b_init = 0.5
    lambda_block_size = 100
    lambda_step = 0.1
    lambda_min = -1.0
    lambda_max = 1.0
    qn_H_on_cpu = False          # set True if OOM on GPU
    # ---------------------------------------------------

    # Project BVP standard: 3 layers x 32 neurons, tanh (held fixed across all
    # BVP benchmarks). Urban et al. Table 1 NLGS used 2x30.
    model = NeuralNetwork(hidden_layers=(32, 32, 32), activation=nn.Tanh())
    print("\nNeural Network Architecture:\n")
    print(model, "\n")

    pinn = PINN_NLGS_Solver(
        model=model,
        lr=1e-3,
        lambda_pde=1.0,
        loss_transform=loss_transform,
        loss_lambda=loss_lambda,
        qn_variant=qn_variant,
        qn_H_on_cpu=qn_H_on_cpu,
        b_coeffs=B_COEFFS,
        toroidal_s=T_S,
        toroidal_Pc=T_PC,
        toroidal_sigma=T_SIGMA,
    )

    # Standardised optimiser protocol: fixed 2000-epoch Adam warm-up, then QN
    # with urban-style early stopping (generous 8000-epoch cap). batch size
    # 8000, training set refreshed every 500 iterations.
    pinn.train(
        n_epochs=10000,
        n_collocation=8000,
        train_split=0.8,
        resample_every=500,
        adam_epochs=2000,
        verbose_freq=500,
        diag_grid_n=60,
        patience=20000,      # legacy both-phase counter disabled; QN ES via es_*
        min_delta=1e-10,
        moving_avg_window=20,
        loss_lambda_schedule=loss_lambda_schedule,
        lambda_phase_b_init=lambda_phase_b_init,
        lambda_block_size=lambda_block_size,
        lambda_step=lambda_step,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    transform_tag = (
        f"boxcox_lam{loss_lambda:g}" if loss_transform == "boxcox" else loss_transform
    )
    if loss_lambda_schedule == "phase_ab":
        transform_tag = (
            f"boxcox_phaseAB_init{lambda_phase_b_init:g}"
            f"_K{lambda_block_size}_d{lambda_step:g}"
        )
    run_name = f"nlgs_{qn_variant}_{transform_tag}_{run_tag}"
    run_dir = os.path.join("..", "results", run_name)
    os.makedirs(run_dir, exist_ok=True)

    pinn.plot_results(n=80, save_path=os.path.join(run_dir, "results.png"))
    pinn.save(f"../models/pinn_nlgs_{qn_variant}_{transform_tag}.pth")
    pinn.save_results(
        run_dir,
        n_eval=80,
        extra_metadata={"run_name": run_name, "run_tag": run_tag},
    )
    print(f"\nAll run artefacts written to: {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
