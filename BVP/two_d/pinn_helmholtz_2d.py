"""
2D Helmholtz (2DH) PINN — replication of section 5, "2D Helmholtz equation",
of

    Urbán, Stefanou & Pons, "Unveiling the optimization process of
    physics informed neural networks: How accurate and competitive can
    PINNs be?", J. Comp. Phys. 523, 113656 (2025).

Equation (eqs. 37-38 of the paper):

    ∇^2 u + k^2 u - q(x, y) = 0,
    q(x, y) = -sin(pi a1 x) sin(pi a2 y) [pi^2 (a1^2 + a2^2) - k^2],

with analytic solution

    u_exact(x, y) = sin(pi a1 x) sin(pi a2 y),

on the square [-1, 1] x [-1, 1]. The wavenumbers (a1, a2) are integers and
k must satisfy k^2 != pi^2 (n^2 + m^2) for any (n, m) in Z^2 so that the
homogeneous problem with periodic BCs has only the zero solution.

Boundary conditions are *periodic* in x and y. They are hard-enforced by
input encoding: the network sees the lifted input

    (cos(pi x), sin(pi x), cos(pi y), sin(pi y))

and outputs u directly, without any additive bubble. This guarantees
u(x + 2, y) = u(x, y) and u(x, y + 2) = u(x, y) by construction (paper eq. 39).

Two configurations from the paper:
    - Low wavenumber  (a1, a2) = (1, 4), k = 1: 2 layers, 20 neurons,
      20 000 iterations, 5 000 Adam, batch 10 000.
    - High wavenumber (a1, a2) = (6, 6), k = 1: 3 layers, 30 neurons,
      50 000 iterations, 5 000 Adam, batch 10 000.

The quasi-Newton variant, the loss transform, and the Phase A / Phase B
lambda schedule can be toggled in `main()`. The ``boxcox`` option applies
g_lambda(J + eps) = expm1(lambda * log(J + eps)) / lambda  (or log(J + eps)
when lambda == 0), evaluated in a numerically stable form that avoids
catastrophic cancellation for small |lambda|.
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
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# =============================================================================
# PROBLEM SETUP: 2D Helmholtz on [-1, 1] x [-1, 1] with periodic BCs
# =============================================================================
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0

# Default to the low-wavenumber configuration. Override in main() for the
# high-wavenumber case (a1 = a2 = 6, k = 1).
A1_WAVENUMBER = 1
A2_WAVENUMBER = 4
K_WAVENUMBER = 1


def u_exact(
    xy: torch.Tensor,
    a1: int = A1_WAVENUMBER,
    a2: int = A2_WAVENUMBER,
) -> torch.Tensor:
    """Analytic solution u(x, y) = sin(pi a1 x) sin(pi a2 y) (paper, p. 12)."""
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return torch.sin(np.pi * a1 * x) * torch.sin(np.pi * a2 * y)


def q_source(
    xy: torch.Tensor,
    a1: int = A1_WAVENUMBER,
    a2: int = A2_WAVENUMBER,
    k: float = K_WAVENUMBER,
) -> torch.Tensor:
    """Source term q(x, y) such that ∇^2 u_exact + k^2 u_exact = q (paper eq. 38).

    q(x, y) = -sin(pi a1 x) sin(pi a2 y) [pi^2 (a1^2 + a2^2) - k^2].
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    factor = (np.pi**2) * (a1**2 + a2**2) - k**2
    return -torch.sin(np.pi * a1 * x) * torch.sin(np.pi * a2 * y) * factor


# Backwards-compatible aliases (any historical code path expecting
# `phi_exact` / `r_source` will keep resolving).
phi_exact = u_exact
r_source = q_source


# =============================================================================
# Neural network u_theta(x, y) with periodic input encoding
# (paper eq. 39: u(x, y) = N(cos(pi x), sin(pi x), cos(pi y), sin(pi y)))
# =============================================================================
class NeuralNetwork(nn.Module):
    """MLP with a hard-enforced 2-periodic input lift.

    The forward pass takes a tensor of shape (B, 2) holding (x, y) and feeds
    the four-feature vector (cos(pi x), sin(pi x), cos(pi y), sin(pi y)) into
    the underlying MLP. Because cos and sin are 2-periodic, the resulting
    surrogate satisfies u(x + 2, y) = u(x, y) and u(x, y + 2) = u(x, y) by
    construction. This is the periodic counterpart of the additive
    Dirichlet ansatz used in the NLP and CFGS solvers.
    """

    def __init__(self, hidden_layers=(32, 32, 32), activation=None) -> None:
        super().__init__()
        activation = activation if activation is not None else nn.Tanh()
        layers: list[nn.Module] = []
        # Input dim is 4 because the periodic lift produces (cos pi x,
        # sin pi x, cos pi y, sin pi y).
        in_dim = 4
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy has shape (B, 2). Lift to the periodic feature space and run
        # the MLP. The lift uses a fixed period of 2 in each axis, matching
        # the [-1, 1] domain.
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        features = torch.cat(
            [
                torch.cos(np.pi * x),
                torch.sin(np.pi * x),
                torch.cos(np.pi * y),
                torch.sin(np.pi * y),
            ],
            dim=1,
        )
        return self.net(features)


# =============================================================================
# PINN solver
# =============================================================================
class PINN_Helmholtz_Solver:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lambda_pde: float = 1.0,
        a1: int = A1_WAVENUMBER,
        a2: int = A2_WAVENUMBER,
        k: float = K_WAVENUMBER,
        loss_transform: str = "identity",
        loss_lambda: float = 0.5,
        loss_eps: float = 1e-12,
        rel_err_eps: float = 1e-12,
        qn_variant: str = "ssbroyden",
        qn_H_on_cpu: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.lambda_pde = float(lambda_pde)
        self.a1 = int(a1)
        self.a2 = int(a2)
        self.k = float(k)
        self.loss_transform = str(loss_transform)
        self.loss_lambda = float(loss_lambda)
        self.loss_eps = float(loss_eps)
        self.rel_err_eps = float(rel_err_eps)

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
            # for small |lam|, the regime where Box-Cox interpolates smoothly between
            # the square-root (lam=0.5) and logarithmic (lam=0) transformations.
            lam = self.loss_lambda
            shifted = J_raw + eps
            if lam == 0.0:
                return torch.log(shifted)
            return torch.expm1(lam * torch.log(shifted)) / lam
        raise ValueError(f"Unknown loss_transform={self.loss_transform!r}")

    def _u_hat(self, xy: torch.Tensor) -> torch.Tensor:
        # The model encodes periodicity internally, so no additive ansatz.
        return self.model(xy)

    # Backwards-compat alias for any caller still expecting `_phi_hat`.
    _phi_hat = _u_hat

    def _residual(self, xy: torch.Tensor, create_graph_second: bool) -> torch.Tensor:
        xy = xy.to(device)
        if not xy.requires_grad:
            xy = xy.requires_grad_(True)

        u = self._u_hat(xy)

        grads = torch.autograd.grad(
            u, xy, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x,
            xy,
            grad_outputs=torch.ones_like(u_x),
            create_graph=create_graph_second,
            retain_graph=True,
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            u_y,
            xy,
            grad_outputs=torch.ones_like(u_y),
            create_graph=create_graph_second,
        )[0][:, 1:2]

        # Helmholtz residual: ∇^2 u + k^2 u - q(x, y).
        q = q_source(xy, self.a1, self.a2, self.k)
        return (u_xx + u_yy) + (self.k**2) * u - q

    def compute_loss(self, xy_interior: torch.Tensor, create_graph_second: bool):
        xy = xy_interior.detach().clone().requires_grad_(True)
        res = self._residual(xy, create_graph_second=create_graph_second)
        area = (x_max - x_min) * (y_max - y_min)
        J_raw = self.lambda_pde * (torch.mean(res**2) * area)
        J_obj = self._transform_objective(J_raw)
        return J_obj, J_raw.detach()

    def _grid(self, n: int):
        xs = np.linspace(x_min, x_max, n)
        ys = np.linspace(y_min, y_max, n)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        XY = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
        return xs, ys, XX, YY, XY

    def compute_pde_l2(self, n: int = 60) -> float:
        xs, ys, _, _, XY = self._grid(n)
        XYt = torch.from_numpy(XY).to(device)
        res = (
            self._residual(XYt, create_graph_second=False)
            .detach()
            .cpu()
            .numpy()
            .reshape(n, n)
        )
        intX = np.trapz(res**2, xs, axis=1)
        return float(np.sqrt(np.trapz(intX, ys, axis=0)))

    def compute_sol_l2(self, n: int = 60) -> float:
        xs, ys, _, _, XY = self._grid(n)
        XYt = torch.from_numpy(XY).to(device)
        u_true = u_exact(XYt, self.a1, self.a2).detach().cpu().numpy().reshape(n, n)
        with torch.no_grad():
            u_pred = self._u_hat(XYt).cpu().numpy().reshape(n, n)
        diff = u_pred - u_true
        intX = np.trapz(diff**2, xs, axis=1)
        return float(np.sqrt(np.trapz(intX, ys, axis=0)))

    def compute_sol_rel_l2(self, n: int = 60) -> float:
        xs, ys, _, _, XY = self._grid(n)
        XYt = torch.from_numpy(XY).to(device)
        u_true = u_exact(XYt, self.a1, self.a2).detach().cpu().numpy().reshape(n, n)
        with torch.no_grad():
            u_pred = self._u_hat(XYt).cpu().numpy().reshape(n, n)
        diff = u_pred - u_true
        num = np.trapz(np.trapz(diff**2, xs, axis=1), ys, axis=0)
        den = np.trapz(np.trapz(u_true**2, xs, axis=1), ys, axis=0)
        return float(np.sqrt(num) / (np.sqrt(den) + self.rel_err_eps))

    # ---- low-level step helpers (used by both the main loop and the
    #      Phase B trial scan in the boxcox/phase_ab schedule) ----
    def _adam_step(self, xy_train: torch.Tensor) -> tuple[float, float]:
        self.adam.zero_grad()
        J_obj, J_raw = self.compute_loss(xy_train, create_graph_second=True)
        J_obj.backward()
        self.adam.step()
        return float(J_obj.item()), float(J_raw.item())

    def _qn_step(self, xy_train: torch.Tensor) -> tuple[float, float]:
        holder: dict = {}

        def closure():
            self.quasi_newton.zero_grad()
            J_obj_c, J_raw_c = self.compute_loss(xy_train, create_graph_second=True)
            holder["J_raw"] = J_raw_c
            J_obj_c.backward()
            return J_obj_c

        def loss_eval():
            J_obj_e, _ = self.compute_loss(xy_train, create_graph_second=False)
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
    #      identity for every candidate so that the comparison is fair
    #      under the changed objective geometry. ----
    def _phase_b_trial_block(
        self,
        xy_train: torch.Tensor,
        xy_val: torch.Tensor,
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
                J_obj_v, J_raw_v = self._qn_step(xy_train)
                with torch.set_grad_enabled(True):
                    val_obj, val_raw = self.compute_loss(
                        xy_val, create_graph_second=False
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

    def train(
        self,
        n_epochs: int = 20000,
        n_collocation: int = 8000,
        train_split: float = 0.8,
        resample_every: int = 500,
        adam_epochs: int = 2000,
        verbose_freq: int = 500,
        diag_grid_n: int = 60,
        # Early stopping is now QN-phase-only; an Adam plateau cannot
        # terminate the run before the curvature-aware optimiser engages.
        # See pinn_poisson_2d_unitsquare.PoissonPINN.train() for the same
        # design rationale.
        patience: int = 500,
        min_delta: float = 1e-10,
        moving_avg_window: int = 20,
        # QN-phase early stopping (urban-style relative-MA criterion), used on
        # the standard (non-schedule) path. Active only after handover.
        early_stop: bool = True,
        es_patience: int = 300,
        es_window: int = 20,
        es_min_delta: float = 1e-4,
        es_stop_loss: float = 0.0,
        # Adaptive Adam -> QN handover. With "plateau" the run switches to
        # the quasi-Newton optimiser the first time the validation J fails
        # to improve by `plateau_min_delta` over `plateau_patience`
        # consecutive Adam epochs (or `handover_max_adam_epochs`, whichever
        # comes first). "fixed" recovers the legacy schedule (switch at
        # exactly `adam_epochs`); the loss-schedule machinery is only
        # supported in that mode.
        handover_strategy: str = "fixed",
        handover_max_adam_epochs: int = 10000,
        plateau_patience: int = 200,
        plateau_min_delta: float = 1e-4,
        loss_threshold: float = 1.0,
        gradnorm_threshold: float = 1e-3,
        scheduler_patience: int = 500,
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
            "\nTraining 2DH PINN: ∇^2 u + k^2 u - q(x, y) = 0  "
            "(Urban et al. 2025, sec. 5)"
        )
        print(f"  Domain:          x in [{x_min}, {x_max}], y in [{y_min}, {y_max}]")
        print(
            f"  Wavenumbers:     a1={self.a1}, a2={self.a2}, k={self.k:g}"
        )
        if self.loss_transform == "boxcox":
            print(
                f"  Loss transform:  {self.loss_transform}  "
                f"(lambda={self.loss_lambda:g}, eps={self.loss_eps:g})"
            )
        else:
            print(f"  Loss transform:  {self.loss_transform}")
        qn_name = self.quasi_newton.param_groups[0]["variant"].upper()
        if handover_strategy == "fixed":
            qn_iters = n_epochs - adam_epochs
            if qn_iters > 0:
                print(
                    f"  Optimizers:      Adam ({adam_epochs} iters) then {qn_name} "
                    f"({qn_iters} iters)  [fixed handover]"
                )
            else:
                print(f"  Optimizers:      Adam ({adam_epochs} iters)  [pure Adam]")
        else:
            print(
                f"  Optimizers:      Adam (handover={handover_strategy}, cap "
                f"{handover_max_adam_epochs}) -> {qn_name} (until "
                f"--patience {patience} of stagnation or epoch {n_epochs})"
            )
        print("-" * 80)

        valid_strategies = ("fixed", "plateau", "loss_threshold", "gradnorm")
        if handover_strategy not in valid_strategies:
            raise ValueError(
                f"handover_strategy must be one of {valid_strategies}, "
                f"got {handover_strategy!r}"
            )

        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be in (0, 1).")
        if n_collocation < 2:
            raise ValueError("n_collocation must be >= 2.")
        if resample_every < 1:
            raise ValueError("resample_every must be >= 1.")
        if adam_epochs < 0 or adam_epochs > n_epochs:
            raise ValueError("adam_epochs must be in [0, n_epochs].")

        schedule_on = loss_lambda_schedule == "phase_ab"
        if schedule_on:
            if handover_strategy != "fixed":
                # The phase A/B trial-block scan keys off the boundary
                # `epoch == adam_epochs + 1`; mixing it with adaptive
                # handover would make that boundary undefined.
                raise ValueError(
                    "loss_lambda_schedule='phase_ab' requires "
                    "handover_strategy='fixed'."
                )
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
            x = torch.empty(n_collocation, 1, device=device).uniform_(x_min, x_max)
            y = torch.empty(n_collocation, 1, device=device).uniform_(y_min, y_max)
            xy = torch.cat([x, y], dim=1)
            perm = torch.randperm(n_collocation, device=device)
            xy = xy[perm]
            return xy[:n_train].detach().clone(), xy[n_train:].detach().clone()

        xy_train, xy_val = resample_block()

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

        # Adam-phase plateau detector for the handover trigger. Tracks the
        # raw validation J (not the MA — we want handover to fire eagerly)
        # and is independent from the early-stop counter above, which is
        # only consulted after handover.
        plateau_best = float("inf")
        plateau_no_improve = 0
        handover_done = False
        self.handover_epoch: int | None = None

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
                xy_train, xy_val = resample_block()

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
            phase_b_idx = epoch - adam_epochs - 1
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
                    xy_train,
                    xy_val,
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
                # The next K - 1 main-loop iterations are absorbed via skip_remaining.
                skip_remaining = lambda_block_size - 1
                continue

            # Handover decision: in fixed mode we mimic the legacy
            # `epoch <= adam_epochs` schedule; in plateau / threshold /
            # gradnorm modes we keep going on Adam until the trigger fires.
            use_adam = not handover_done
            opt = self.adam if use_adam else self.quasi_newton
            sch = sch_adam if use_adam else sch_qn
            grad_norm: float | None = None

            if use_adam:
                opt.zero_grad()
                J_obj, J_raw = self.compute_loss(xy_train, create_graph_second=True)
                J_obj.backward()
                if handover_strategy == "gradnorm":
                    with torch.no_grad():
                        sq = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                sq += float((p.grad ** 2).sum().item())
                        grad_norm = float(sq ** 0.5)
                opt.step()
            else:
                holder: dict = {}

                def closure():
                    opt.zero_grad()
                    J_obj_c, J_raw_c = self.compute_loss(
                        xy_train, create_graph_second=True
                    )
                    holder["J_raw"] = J_raw_c
                    J_obj_c.backward()
                    return J_obj_c

                def loss_eval():
                    J_obj_e, _ = self.compute_loss(
                        xy_train, create_graph_second=False
                    )
                    return J_obj_e

                J_obj = opt.step(closure, loss_eval)
                J_raw = holder["J_raw"]

            with torch.set_grad_enabled(True):
                val_obj, val_raw = self.compute_loss(xy_val, create_graph_second=False)

            self.obj_train.append(float(J_obj.item()))
            self.obj_val.append(float(val_obj.item()))
            self.J_train.append(float(J_raw.item()))
            self.J_val.append(float(val_raw.item()))

            # ---------------------------------------------------------------
            # Adam phase: track plateau on raw val J for the handover
            # trigger. The early-stop MA counter is intentionally NOT
            # advanced here; an Adam plateau must hand over, not terminate.
            # ---------------------------------------------------------------
            if use_adam:
                v_raw = float(val_raw.item())
                if v_raw + plateau_min_delta < plateau_best:
                    plateau_best = v_raw
                    plateau_no_improve = 0
                else:
                    plateau_no_improve += 1

                handover_now = False
                if epoch >= handover_max_adam_epochs:
                    handover_now = True
                elif handover_strategy == "fixed":
                    handover_now = epoch >= adam_epochs
                elif handover_strategy == "plateau":
                    handover_now = plateau_no_improve >= plateau_patience
                elif handover_strategy == "loss_threshold":
                    handover_now = v_raw < loss_threshold
                elif handover_strategy == "gradnorm":
                    handover_now = (
                        grad_norm is not None and grad_norm < gradnorm_threshold
                    )

                if handover_now:
                    handover_done = True
                    self.handover_epoch = epoch
                    # Reset best-state and MA so QN-phase improvement over
                    # the (much higher) Adam-phase floor counts as new
                    # progress; otherwise the first QN step blows past
                    # best_val_ma and the counter never ticks.
                    self.best_val_ma = float("inf")
                    self.best_state = None
                    ma_buf = []
                    epochs_no_improve = 0
                    es_hist.clear()
                    es_best_ma = float("inf")
                    es_bad = 0
                    print(
                        f"  [handover] epoch {epoch}: Adam -> "
                        f"{self.quasi_newton.param_groups[0]['variant'].upper()} "
                        f"(strategy={handover_strategy}, "
                        f"plateau_no_improve={plateau_no_improve}, "
                        f"val_J={v_raw:.3e})"
                    )
            # ---------------------------------------------------------------
            # QN phase: MA-based early stopping.
            # ---------------------------------------------------------------
            else:
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

                # Urban-style relative-MA early-stop counter (raw val J).
                if early_stop:
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

            # QN-phase early stopping. The urban-style relative-MA criterion is
            # the primary trigger on the standard path; the legacy absolute-MA
            # counter remains as a backstop (and covers the phase_ab schedule
            # path, where the urban counter is not advanced).
            if es_stopped_at is not None:
                n_qn = epoch - (self.handover_epoch or adam_epochs)
                print(
                    f"  [QN early stop] epoch {epoch} ({n_qn} QN steps): "
                    f"{es_reason}"
                )
                break
            if handover_done and epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(QN-phase val-MA has not improved for {patience} epochs)."
                )
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        print("-" * 80)
        print(f"Done. Best val objective moving average: {self.best_val_ma:.6e}")

    def plot_results(
        self, n: int = 80, save_path: str | None = None, dpi: int = 150
    ) -> None:
        xs, ys, XX, YY, XY = self._grid(n)
        XYt = torch.from_numpy(XY).to(device)

        u_true = u_exact(XYt, self.a1, self.a2).detach().cpu().numpy().reshape(n, n)
        with torch.no_grad():
            u_pred = self._u_hat(XYt).cpu().numpy().reshape(n, n)
        abs_err = np.abs(u_pred - u_true)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        im0 = axes[0, 0].contourf(XX, YY, u_true, levels=30, cmap="viridis")
        fig.colorbar(im0, ax=axes[0, 0])
        axes[0, 0].set_title(r"$\phi_{\mathrm{exact}}(x, y)$")

        im1 = axes[0, 1].contourf(XX, YY, u_pred, levels=30, cmap="viridis")
        fig.colorbar(im1, ax=axes[0, 1])
        axes[0, 1].set_title(r"$\phi_{\mathrm{PINN}}(x, y)$")

        if self.obj_train:
            epochs_arr = np.arange(1, len(self.obj_train) + 1)
            axes[1, 0].semilogy(epochs_arr, self.obj_train, label="obj(train)")
            axes[1, 0].semilogy(epochs_arr, self.obj_val, label="obj(val)")
            axes[1, 0].semilogy(epochs_arr, self.J_train, "--", label="J(train)")
            axes[1, 0].semilogy(epochs_arr, self.J_val, "--", label="J(val)")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_title("Loss curves")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        im3 = axes[1, 1].contourf(XX, YY, abs_err, levels=30, cmap="magma")
        fig.colorbar(im3, ax=axes[1, 1])
        axes[1, 1].set_title(r"$|\phi_{\mathrm{PINN}} - \phi_{\mathrm{exact}}|$")

        for i, j in [(0, 0), (0, 1), (1, 1)]:
            axes[i, j].set_xlabel("x")
            axes[i, j].set_ylabel("y")

        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure to: {save_path}")
        plt.close(fig)

    def save(self, path: str = "../models/pinn_helmholtz.pth") -> None:
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
            fields.npz     — final phi_exact, phi_pred, abs_err on the (x, y) grid
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

        xs, ys, XX, YY, XY = self._grid(n_eval)
        XYt = torch.from_numpy(XY).to(device)
        u_true = u_exact(XYt, self.a1, self.a2).detach().cpu().numpy().reshape(n_eval, n_eval)
        with torch.no_grad():
            u_pred = self._u_hat(XYt).cpu().numpy().reshape(n_eval, n_eval)
        abs_err = np.abs(u_pred - u_true)

        fields_path = os.path.join(run_dir, "fields.npz")
        np.savez(
            fields_path,
            x=xs.astype(np.float64),
            y=ys.astype(np.float64),
            phi_exact=u_true.astype(np.float64),
            phi_pred=u_pred.astype(np.float64),
            abs_err=abs_err.astype(np.float64),
        )
        print(f"Saved field snapshots to: {fields_path}")

        summary = {
            "problem": "2DH (Urban et al. 2025, sec. 5)",
            "qn_variant": self.quasi_newton.param_groups[0]["variant"],
            "loss_transform": self.loss_transform,
            "loss_lambda": self.loss_lambda,
            "loss_eps": self.loss_eps,
            "lambda_pde": self.lambda_pde,
            "lambda_history": [
                [int(e), float(lam)] for e, lam in self.lambda_history
            ],
            "a1_wavenumber": self.a1,
            "a2_wavenumber": self.a2,
            "k_wavenumber": self.k,
            "domain": {
                "x": [x_min, x_max],
                "y": [y_min, y_max],
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
# MAIN — Urban et al. (2025), Table 4 2DH rows
# =============================================================================
# Two problem configurations from the paper (wavenumbers/budgets). The network
# is held at the project BVP standard 3x32 tanh for BOTH, overriding the paper's
# per-config nets (low: 2x20, high: 3x30), so all BVP benchmarks share one net.
#   "low":  (a1, a2) = (1, 4), k = 1 — 20 000 iter, 5 000 Adam, batch 10 000.
#   "high": (a1, a2) = (6, 6), k = 1 — 50 000 iter, 5 000 Adam, batch 10 000.
HELMHOLTZ_CONFIGS: dict[str, dict] = {
    "low": {
        "a1": 1,
        "a2": 4,
        "k": 1.0,
        "hidden_layers": (32, 32, 32),
        "n_epochs": 10000,
        "adam_epochs": 2000,
        "n_collocation": 10000,
    },
    "high": {
        "a1": 6,
        "a2": 6,
        "k": 1.0,
        "hidden_layers": (32, 32, 32),
        "n_epochs": 20000,
        "adam_epochs": 2000,
        "n_collocation": 10000,
    },
}


def main() -> None:
    # --- user knobs reproducing the paper's optimizer/loss sweeps ---
    config_name = "low"          # "low" | "high"
    qn_variant = "ssbroyden"     # "bfgs", "ssbfgs", "ssbroyden"
    loss_transform = "identity"  # "identity", "sqrt", "log", "boxcox"
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
    qn_H_on_cpu = False
    # ---------------------------------------------------------------

    cfg = HELMHOLTZ_CONFIGS[config_name]

    model = NeuralNetwork(hidden_layers=cfg["hidden_layers"], activation=nn.Tanh())
    print("\nNeural Network Architecture:\n")
    print(model, "\n")

    pinn = PINN_Helmholtz_Solver(
        model=model,
        lr=1e-3,
        lambda_pde=1.0,
        a1=cfg["a1"],
        a2=cfg["a2"],
        k=cfg["k"],
        loss_transform=loss_transform,
        loss_lambda=loss_lambda,
        qn_variant=qn_variant,
        qn_H_on_cpu=qn_H_on_cpu,
    )

    pinn.train(
        n_epochs=cfg["n_epochs"],
        n_collocation=cfg["n_collocation"],
        train_split=0.8,
        resample_every=500,
        adam_epochs=cfg["adam_epochs"],
        verbose_freq=500,
        diag_grid_n=60,
        patience=cfg["n_epochs"],   # disable early stop — match paper's fixed budget
        min_delta=1e-10,
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
    run_name = (
        f"helmholtz_{config_name}_{qn_variant}_{transform_tag}_{run_tag}"
    )
    run_dir = os.path.join("..", "results", run_name)
    os.makedirs(run_dir, exist_ok=True)

    pinn.plot_results(n=80, save_path=os.path.join(run_dir, "results.png"))
    pinn.save(
        f"../models/pinn_helmholtz_{config_name}_{qn_variant}_{transform_tag}.pth"
    )
    pinn.save_results(
        run_dir,
        n_eval=80,
        extra_metadata={"run_name": run_name, "run_tag": run_tag},
    )
    print(f"\nAll run artefacts written to: {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
