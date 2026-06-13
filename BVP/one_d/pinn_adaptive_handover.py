"""
PINN with conditional Adam->QN handover.

The base `PINN_BVP_SSBroyden.train()` switches optimisers at a *fixed* epoch
(`adam_epochs`). For sweep experiments where Adam plateaus at very different
epochs across (lambda, seed) pairs, a fixed cutoff either wastes Adam budget
on already-stalled trajectories or hands a still-improving network over to
the curvature-aware phase too early. This subclass adds three condition-based
handover strategies:

    plateau         switch when val J_obj has not improved by min_delta over
                    the last `plateau_patience` epochs (using the same
                    moving-average buffer the base class already maintains).
                    This is the recommended default.

    loss_threshold  switch as soon as J_val drops below `loss_threshold`.
                    Useful for engaging SSBroyden inside the small-loss
                    regime where Box-Cox actually amplifies curvature.

    gradnorm        switch when the parameter-gradient L^2 norm drops below
                    `gradnorm_threshold`, i.e. when Adam is at its first-order
                    stationarity floor.

    fixed           legacy behaviour: switch at epoch `adam_epochs`.

In every case a safety cap `max_adam_epochs` forces the handover so a flat
landscape cannot trap Adam forever. Early stopping (the base-class logic
that monitors the validation MA against `patience`) is orthogonal to the
handover; it is *enabled by default* in this subclass with `patience=500`,
and can be disabled by passing `patience=n_epochs`.
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from pinn_ssbroyden_1d import PINN_BVP_SSBroyden, A, B  # noqa: E402


HANDOVER_STRATEGIES: tuple[str, ...] = (
    "fixed",
    "plateau",
    "loss_threshold",
    "gradnorm",
)


class PINN_BVP_AdaptiveHandover(PINN_BVP_SSBroyden):
    """Same loss/forward as the base class; replaces the fixed-epoch
    Adam->QN handover with a condition-based switch."""

    def train(  # type: ignore[override]
        self,
        n_epochs: int = 5000,
        n_collocation: int = 400,
        train_split: float = 0.8,
        resample_every: int = 500,
        adam_epochs: int = 2000,
        verbose_freq: int = 200,
        diag_grid_n: int = 400,
        # Early-stopping (base-class compatible).
        patience: int = 500,
        min_delta: float = 1e-10,
        moving_avg_window: int = 20,
        # Plateau scheduler for the lr.
        scheduler_patience: int = 300,
        scheduler_threshold: float = 1e-4,
        scheduler_gamma: float = 0.9,
        scheduler_min_lr: float = 1e-6,
        # ---- handover strategy (default 'fixed' at adam_epochs) ----
        handover_strategy: str = "fixed",
        handover_max_adam_epochs: int = 10000,
        plateau_patience: int = 200,
        plateau_min_delta: float = 1e-4,
        loss_threshold: float = 1.0,
        gradnorm_threshold: float = 1e-3,
        # ---- QN-phase early stopping (urban-style relative-MA criterion) ----
        # Mirrors pinn_ssbroyden_2d_urban.py: stop once the moving average of
        # the raw validation residual stops improving by `es_min_delta`
        # (relative) over `es_patience` epochs. Active ONLY after handover, so
        # the fixed Adam warm-up always runs to completion.
        early_stop: bool = True,
        es_patience: int = 300,
        es_window: int = 20,
        es_min_delta: float = 1e-4,
        es_stop_loss: float = 0.0,
    ) -> None:
        if handover_strategy not in HANDOVER_STRATEGIES:
            raise ValueError(
                f"handover_strategy must be one of {HANDOVER_STRATEGIES}, "
                f"got {handover_strategy!r}"
            )

        print(
            "\nTraining 1D PINN (adaptive handover): "
            f"-u'' = (k pi)^2 sin(k pi x), k = {self.k:g}, "
            f"domain [{A}, {B}]"
        )
        print(f"  Loss transform:    {self.loss_transform} (lambda={self.loss_lambda:g})")
        print(f"  Handover strategy: {handover_strategy}")
        if handover_strategy == "fixed":
            print(f"     adam_epochs = {adam_epochs}")
        elif handover_strategy == "plateau":
            print(
                f"     plateau_patience = {plateau_patience}, "
                f"plateau_min_delta = {plateau_min_delta}, "
                f"safety cap = {handover_max_adam_epochs}"
            )
        elif handover_strategy == "loss_threshold":
            print(
                f"     J_threshold = {loss_threshold}, "
                f"safety cap = {handover_max_adam_epochs}"
            )
        elif handover_strategy == "gradnorm":
            print(
                f"     gradnorm_threshold = {gradnorm_threshold}, "
                f"safety cap = {handover_max_adam_epochs}"
            )
        print(f"  Early stopping:    patience = {patience}, min_delta = {min_delta}")
        print(
            f"  QN variant:        "
            f"{self.quasi_newton.param_groups[0]['variant'].upper()}"
        )
        print("-" * 72)

        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be in (0, 1).")
        if n_collocation < 2:
            raise ValueError("n_collocation must be >= 2.")
        if resample_every < 1:
            raise ValueError("resample_every must be >= 1.")

        n_train = int(n_collocation * train_split)
        n_train = min(max(n_train, 1), n_collocation - 1)
        device = next(self.model.parameters()).device

        def resample_block():
            x = torch.empty(n_collocation, 1, device=device).uniform_(A, B)
            perm = torch.randperm(n_collocation, device=device)
            x = x[perm]
            return x[:n_train].detach().clone(), x[n_train:].detach().clone()

        x_train, x_val = resample_block()

        def make_plateau(opt):
            return optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=scheduler_gamma,
                patience=scheduler_patience,
                threshold=scheduler_threshold,
                min_lr=scheduler_min_lr,
            )

        sch_adam = make_plateau(self.adam)
        # The urban QN engine has no per-group 'lr', so a plateau scheduler on
        # it would KeyError; the strong-Wolfe search re-offers the unit step
        # every iteration anyway, so no QN scheduler is attached in that regime.
        sch_qn = None if self.qn_engine == "urban" else make_plateau(self.quasi_newton)

        # Plateau detector for handover (separate from the early-stop
        # detector below — different patience/min_delta).
        handover_done = False
        handover_epoch: Optional[int] = None
        plateau_best = float("inf")
        plateau_no_improve = 0

        # Best-weights tracker (restores the lowest val-MA iterate at the end).
        ma_buf: list[float] = []
        epochs_no_improve = 0
        last_pde = np.nan
        last_sol = np.nan
        last_rel = np.nan

        # QN-phase early-stop detector (urban-style relative-MA on raw val J).
        # Independent of the best-weights tracker above; only consulted once
        # `handover_done` is True.
        es_hist: "deque[float]" = deque(maxlen=es_window)
        es_best_ma = float("inf")
        es_bad = 0
        es_stopped_at: Optional[int] = None
        es_reason = ""

        def maybe_handover(epoch: int, val_J_raw: float, grad_norm: Optional[float]) -> bool:
            """Return True iff the QN phase should engage from this epoch on."""
            nonlocal plateau_best, plateau_no_improve

            if epoch >= handover_max_adam_epochs:
                return True
            if handover_strategy == "fixed":
                return epoch >= adam_epochs
            if handover_strategy == "loss_threshold":
                return val_J_raw < loss_threshold
            if handover_strategy == "gradnorm":
                return grad_norm is not None and grad_norm < gradnorm_threshold
            if handover_strategy == "plateau":
                if val_J_raw + plateau_min_delta < plateau_best:
                    plateau_best = val_J_raw
                    plateau_no_improve = 0
                else:
                    plateau_no_improve += 1
                return plateau_no_improve >= plateau_patience
            return False

        for epoch in range(1, n_epochs + 1):
            if epoch != 1 and ((epoch - 1) % resample_every == 0):
                x_train, x_val = resample_block()

            use_adam = not handover_done
            opt = self.adam if use_adam else self.quasi_newton
            sch = sch_adam if use_adam else sch_qn

            grad_norm: Optional[float] = None

            if use_adam:
                opt.zero_grad()
                J_obj, J_raw = self.compute_loss(x_train, create_graph_second=True)
                J_obj.backward()
                # Capture the gradient norm for the gradnorm-handover trigger.
                with torch.no_grad():
                    sq = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            sq += float((p.grad ** 2).sum().item())
                    grad_norm = float(sq ** 0.5)
                opt.step()
            elif self.qn_engine == "urban":
                holder: dict = {}

                def loss_and_grad(x_vec):
                    with torch.no_grad():
                        offset = 0
                        for p in self.model.parameters():
                            numel = p.numel()
                            p.copy_(x_vec[offset:offset + numel].to(p.dtype).view_as(p))
                            offset += numel
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    J_obj_c, J_raw_c = self.compute_loss(x_train, create_graph_second=True)
                    J_obj_c.backward()
                    grads = torch.cat([
                        (p.grad if p.grad is not None else torch.zeros_like(p)).detach().view(-1)
                        for p in self.model.parameters()
                    ])
                    holder["J_raw"] = J_raw_c
                    return J_obj_c.detach(), grads

                J_obj = opt.step(loss_and_grad)
                J_raw = holder["J_raw"]
            else:
                holder: dict = {}

                def closure():
                    opt.zero_grad()
                    J_obj_c, J_raw_c = self.compute_loss(
                        x_train, create_graph_second=True
                    )
                    holder["J_raw"] = J_raw_c
                    J_obj_c.backward()
                    return J_obj_c

                def loss_eval():
                    J_obj_e, _ = self.compute_loss(x_train, create_graph_second=False)
                    return J_obj_e

                J_obj = opt.step(closure, loss_eval)
                J_raw = holder["J_raw"]

            with torch.set_grad_enabled(True):
                val_obj, val_raw = self.compute_loss(
                    x_val, create_graph_second=False
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

            if sch is not None:
                sch.step(float(val_obj.item()))

            # Diagnostics are computed EVERY epoch (printing stays throttled by
            # verbose_freq below). The previous every-verbose_freq sampling
            # turned the solution-error history into a coarse staircase.
            last_pde = self.compute_pde_l2(n=diag_grid_n)
            last_sol = self.compute_sol_l2(n=diag_grid_n)
            last_rel = self.compute_sol_rel_l2(n=diag_grid_n)
            self.pde_l2.append(last_pde)
            self.sol_l2.append(last_sol)
            self.sol_rel_l2.append(last_rel)

            if epoch == 1 or (epoch % verbose_freq == 0):
                lr_now = opt.param_groups[0].get("lr", float("nan"))
                phase = "ADAM" if use_adam else self.quasi_newton.param_groups[0][
                    "variant"
                ].upper()
                print(
                    f"Epoch {epoch:6d} [{phase}] | "
                    f"obj={self.obj_train[-1]:.3e}, val={self.obj_val[-1]:.3e} | "
                    f"J={self.J_train[-1]:.3e}, valJ={self.J_val[-1]:.3e} | "
                    f"pdeL2={last_pde:.3e}, solL2={last_sol:.3e}, "
                    f"relL2={last_rel:.3e} | lr={lr_now:.2e}"
                )

            # Decide whether to switch optimisers AT THE END of this iteration,
            # so the next loop iteration begins under the QN regime.
            if use_adam and not handover_done:
                if maybe_handover(epoch, float(val_raw.item()), grad_norm):
                    handover_done = True
                    handover_epoch = epoch
                    print(
                        f"  [handover] epoch {epoch}: switching Adam -> "
                        f"{self.quasi_newton.param_groups[0]['variant'].upper()} "
                        f"(strategy={handover_strategy}, J_val={float(val_raw.item()):.3e}, "
                        f"|grad|={grad_norm if grad_norm is not None else float('nan'):.3e})"
                    )

            # QN-phase early stopping (urban-style relative-MA criterion). Gated
            # on `handover_done`, so it can never cut the fixed Adam warm-up
            # short; it only ends the quasi-Newton phase once it has converged.
            if early_stop and handover_done:
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
                if es_stopped_at is not None:
                    n_qn = epoch - (handover_epoch or adam_epochs)
                    print(
                        f"  [QN early stop] epoch {epoch} "
                        f"({n_qn} QN steps): {es_reason}"
                    )
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        print("-" * 72)
        print(
            f"Done. Best val objective MA: {self.best_val_ma:.6e}. "
            f"Handover @ epoch: {handover_epoch}. "
            f"QN early stop @ epoch: {es_stopped_at}"
        )
        # Surface the handover and early-stop epochs for downstream callers.
        self.handover_epoch = handover_epoch
        self.es_stopped_at = es_stopped_at
