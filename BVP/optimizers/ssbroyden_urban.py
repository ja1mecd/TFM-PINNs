"""Urban-style self-scaled quasi-Newton optimiser (PyTorch port).

This is a faithful port of Jorge Urbán's modified-SciPy optimiser
(https://github.com/jorgeurban/self_scaled_algorithms_pinns,
``modified_optimize.py``) into PyTorch, suitable for running on GPU with
float64 weights. It is intentionally kept SEPARATE from
``BVP/optimizers/ssbroyden.py`` so the two implementations can be compared
side by side.

Key differences vs. ``ssbroyden.py``
-----------------------------------
1. **Strong Wolfe line search** (Armijo + curvature condition) instead of
   pure Armijo backtracking. This is what the BFGS family theoretically
   requires to keep H positive-definite.
2. **All seven update variants from Urbán et al. (2025) JCP**:
       ``BFGS``        - efficient BFGS form (eq. (10) with tau=phi=1)
       ``BFGS_scipy``  - SciPy's two-matrix BFGS form (slower but parity)
       ``SSBFGS_OL``   - Self-scaled BFGS, tau_k = 1/b_k (Oren-Luenberger)
       ``SSBFGS_AB``   - tau_k = min(1, 1/b_k) (Al-Baali / SS-BFGS)
       ``SSBroyden1``  - eqs. (13)-(23), saddle-avoiding theta branch
       ``SSBroyden2``  - eqs. (13)-(23), midpoint theta branch (paper default)
       ``SSBroyden3``  - eqs. (13)-(23), tightest theta branch
3. **No upper clamp on tau_k** (Urban lets tau_k > 1 when self-scaling
   demands H be scaled up).
4. **abs(a_k) inside sqrt** so the SSBroyden formulas don't silently
   degenerate to plain BFGS when a_k is small or negative.
5. **initial_scale option**: when H_k = I (first step or after reset),
   switch to a separate scaling branch matching Urbán's
   ``initial_scale and np.allclose(Hk, I)`` block.

API
---
The optimiser takes a ``loss_and_grad(params_vec) -> (loss_scalar_tensor,
grad_vec_tensor)`` callable. This mirrors SciPy's ``minimize(..., jac=True)``
contract and is what Wolfe needs (the curvature check requires gradient
evaluations at trial points). Convert your PyTorch closure with the helper
below::

    optimiser = SSBroydenUrbanOptimizer(
        model.parameters(), variant="SSBroyden2"
    )

    def loss_and_grad(x_vec):
        # set parameters
        offset = 0
        for p in model.parameters():
            n = p.numel()
            with torch.no_grad():
                p.copy_(x_vec[offset:offset+n].view_as(p))
            offset += n
        # forward / backward
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss = compute_loss(...)
        loss.backward()
        # collect gradient
        g = torch.cat([p.grad.detach().view(-1) for p in model.parameters()])
        return loss.detach(), g

    new_loss = optimiser.step(loss_and_grad)

WARNINGS
--------
* O(n^2) memory and time per step in the number of trainable parameters.
  Use on small networks (< ~10k params), or pass ``H_on_cpu=True`` to keep
  the dense Hessian on the host.
* Use ``dtype=torch.float64`` for parity with Urbán's NumPy/SciPy run.
"""

from __future__ import annotations

from typing import Callable

import math
import torch
import torch.optim as optim


_VARIANTS = (
    "BFGS",
    "BFGS_scipy",
    "SSBFGS_OL",
    "SSBFGS_AB",
    "SSBroyden1",
    "SSBroyden2",
    "SSBroyden3",
)


# =====================================================================
# Strong Wolfe line search (Nocedal & Wright Alg. 3.5 + 3.6 / "zoom")
# =====================================================================
def _cubic_interpolate(
    x1: float,
    f1: float,
    g1: float,
    x2: float,
    f2: float,
    g2: float,
    bounds: tuple[float, float] | None = None,
) -> float:
    """Cubic interpolation minimum between (x1, f1, g1) and (x2, f2, g2).

    Reproduces the helper used in ``torch.optim.LBFGS._strong_wolfe`` and
    Nocedal & Wright (Numerical Optimization, 2nd ed., p.~58).
    """
    if bounds is not None:
        xmin, xmax = bounds
    else:
        xmin = min(x1, x2)
        xmax = max(x1, x2)

    d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2)
    d2_squared = d1 * d1 - g1 * g2
    if d2_squared >= 0.0:
        d2 = math.sqrt(d2_squared)
        if x1 <= x2:
            sign = 1.0
        else:
            sign = -1.0
        denom = g2 - g1 + 2.0 * sign * d2
        if denom == 0.0:
            return 0.5 * (xmin + xmax)
        candidate = x2 - (x2 - x1) * (g2 + sign * d2 - d1) / denom
        return min(max(candidate, xmin), xmax)
    return 0.5 * (xmin + xmax)


def strong_wolfe_line_search(
    loss_and_grad: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    x0: torch.Tensor,
    p: torch.Tensor,
    f0: float,
    g0: torch.Tensor,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    amin: float = 1e-25,
    amax: float = 1e25,
    max_ls: int = 25,
    tolerance_change: float = 1e-9,
) -> tuple[float, float, torch.Tensor, int]:
    """Strong Wolfe line search using bracket + cubic-interpolated zoom.

    Returns ``(alpha, f_alpha, grad_alpha, n_evals)``. If the bracketing
    phase exhausts ``max_ls`` evaluations, returns the best alpha found so
    far (Armijo-feasible if any) with ``alpha > 0`` and the corresponding
    function/gradient values; if no Armijo-feasible point was found,
    returns ``alpha = 0``.

    Conventions match Nocedal & Wright Algorithms 3.5/3.6: bracketing of
    [a_lo, a_hi] is asymmetric (a_lo always has the lowest f among the
    candidates seen so far, no ordering assumed between a_lo and a_hi).
    """
    gTp0 = float(torch.dot(g0, p).item())
    if gTp0 >= 0.0:
        # Not a descent direction: caller must reset H.
        return 0.0, f0, g0.clone(), 0

    alpha_prev = 0.0
    f_prev = f0
    gTp_prev = gTp0

    alpha = float(alpha0)

    n_evals = 0
    best_alpha = 0.0
    best_f = f0
    best_g = g0.clone()

    for i in range(max_ls):
        if alpha < amin or alpha > amax:
            break

        x_try = x0 + alpha * p
        f_try_t, g_try = loss_and_grad(x_try)
        f_try = float(f_try_t.item())
        n_evals += 1

        if not math.isfinite(f_try):
            # Step too aggressive — contract and continue bracketing.
            alpha *= 0.5
            continue

        # Track best Armijo-feasible candidate as a fallback.
        if f_try <= f0 + c1 * alpha * gTp0 and f_try < best_f:
            best_alpha = alpha
            best_f = f_try
            best_g = g_try.clone()

        # Test 1: Armijo violated, OR not first iter and f increased.
        if f_try > f0 + c1 * alpha * gTp0 or (i > 0 and f_try >= f_prev):
            return _zoom(
                loss_and_grad,
                x0,
                p,
                f0,
                gTp0,
                alpha_prev,
                f_prev,
                gTp_prev,
                alpha,
                f_try,
                float(torch.dot(g_try, p).item()),
                c1,
                c2,
                max_ls - i,
                tolerance_change,
                n_evals_so_far=n_evals,
                best_so_far=(best_alpha, best_f, best_g),
            )

        gTp_try = float(torch.dot(g_try, p).item())

        # Test 2: strong Wolfe curvature satisfied.
        if abs(gTp_try) <= -c2 * gTp0:
            return alpha, f_try, g_try, n_evals

        # Test 3: directional derivative changed sign — bracket reverses.
        if gTp_try >= 0.0:
            return _zoom(
                loss_and_grad,
                x0,
                p,
                f0,
                gTp0,
                alpha,
                f_try,
                gTp_try,
                alpha_prev,
                f_prev,
                gTp_prev,
                c1,
                c2,
                max_ls - i,
                tolerance_change,
                n_evals_so_far=n_evals,
                best_so_far=(best_alpha, best_f, best_g),
            )

        alpha_prev = alpha
        f_prev = f_try
        gTp_prev = gTp_try
        alpha = min(2.0 * alpha, amax)

    # Bracketing failed (e.g. monotone descent up to amax). Return the best
    # Armijo-feasible alpha seen so far (or zero if none).
    return best_alpha, best_f, best_g, n_evals


def _zoom(
    loss_and_grad: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    x0: torch.Tensor,
    p: torch.Tensor,
    f0: float,
    gTp0: float,
    a_lo: float,
    f_lo: float,
    gTp_lo: float,
    a_hi: float,
    f_hi: float,
    gTp_hi: float,
    c1: float,
    c2: float,
    max_ls: int,
    tolerance_change: float,
    n_evals_so_far: int,
    best_so_far: tuple[float, float, torch.Tensor],
) -> tuple[float, float, torch.Tensor, int]:
    """Zoom phase of strong Wolfe line search."""
    n_evals = n_evals_so_far
    best_alpha, best_f, best_g = best_so_far

    for _ in range(max(max_ls, 1)):
        if abs(a_hi - a_lo) < tolerance_change:
            break

        alpha = _cubic_interpolate(
            a_lo,
            f_lo,
            gTp_lo,
            a_hi,
            f_hi,
            gTp_hi,
            bounds=(min(a_lo, a_hi), max(a_lo, a_hi)),
        )

        # Safeguard: alpha must be strictly between the bracket endpoints.
        lo, hi = sorted([a_lo, a_hi])
        margin = 0.1 * (hi - lo)
        if margin > 0.0 and (alpha < lo + margin or alpha > hi - margin):
            alpha = 0.5 * (a_lo + a_hi)

        x_try = x0 + alpha * p
        f_try_t, g_try = loss_and_grad(x_try)
        f_try = float(f_try_t.item())
        n_evals += 1

        if not math.isfinite(f_try):
            a_hi = alpha
            f_hi = f_try
            gTp_hi = 0.0
            continue

        if f_try <= f0 + c1 * alpha * gTp0 and f_try < best_f:
            best_alpha = alpha
            best_f = f_try
            best_g = g_try.clone()

        if f_try > f0 + c1 * alpha * gTp0 or f_try >= f_lo:
            a_hi = alpha
            f_hi = f_try
            gTp_hi = float(torch.dot(g_try, p).item())
        else:
            gTp_try = float(torch.dot(g_try, p).item())
            if abs(gTp_try) <= -c2 * gTp0:
                return alpha, f_try, g_try, n_evals
            if gTp_try * (a_hi - a_lo) >= 0.0:
                a_hi = a_lo
                f_hi = f_lo
                gTp_hi = gTp_lo
            a_lo = alpha
            f_lo = f_try
            gTp_lo = gTp_try

    return best_alpha, best_f, best_g, n_evals


# =====================================================================
# Quasi-Newton optimiser
# =====================================================================
class SSBroydenUrbanOptimizer(optim.Optimizer):
    """Dense self-scaled (Broyden-family) quasi-Newton optimiser, Urban-style.

    Parameters
    ----------
    params : iterable
        Parameters to optimise.
    variant : {"BFGS","BFGS_scipy","SSBFGS_OL","SSBFGS_AB",
               "SSBroyden1","SSBroyden2","SSBroyden3"}, default "SSBroyden2"
        Hessian-update formula. SSBroyden2 matches the SSBroyden update used
        as the paper's headline result.
    c1, c2 : float, default 1e-4 / 0.9
        Strong-Wolfe constants. ``0 < c1 < c2 < 1``.
    amin, amax : float, default 1e-25 / 1e25
        Step-length bounds for the line search bracket.
    max_ls : int, default 25
        Maximum total function/gradient evaluations per ``step``.
    initial_scale : bool, default False
        If True and ``H == I`` (first step or after a reset), apply Urbán's
        initial-scaling branch matching ``initial_scale=True`` in the
        SciPy port.
    tau_min, tau_max : float | None, default (1e-12, None)
        Optional lower/upper clamps on tau_k. ``None`` means no clamp. The
        Urban code does not clamp tau_k (set both to ``None`` for parity).
    damping : float, default 1e-30
        Floor used to guard divisions; Urban relies on raw float64 (no
        explicit damping). Keep tiny.
    dtype : torch.dtype, default torch.float64
        Dtype used for the dense Hessian. Float64 is recommended for parity
        with Urbán's NumPy run, even when the network weights are float32.
    H_device : torch.device | str | None
        Device for the dense Hessian. Defaults to the first parameter's
        device. Pass ``"cpu"`` to keep H off-GPU when running large nets.
    """

    def __init__(
        self,
        params,
        variant: str = "SSBroyden2",
        c1: float = 1e-4,
        c2: float = 0.9,
        amin: float = 1e-25,
        amax: float = 1e25,
        max_ls: int = 25,
        initial_scale: bool = False,
        tau_min: float | None = 1e-12,
        tau_max: float | None = None,
        damping: float = 1e-30,
        dtype: torch.dtype = torch.float64,
        H_device: torch.device | str | None = None,
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {_VARIANTS}, got {variant!r}"
            )
        if not (0.0 < c1 < c2 < 1.0):
            raise ValueError(
                f"Wolfe constants must satisfy 0 < c1 < c2 < 1; "
                f"got c1={c1}, c2={c2}"
            )
        defaults = dict(
            variant=variant,
            c1=c1,
            c2=c2,
            amin=amin,
            amax=amax,
            max_ls=max_ls,
            initial_scale=initial_scale,
            tau_min=tau_min,
            tau_max=tau_max,
            damping=damping,
            dtype=dtype,
        )
        super().__init__(params, defaults)
        if len(self.param_groups) > 1:
            raise ValueError(
                "SSBroydenUrbanOptimizer expects a single parameter group."
            )

        self._H: torch.Tensor | None = None
        self._H_device = H_device
        self._H_dtype = dtype
        self._n_steps = 0
        self._n_resets = 0
        self._n_ls_failures = 0
        self._n_func_evals = 0
        self._last_alpha = float("nan")
        self._last_tau = float("nan")
        self._last_phi = float("nan")
        self._last_gnorm = float("nan")

    # ----- parameter / gradient flatten helpers -----
    def _params(self):
        return self.param_groups[0]["params"]

    def _flat_param_count(self) -> int:
        return sum(p.numel() for p in self._params())

    def _gather_params(self) -> torch.Tensor:
        return torch.cat(
            [p.detach().view(-1).to(self._H_dtype) for p in self._params()]
        )

    @torch.no_grad()
    def _scatter_params(self, x_vec: torch.Tensor) -> None:
        offset = 0
        for p in self._params():
            n = p.numel()
            chunk = x_vec[offset : offset + n].to(p.dtype).view_as(p)
            p.copy_(chunk)
            offset += n

    # ----- public diagnostics -----
    @property
    def H(self) -> torch.Tensor | None:
        return self._H

    @H.setter
    def H(self, value: torch.Tensor | None) -> None:
        self._H = value

    def diagnostics(self) -> dict:
        return {
            "n_steps": self._n_steps,
            "n_resets": self._n_resets,
            "n_ls_failures": self._n_ls_failures,
            "n_func_evals": self._n_func_evals,
            "last_alpha": self._last_alpha,
            "last_tau": self._last_tau,
            "last_phi": self._last_phi,
            "last_gnorm": self._last_gnorm,
        }

    # ----- Hessian setup / reset -----
    @torch.no_grad()
    def reset_H(self, n: int | None = None, ref: torch.Tensor | None = None) -> None:
        if n is None:
            n = self._flat_param_count()
        if self._H_device is not None:
            dev = torch.device(self._H_device)
        elif ref is not None:
            dev = ref.device
        else:
            dev = next(self._params()).device
        self._H = torch.eye(n, device=dev, dtype=self._H_dtype)
        self._n_resets += 1

    @torch.no_grad()
    def warm_start_from(self, H_warm: torch.Tensor, *, cholesky_check: bool = True) -> bool:
        """Warm-start with a user-supplied dense Hessian.

        Symmetrises and (optionally) Cholesky-checks for positive
        definiteness. If the check fails, falls back to identity. Returns
        ``True`` when the warm start was accepted, ``False`` if it was
        rejected and the Hessian was reset to identity.
        """
        n = self._flat_param_count()
        if H_warm.shape != (n, n):
            raise ValueError(
                f"H_warm has shape {tuple(H_warm.shape)}, expected ({n}, {n})"
            )
        H_sym = 0.5 * (H_warm + H_warm.t())
        if cholesky_check:
            try:
                torch.linalg.cholesky(H_sym.to(torch.float64))
            except RuntimeError:
                self.reset_H(n=n, ref=H_warm)
                return False
        self._H = H_sym.to(dtype=self._H_dtype)
        return True

    # ----- main step -----
    def step(  # type: ignore[override]
        self,
        loss_and_grad: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Take one quasi-Newton step.

        ``loss_and_grad(x_vec) -> (loss_tensor, grad_tensor)`` must:
            - copy ``x_vec`` into the model parameters,
            - run forward + backward,
            - return ``(loss_value, flat_grad_vector)`` (both detached).

        The optimiser leaves the model parameters at the accepted ``x``.
        Returns the loss scalar tensor at the new point.
        """
        group = self.param_groups[0]
        variant = group["variant"]
        c1 = group["c1"]
        c2 = group["c2"]
        amin = group["amin"]
        amax = group["amax"]
        max_ls = group["max_ls"]
        initial_scale = group["initial_scale"]
        tau_min = group["tau_min"]
        tau_max = group["tau_max"]
        damping = group["damping"]
        dtype = group["dtype"]

        # Current point, loss, gradient.
        x = self._gather_params()
        n = x.numel()
        loss_t, g = loss_and_grad(x)
        g = g.to(dtype)
        f0 = float(loss_t.item())
        self._last_gnorm = float(torch.linalg.vector_norm(g).item())

        if self._H is None or self._H.shape[0] != n:
            self.reset_H(n=n, ref=g)

        H = self._H
        assert H is not None
        # Make sure x and g live on H's device & dtype so the matmul is cheap.
        x_H = x.to(device=H.device, dtype=dtype)
        g_H = g.to(device=H.device, dtype=dtype)

        # Search direction p = -H g (Urban: pk = -np.dot(Hk, gfk)).
        Hg = H.matmul(g_H)
        p_H = -Hg
        p = p_H.to(device=g.device, dtype=g.dtype)

        # ---- Strong Wolfe line search ----
        def f_and_g(x_vec):
            loss_v, grad_v = loss_and_grad(x_vec.to(g.device, dtype=g.dtype))
            return loss_v.detach().to(dtype), grad_v.detach().to(dtype)

        alpha, f_new, g_new, ne = strong_wolfe_line_search(
            f_and_g,
            x.to(dtype),
            p.to(dtype),
            f0,
            g.to(dtype),
            alpha0=1.0,
            c1=c1,
            c2=c2,
            amin=amin,
            amax=amax,
            max_ls=max_ls,
        )
        self._n_func_evals += ne
        self._last_alpha = alpha

        if alpha == 0.0 or not math.isfinite(alpha):
            # Line search failed: restore original parameters, reset H, exit.
            self._scatter_params(x)
            self.reset_H(n=n, ref=g)
            self._n_ls_failures += 1
            self._n_steps += 1
            return loss_t

        # Commit new point.
        x_new = x.to(dtype) + alpha * p.to(dtype)
        self._scatter_params(x_new)

        # Curvature information.
        s = (x_new - x.to(dtype)).to(device=H.device, dtype=dtype)
        y = (g_new.to(device=H.device, dtype=dtype) - g_H)

        ys = float(torch.dot(y, s).item())
        if not math.isfinite(ys) or abs(ys) <= damping:
            # Curvature pair degenerate: Hessian update would not preserve
            # positive definiteness; reset and exit.
            self.reset_H(n=n, ref=g_new)
            self._n_steps += 1
            return torch.as_tensor(f_new, dtype=loss_t.dtype, device=loss_t.device)

        rhok = 1.0 / ys
        Hy = H.matmul(y)
        yHy = float(torch.dot(y, Hy).item())
        if not math.isfinite(yHy) or abs(yHy) <= damping:
            self.reset_H(n=n, ref=g_new)
            self._n_steps += 1
            return torch.as_tensor(f_new, dtype=loss_t.dtype, device=loss_t.device)

        # Helper: identity-check matches Urban's `np.allclose(Hk, np.eye(N))`.
        is_identity = (
            initial_scale
            and self._n_steps == 0  # cheap and reliable proxy
        )

        # Compute and apply the variant-specific update.
        H = self._update_H(
            H,
            s,
            y,
            Hy,
            ys,
            yHy,
            rhok,
            alpha,
            g_H,
            n,
            variant,
            damping,
            tau_min,
            tau_max,
            is_identity_initial=is_identity,
        )

        # Symmetrise to fight numerical drift over many steps.
        self._H = 0.5 * (H + H.t())
        self._n_steps += 1

        return torch.as_tensor(f_new, dtype=loss_t.dtype, device=loss_t.device)

    # ----- variant-specific Hessian update -----
    def _update_H(
        self,
        H: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
        Hy: torch.Tensor,
        ys: float,
        yHy: float,
        rhok: float,
        alpha: float,
        g: torch.Tensor,
        n: int,
        variant: str,
        damping: float,
        tau_min: float | None,
        tau_max: float | None,
        is_identity_initial: bool,
    ) -> torch.Tensor:
        # ``hk = (y H y) / (y s)``    Urban: ykHkyk*rhok
        hk = yHy * rhok
        # ``bk = -alpha * rhok * (s . g)``  with g being g_k (start point grad)
        sTg = float(torch.dot(s, g).item())
        bk = -alpha * rhok * sTg

        # Useful aliases (eq. 10 form): we need the rank-2 sk⊗sk and sk⊗yk pieces
        I_eye = None  # only allocated for "BFGS_scipy" branch

        if variant == "BFGS":
            if is_identity_initial:
                tauk = rhok * float(torch.dot(y, y).item())
                if tauk != 0.0:
                    H = H / tauk
                    Hy = H.matmul(y)
                    yHy = float(torch.dot(y, Hy).item())
                    hk = yHy * rhok
            self._last_tau = 1.0
            self._last_phi = 1.0
            return _bfgs_efficient_update(H, s, y, Hy, rhok, hk)

        if variant == "BFGS_scipy":
            if is_identity_initial:
                tauk = rhok * float(torch.dot(y, y).item())
                if tauk != 0.0:
                    H = H / tauk
            self._last_tau = 1.0
            self._last_phi = 1.0
            I_eye = torch.eye(n, device=H.device, dtype=H.dtype)
            A1 = I_eye - rhok * torch.outer(s, y)
            A2 = I_eye - rhok * torch.outer(y, s)
            return A1 @ H @ A2 + rhok * torch.outer(s, s)

        if variant == "SSBFGS_OL":
            if is_identity_initial:
                tauk = rhok * float(torch.dot(y, y).item())
            else:
                tauk = 1.0 / bk if abs(bk) > damping else 1.0
            tauk = _clamp(tauk, tau_min, tau_max)
            if tauk != 0.0:
                H = H / tauk
                Hy = H.matmul(y)
                yHy = float(torch.dot(y, Hy).item())
                hk = yHy * rhok
            self._last_tau = float(tauk)
            self._last_phi = 1.0
            return _bfgs_efficient_update(H, s, y, Hy, rhok, hk)

        if variant == "SSBFGS_AB":
            if is_identity_initial:
                tauk = rhok * float(torch.dot(y, y).item())
            else:
                tauk = min(1.0, 1.0 / bk) if abs(bk) > damping else 1.0
            tauk = _clamp(tauk, tau_min, tau_max)
            if tauk != 0.0:
                H = H / tauk
                Hy = H.matmul(y)
                yHy = float(torch.dot(y, Hy).item())
                hk = yHy * rhok
            self._last_tau = float(tauk)
            self._last_phi = 1.0
            return _bfgs_efficient_update(H, s, y, Hy, rhok, hk)

        # ---- SSBroyden family ----
        ak = bk * hk - 1.0

        # rho_minus = min(1, hk * (1 - sqrt(|ak|/(1+ak))))
        denom_root = 1.0 + ak
        ratio = abs(ak) / denom_root if denom_root > damping else 0.0
        if ratio < 0.0:
            ratio = 0.0
        sqrt_ratio = math.sqrt(ratio)
        rhokm = min(1.0, hk * (1.0 - sqrt_ratio))
        rhokp = max(1.0, bk * (1.0 + sqrt_ratio))  # used only by SSBroyden1

        ak_safe = ak if abs(ak) > damping else math.copysign(damping, ak or 1.0)

        thetakm = (rhokm - 1.0) / ak_safe

        if variant == "SSBroyden1":
            # Branches by sign of (b_k - 1) and use Sherman-Morrison "SR1" theta.
            if is_identity_initial:
                # Urban (initial_scale and H==I): use thetakp = 1/rhokm; else thetakp = 1.
                thetakp = 1.0 / rhokm if abs(rhokm) > damping else 0.0
                thetakSR1 = 1.0 / (1.0 - bk) if abs(1.0 - bk) > damping else 0.0
                if abs(bk - 1.0) <= damping:
                    thetak = thetakm
                elif bk > 1.0:
                    thetak = max(thetakm, thetakSR1)
                else:
                    thetak = min(thetakp, thetakSR1)
                tauk = hk / (1.0 + ak * thetak) if abs(1.0 + ak * thetak) > damping else 1.0
            else:
                thetakp = rhokp
                thetakSR1 = 1.0 / (1.0 - bk) if abs(1.0 - bk) > damping else 0.0
                if abs(bk - 1.0) <= damping:
                    thetak = thetakm
                elif bk > 1.0:
                    thetak = max(thetakm, thetakSR1)
                else:
                    thetak = min(thetakp, thetakSR1)
                rhokk = min(1.0, 1.0 / bk) if abs(bk) > damping else 1.0
                sigmak = 1.0 + thetak * ak
                sigma_abs = abs(sigmak) if abs(sigmak) > damping else damping
                # |sigma|^(1/(1-N))
                if n != 1:
                    sigmaknm1 = sigma_abs ** (1.0 / (1.0 - n))
                else:
                    sigmaknm1 = 1.0
                if thetak <= 0.0:
                    tauk = min(rhokk * sigmaknm1, sigmak) if sigmak > 0.0 else rhokk * sigmaknm1
                else:
                    inv_th = 1.0 / thetak if abs(thetak) > damping else 1.0
                    tauk = rhokk * min(sigmaknm1, inv_th)
        elif variant == "SSBroyden2":
            if is_identity_initial:
                thetakp = 1.0 / rhokm if abs(rhokm) > damping else 0.0
                thetak = max(thetakm, min(thetakp, (1.0 - bk) / bk if abs(bk) > damping else 0.0))
                tauk = hk / (1.0 + ak * thetak) if abs(1.0 + ak * thetak) > damping else 1.0
            else:
                thetakp = 1.0 / rhokm if abs(rhokm) > damping else 0.0
                ratio_bk = (1.0 - bk) / bk if abs(bk) > damping else 0.0
                thetak = max(thetakm, min(thetakp, ratio_bk))
                rhokk = min(1.0, 1.0 / bk) if abs(bk) > damping else 1.0
                sigmak = 1.0 + thetak * ak
                sigma_abs = abs(sigmak) if abs(sigmak) > damping else damping
                if n != 1:
                    sigmaknm1 = sigma_abs ** (1.0 / (1.0 - n))
                else:
                    sigmaknm1 = 1.0
                if thetak <= 0.0:
                    tauk = min(rhokk * sigmaknm1, sigmak) if sigmak > 0.0 else rhokk * sigmaknm1
                else:
                    inv_th = 1.0 / thetak if abs(thetak) > damping else 1.0
                    tauk = rhokk * min(sigmaknm1, inv_th)
        else:  # SSBroyden3
            if is_identity_initial:
                thetak = thetakm
                tauk = hk / (1.0 + ak * thetak) if abs(1.0 + ak * thetak) > damping else 1.0
            else:
                thetak = thetakm
                rhokk = min(1.0, 1.0 / bk) if abs(bk) > damping else 1.0
                sigmak = 1.0 + thetak * ak
                sigma_abs = abs(sigmak) if abs(sigmak) > damping else damping
                if n != 1:
                    sigmaknm1 = sigma_abs ** (1.0 / (1.0 - n))
                else:
                    sigmaknm1 = 1.0
                if thetak <= 0.0:
                    tauk = min(rhokk * sigmaknm1, sigmak) if sigmak > 0.0 else rhokk * sigmaknm1
                else:
                    inv_th = 1.0 / thetak if abs(thetak) > damping else 1.0
                    tauk = rhokk * min(sigmaknm1, inv_th)

        if not math.isfinite(tauk) or abs(tauk) <= damping:
            tauk = 1.0
        tauk = _clamp(tauk, tau_min, tau_max)
        denom_phi = 1.0 + ak * thetak
        if abs(denom_phi) <= damping:
            phik = 0.0
        else:
            phik = (1.0 - thetak) / denom_phi
        self._last_tau = float(tauk)
        self._last_phi = float(phik)

        # vk = sk * rhok - Hkyk / ykHkyk
        if abs(yHy) <= damping:
            return H  # caller saw degenerate yHy already; safe-guard
        vk = s * rhok - Hy / yHy

        # Hk_new = (Hk - Hkyk⊗Hkyk / ykHkyk + phik * ykHkyk * vk⊗vk) / tauk
        #          + sk⊗sk * rhok
        H_term = H - torch.outer(Hy, Hy) / yHy + phik * yHy * torch.outer(vk, vk)
        H_new = H_term / tauk + rhok * torch.outer(s, s)
        return H_new


# =====================================================================
# Helpers
# =====================================================================
def _clamp(value: float, lo: float | None, hi: float | None) -> float:
    if lo is not None and value < lo:
        return lo
    if hi is not None and value > hi:
        return hi
    return value


def _bfgs_efficient_update(
    H: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    Hy: torch.Tensor,
    rhok: float,
    hk: float,
) -> torch.Tensor:
    """Urban's efficient BFGS form (eq. 10 with tau=phi=1)::

        H = H - rho * (Hy s^T + s Hy^T) + rho (1 + h) s s^T

    Mathematically equivalent to the SciPy two-matrix form but a few
    hundred times faster for n in the thousands.
    """
    return (
        H
        - rhok * (torch.outer(Hy, s) + torch.outer(s, Hy))
        + rhok * (1.0 + hk) * torch.outer(s, s)
    )
