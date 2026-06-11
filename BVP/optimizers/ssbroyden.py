"""
Quasi-Newton optimizer with three variants from Urbán et al. (2025, JCP):

    - variant="bfgs"       standard BFGS              (tau_k = 1, phi_k = 1)
    - variant="ssbfgs"     self-scaled BFGS           (tau_k from eq. (11), phi_k = 1)
    - variant="ssbroyden"  self-scaled Broyden        (tau_k, phi_k from eqs. (13)-(14))

The class keeps the dense inverse-Hessian approximation H_k and applies a
backtracking Armijo line search. Expected use:

    opt = SSBroydenOptimizer(model.parameters(), variant="ssbroyden")
    opt.step(closure, loss_eval)

`closure` performs zero_grad + forward + backward and returns the loss tensor.
`loss_eval` computes the loss only (no backward) — used during line search.

WARNING: O(n^2) memory/time in the number of trainable parameters.
         Use on small networks (< ~10k params) or set H_on_cpu=True.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

try:
    # Cubic-interpolating strong-Wolfe line search shipped with
    # torch.optim.LBFGS. Private but stable across torch 1.x-2.x (present in
    # the 1.13.1/py3.7 build on the training machine).
    from torch.optim.lbfgs import _strong_wolfe
except ImportError:  # pragma: no cover
    _strong_wolfe = None


class SSBroydenOptimizer(optim.Optimizer):
    """Dense self-scaled (Broyden-family) quasi-Newton optimizer.

    Parameters
    ----------
    params : iterable
        Parameters to optimize (as usual for torch.optim).
    variant : {"bfgs", "ssbfgs", "ssbroyden"}, default "ssbroyden"
        Which update formula to use for the inverse-Hessian approximation.
    lr : float, default 1.0
        Initial step length passed to the line search.
    line_search : bool or {"armijo", "strong_wolfe"}, default True
        Line-search mode. True (or "armijo") runs Armijo backtracking;
        "strong_wolfe" runs a cubic-interpolating strong-Wolfe search whose
        curvature condition guarantees y.s > 0, so the inverse-Hessian
        update stays positive definite without H resets (matching scipy's
        BFGS line search as used by Urban et al. 2025). False disables the
        search (full step lr).
    c1 : float, default 1e-4
        Armijo / sufficient-decrease condition constant.
    c2 : float, default 0.9
        Wolfe curvature condition constant (strong_wolfe mode only).
    backtrack : float, default 0.5
        Line-search contraction factor.
    max_ls : int, default 20
        Maximum backtracking steps.
    damping : float, default 1e-12
        Lower bound used to guard denominators.
    tau_min, tau_max : float
        Clamp for the self-scaling factor tau_k (ssbroyden variant only).
    reset_on_fail : bool, default True
        Reset H_k to identity when the curvature condition fails.
    H_on_cpu : bool, default False
        Keep H_k on CPU to save GPU memory (useful for large networks).
    """

    _VALID_VARIANTS = ("bfgs", "ssbfgs", "ssbroyden")

    def __init__(
        self,
        params,
        variant: str = "ssbroyden",
        lr: float = 1.0,
        line_search: bool | str = True,
        c1: float = 1e-4,
        c2: float = 0.9,
        backtrack: float = 0.5,
        max_ls: int = 20,
        damping: float = 1e-12,
        tau_min: float = 1e-6,
        tau_max: float = 1.0,
        reset_on_fail: bool = True,
        H_on_cpu: bool = False,
    ) -> None:
        if variant not in self._VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {self._VALID_VARIANTS}, got {variant!r}"
            )
        if isinstance(line_search, str) and line_search not in (
            "armijo",
            "strong_wolfe",
        ):
            raise ValueError(
                "line_search must be a bool, 'armijo' or 'strong_wolfe', "
                f"got {line_search!r}"
            )
        if line_search == "strong_wolfe" and _strong_wolfe is None:
            raise RuntimeError(
                "line_search='strong_wolfe' requires torch.optim.lbfgs."
                "_strong_wolfe, which this torch build does not provide."
            )
        defaults = dict(
            variant=variant,
            lr=lr,
            line_search=line_search,
            c1=c1,
            c2=c2,
            backtrack=backtrack,
            max_ls=max_ls,
            damping=damping,
            tau_min=tau_min,
            tau_max=tau_max,
            reset_on_fail=reset_on_fail,
            H_on_cpu=H_on_cpu,
        )
        super().__init__(params, defaults)
        self.H: torch.Tensor | None = None

    # ---------- parameter/gradient vector helpers ----------
    def _get_param_vector(self) -> torch.Tensor:
        return torch.cat(
            [p.data.view(-1) for g in self.param_groups for p in g["params"]]
        )

    def _set_param_vector(self, vec: torch.Tensor) -> None:
        offset = 0
        for g in self.param_groups:
            for p in g["params"]:
                n = p.numel()
                p.data.copy_(vec[offset : offset + n].view_as(p))
                offset += n

    def _get_grad_vector(self) -> torch.Tensor:
        grads = []
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p.data).view(-1))
                else:
                    grads.append(p.grad.data.view(-1))
        return torch.cat(grads)

    @torch.no_grad()
    def _init_H(
        self,
        n: int,
        ref_tensor: torch.Tensor,
        H_on_cpu: bool,
        force: bool = False,
    ) -> None:
        # First-time initialisation OR explicit reset (e.g. from a failed
        # curvature update with reset_on_fail=True). Without `force`, the
        # method is a no-op once H has the right shape, and reset_on_fail
        # silently fails to recover the optimiser — which is what triggers
        # the runaway J ~ 10^5 plateau on long-horizon runs.
        if force or self.H is None or self.H.shape[0] != n:
            dev = torch.device("cpu") if H_on_cpu else ref_tensor.device
            self.H = torch.eye(n, device=dev, dtype=ref_tensor.dtype)

    # ---------- main step ----------
    def step(self, closure, loss_eval):  # type: ignore[override]
        group = self.param_groups[0]
        variant = group["variant"]
        lr = group["lr"]
        line_search = group["line_search"]
        c1 = group["c1"]
        c2 = group["c2"]
        backtrack = group["backtrack"]
        max_ls = group["max_ls"]
        damping = group["damping"]
        tau_min = group["tau_min"]
        tau_max = group["tau_max"]
        reset_on_fail = group["reset_on_fail"]
        H_on_cpu = group["H_on_cpu"]

        loss = closure()
        g = self._get_grad_vector().detach()
        x = self._get_param_vector().detach()
        n = g.numel()
        self._init_H(n, g, H_on_cpu)

        # Search direction p = -H g
        gH = g.detach().cpu() if self.H.device != g.device else g
        Hg = self.H.matmul(gH)
        p_dir = (-Hg).to(g.device) if self.H.device != g.device else -Hg

        gtd = torch.dot(g, p_dir)
        gTp = float(gtd.item())
        f0 = float(loss.item())

        ls_mode = (
            line_search
            if isinstance(line_search, str)
            else ("armijo" if line_search else "none")
        )

        if ls_mode == "strong_wolfe":
            if gTp >= 0.0:
                # H lost positive definiteness numerically; this step falls
                # back to steepest descent and curvature is rebuilt from
                # identity.
                self._init_H(n, g, H_on_cpu, force=True)
                p_dir = -g
                gtd = torch.dot(g, p_dir)
                gTp = float(gtd.item())

            def _wolfe_obj(x_base, t, d):
                self._set_param_vector(x_base + t * d)
                loss_t = closure()
                g_t = self._get_grad_vector().detach().clone()
                return float(loss_t.item()), g_t

            _, _, alpha, _ = _strong_wolfe(
                _wolfe_obj,
                x,
                lr,
                p_dir,
                f0,
                g.clone(),
                gtd,
                c1=c1,
                c2=c2,
                max_ls=max_ls,
            )
            alpha = float(alpha)
            if (not np.isfinite(alpha)) or alpha <= 0.0:
                self._set_param_vector(x)
                if reset_on_fail:
                    self._init_H(n, g, H_on_cpu, force=True)
                return loss

            s = alpha * p_dir
            x_new = x + s
            self._set_param_vector(x_new)

            # Re-run the closure at the accepted iterate: the curvature pair
            # needs the gradient there, and caller-side logging hooked into
            # the closure (raw-J capture) must reflect the accepted point,
            # not the last line-search trial.
            new_loss = closure()
            g_new = self._get_grad_vector().detach()
            y = g_new - g
        else:
            # Armijo backtracking line search
            alpha = lr
            if ls_mode == "armijo":
                for _ in range(max_ls):
                    x_try = x + alpha * p_dir
                    self._set_param_vector(x_try)
                    f_try = float(loss_eval().item())
                    if f_try <= f0 + c1 * alpha * gTp:
                        break
                    alpha *= backtrack
                else:
                    alpha = 0.0

            if alpha == 0.0 or not np.isfinite(alpha):
                self._set_param_vector(x)
                if reset_on_fail:
                    self._init_H(n, g, H_on_cpu, force=True)
                return loss

            s = alpha * p_dir
            x_new = x + s
            self._set_param_vector(x_new)

            new_loss = closure()
            g_new = self._get_grad_vector().detach()
            y = g_new - g

        ys = torch.dot(y, s)
        if (not torch.isfinite(ys)) or (ys.abs() <= damping):
            if reset_on_fail:
                self._init_H(n, g_new, H_on_cpu, force=True)
            return new_loss

        # Cast to the device where H lives (for H_on_cpu mode)
        if self.H.device != g.device:
            yH = y.detach().cpu()
            sH = s.detach().cpu()
            gH2 = g.detach().cpu()
            HgH2 = Hg.detach()
        else:
            yH = y
            sH = s
            gH2 = g
            HgH2 = Hg

        Hy = self.H.matmul(yH)
        yHy = torch.dot(yH, Hy)
        if (not torch.isfinite(yHy)) or (yHy.abs() <= damping):
            if reset_on_fail:
                self._init_H(n, g_new, H_on_cpu, force=True)
            return new_loss

        # Scaling factor tau_k and weight phi_k — branch by variant
        if variant == "bfgs":
            tau_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
            phi_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
        elif variant == "ssbfgs":
            # SSBFGS: tau_k = min{1, (y.s) / (s . H^{-1} s)}, phi_k = 1  (eq. 11-12).
            # s . H^{-1} s is estimated without inverting H via: s . H^{-1} s = alpha * g . (-s) ... cannot
            # be done exactly without H^{-1}. Paper's appendix B shows tau_k = (y.s) / denom_B, where
            # denom_B = -alpha * g.s (using p = -H g, so H^{-1} s = -g / alpha, hence s.H^{-1}s = -g.s/alpha * alpha = -g.s).
            # Using s = alpha * p, g.p = -g.Hg, so -g.s = alpha * g.Hg = alpha * (s/alpha).Hg ... leading to
            # s.H^{-1}s = -g.s. See Urban et al. (2025), Appendix B.
            sTg = torch.dot(sH, gH2)
            denom = -sTg  # equals s . H_k^{-1} s under the search direction p = -H_k g
            denom_safe = torch.clamp(denom, min=damping)
            tau_k = torch.minimum(
                torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype),
                ys / denom_safe,
            )
            tau_k = torch.clamp(tau_k, min=tau_min, max=tau_max)
            phi_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
        elif variant == "ssbroyden":
            # Full self-scaled Broyden: eqs. (13)-(23) of the paper.
            sTg = torch.dot(sH, gH2)
            b_k = (-alpha * sTg) / ys
            h_k = yHy / ys
            a_k = h_k * b_k - 1.0

            if (
                (not torch.isfinite(a_k))
                or (a_k <= damping)
                or (not torch.isfinite(b_k))
                or (b_k <= damping)
            ):
                tau_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
                phi_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
            else:
                c_k = torch.sqrt(torch.clamp(a_k / (a_k + 1.0), min=0.0))
                rho_minus = torch.minimum(
                    torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype),
                    h_k * (1.0 - c_k),
                )
                rho_safe = torch.clamp(rho_minus, min=damping)

                theta_minus = rho_minus - (1.0 / a_k)
                theta_plus = 1.0 / rho_safe
                theta_hat = (1.0 - b_k) / torch.clamp(b_k, min=damping)

                theta_k = torch.maximum(
                    theta_minus, torch.minimum(theta_plus, theta_hat)
                )
                sigma_k = 1.0 + a_k * theta_k
                sigma_safe = torch.clamp(sigma_k, min=damping)

                gHg = torch.dot(gH2, HgH2)
                denom = (alpha * alpha) * torch.clamp(gHg, min=damping)
                tau1 = torch.minimum(
                    torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype),
                    ys / denom,
                )

                sigma_pow = torch.exp(-torch.log(sigma_safe) / (n - 1.0))

                if theta_k > 0:
                    tau2 = tau1 * torch.minimum(
                        sigma_pow, 1.0 / torch.clamp(theta_k, min=damping)
                    )
                else:
                    tau2 = torch.minimum(tau1 * sigma_pow, sigma_safe)

                tau_k = torch.clamp(tau2, min=tau_min, max=tau_max)
                phi_k = (1.0 - theta_k) / sigma_safe
        else:  # pragma: no cover — validated in __init__
            raise ValueError(f"Unknown variant {variant!r}")

        # Inverse-Hessian update (eq. 10 in the paper)
        v = torch.sqrt(torch.clamp(yHy, min=damping)) * (sH / ys - Hy / yHy)
        term = self.H - torch.outer(Hy, Hy) / yHy + phi_k * torch.outer(v, v)
        H_new = (1.0 / tau_k) * term + torch.outer(sH, sH) / ys
        self.H = 0.5 * (H_new + H_new.t())  # symmetrize for numerical stability

        return new_loss
