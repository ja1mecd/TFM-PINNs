import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class SSBroydenOptimizer(optim.Optimizer):
    """
    Dense self-scaled Broyden update of inverse Hessian approximation H_k.

    WARNING: O(n^2) memory/time.
    """

    def __init__(
        self,
        params,
        lr=1.0,
        line_search=True,
        c1=1e-4,
        backtrack=0.5,
        max_ls=20,
        damping=1e-12,
        tau_min=1e-6,
        tau_max=1.0,
        reset_on_fail=True,
        H_on_cpu=False,
    ):
        defaults = dict(
            lr=lr,
            line_search=line_search,
            c1=c1,
            backtrack=backtrack,
            max_ls=max_ls,
            damping=damping,
            tau_min=tau_min,
            tau_max=tau_max,
            reset_on_fail=reset_on_fail,
            H_on_cpu=H_on_cpu,
        )
        super().__init__(params, defaults)
        self.H = None

    def _get_param_vector(self):
        return torch.cat([p.data.view(-1) for g in self.param_groups for p in g["params"]])

    def _set_param_vector(self, vec):
        offset = 0
        for g in self.param_groups:
            for p in g["params"]:
                n = p.numel()
                p.data.copy_(vec[offset : offset + n].view_as(p))
                offset += n

    def _get_grad_vector(self):
        grads = []
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p.data).view(-1))
                else:
                    grads.append(p.grad.data.view(-1))
        return torch.cat(grads)

    @torch.no_grad()
    def _init_H(self, n, ref_tensor, H_on_cpu: bool):
        if self.H is None or self.H.shape[0] != n:
            dev = torch.device("cpu") if H_on_cpu else ref_tensor.device
            self.H = torch.eye(n, device=dev, dtype=ref_tensor.dtype)

    def step(self, closure, loss_eval):
        """
        closure(): zero_grad -> compute objective -> backward -> return objective tensor
        loss_eval(): compute objective only (no backward), used for line search
        """
        group = self.param_groups[0]
        lr = group["lr"]
        line_search = group["line_search"]
        c1 = group["c1"]
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

        if self.H.device != g.device:
            gH = g.detach().cpu()
        else:
            gH = g

        Hg = self.H.matmul(gH)
        p_dir_H = -Hg
        p_dir = p_dir_H.to(g.device) if self.H.device != g.device else p_dir_H

        gTp = torch.dot(g, p_dir).item()
        f0 = float(loss.item())

        alpha = lr
        if line_search:
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
                self._init_H(n, g, H_on_cpu)
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
                self._init_H(n, g_new, H_on_cpu)
            return new_loss

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
                self._init_H(n, g_new, H_on_cpu)
            return new_loss

        sTg = torch.dot(sH, gH2)
        b_k = (-alpha * sTg) / ys
        h_k = yHy / ys
        a_k = h_k * b_k - 1.0

        if (not torch.isfinite(a_k)) or (a_k <= damping) or (not torch.isfinite(b_k)) or (b_k <= damping):
            tau_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
            phi_k = torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype)
        else:
            c_k = torch.sqrt(torch.clamp(a_k / (a_k + 1.0), min=0.0))
            rho_minus = torch.minimum(torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype), h_k * (1.0 - c_k))
            rho_safe = torch.clamp(rho_minus, min=damping)

            theta_minus = rho_minus - (1.0 / a_k)
            theta_plus = 1.0 / rho_safe
            theta_hat = (1.0 - b_k) / torch.clamp(b_k, min=damping)

            theta_k = torch.maximum(theta_minus, torch.minimum(theta_plus, theta_hat))
            sigma_k = 1.0 + a_k * theta_k
            sigma_safe = torch.clamp(sigma_k, min=damping)

            gHg = torch.dot(gH2, HgH2)
            denom = (alpha * alpha) * torch.clamp(gHg, min=damping)
            tau1 = torch.minimum(torch.tensor(1.0, device=self.H.device, dtype=self.H.dtype), ys / denom)

            sigma_pow = torch.exp(-torch.log(sigma_safe) / (n - 1.0))

            if theta_k > 0:
                tau2 = tau1 * torch.minimum(sigma_pow, 1.0 / torch.clamp(theta_k, min=damping))
            else:
                tau2 = torch.minimum(tau1 * sigma_pow, sigma_safe)

            tau_k = torch.clamp(tau2, min=tau_min, max=tau_max)
            phi_k = (1.0 - theta_k) / sigma_safe

        v = torch.sqrt(torch.clamp(yHy, min=damping)) * (sH / ys - Hy / yHy)
        term = self.H - torch.outer(Hy, Hy) / yHy + phi_k * torch.outer(v, v)
        H_new = (1.0 / tau_k) * term + torch.outer(sH, sH) / ys
        self.H = 0.5 * (H_new + H_new.t())

        return new_loss

