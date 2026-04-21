"""
CFGS PINN with objective transform:
  - raw functional:      J >= 0
  - optimized objective: can be
        identity:  J
        sqrt:      (J + eps)^(1/2)
        root:      (J + eps)^(1/n)     <-- NEW: n-th root (generalizes sqrt)
        log:       log(J + eps)

PDE (charge-free / vacuum Grad–Shafranov):
    Δ*ψ = ψ_RR - (1/R) ψ_R + ψ_ZZ = 0    on (R,Z) ∈ [Rmin,Rmax]×[Zmin,Zmax], with Rmin > 0.

Boundary conditions:
  - Enforced "hard" via ansatz ψ_hat = ψ_bc + g(R,Z) * NN(R,Z), where g=0 on the box boundary.
  - In this demo, ψ_bc = ψ_exact and ψ_exact(R,Z)=R^2 (exactly satisfies Δ*ψ=0).

Optimizers:
  - Adam for `adam_epochs`
  - then dense Self-Scaled Broyden (SSBroyden) for remaining epochs

IMPORTANT:
  Dense SSBroyden stores H ∈ R^{n×n} (n=#parameters) => O(n^2) memory.
  If you get OOM, reduce network size or set `ssbroyden_H_on_cpu=True`.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# =============================================================================
# DEVICE + performance knobs
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    _ = torch.zeros(1, device=device)  # initialize CUDA context early
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# =============================================================================
# CFGS DOMAIN (R,Z). IMPORTANT: Rmin > 0 to avoid the 1/R singularity.
# =============================================================================
Rmin, Rmax = 1.0, 2.0
Zmin, Zmax = -1.0, 1.0


# =============================================================================
# EXACT SOLUTION / BOUNDARY DATA (demo)
# ψ(R,Z) = R^2 satisfies Δ*ψ = ψ_RR - (1/R)ψ_R + ψ_ZZ = 2 - 2 + 0 = 0.
# =============================================================================
def psi_exact(RZ):
    if isinstance(RZ, torch.Tensor):
        R = RZ[:, 0:1]
        return R**2
    else:
        R = RZ[:, 0:1]
        return R**2


# =============================================================================
# Neural network ψθ(R,Z)
# =============================================================================
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers=(64, 64, 64), activation=nn.Tanh()):
        super().__init__()
        layers = []
        in_dim = 2
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Dense Self-Scaled Broyden optimizer (inverse-Hessian update)
# =============================================================================
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


# =============================================================================
# PINN solver for CFGS: Δ*ψ = 0, with objective transform (including n-th root)
# =============================================================================
class PINN_CFGS_Solver:
    def __init__(
        self,
        model,
        lr=1e-3,
        lambda_pde=1.0,
        loss_transform="root",   # "root" (default), "sqrt", "log", "identity"
        root_n=2,                # <-- NEW: n for n-th root (used when loss_transform=="root")
        loss_eps=1e-12,          # epsilon inside root/log
        rel_err_eps=1e-12,       # epsilon in denominators for relative errors
        ssbroyden_H_on_cpu=False,
    ):
        self.model = model.to(device)
        self.lambda_pde = float(lambda_pde)

        self.loss_transform = str(loss_transform)
        self.loss_eps = float(loss_eps)
        self.rel_err_eps = float(rel_err_eps)
        self.root_n = int(root_n)
        if self.root_n < 1:
            raise ValueError("root_n must be >= 1 for n-th root transform.")

        self.adam = optim.Adam(self.model.parameters(), lr=lr)

        self.ssbroyden = SSBroydenOptimizer(
            self.model.parameters(),
            lr=1.0,
            line_search=True,
            c1=1e-4,
            backtrack=0.5,
            max_ls=20,
            damping=1e-12,
            tau_min=1e-6,
            tau_max=1.0,
            reset_on_fail=True,
            H_on_cpu=ssbroyden_H_on_cpu,
        )

        # Logs
        self.obj_train = []
        self.obj_val = []
        self.J_train = []
        self.J_val = []
        self.pde_l2 = []
        self.sol_l2 = []
        self.sol_rel_l2 = []

        self.best_state = None
        self.best_val_ma = float("inf")

    # ---- transform J -> objective ----
    def _transform_objective(self, J_raw: torch.Tensor) -> torch.Tensor:
        eps = self.loss_eps
        if self.loss_transform == "identity":
            return J_raw
        if self.loss_transform == "sqrt":
            return torch.sqrt(J_raw + eps)
        if self.loss_transform == "root":
            inv_n = 1.0 / float(self.root_n)
            return torch.pow(J_raw + eps, inv_n)
        if self.loss_transform == "log":
            return torch.log(J_raw + eps)
        raise ValueError(f"Unknown loss_transform={self.loss_transform!r}")

    # ---- hard Dirichlet BC on box boundary ----
    def _psi_hat(self, RZ: torch.Tensor) -> torch.Tensor:
        R = RZ[:, 0:1]
        Z = RZ[:, 1:2]
        g = (R - Rmin) * (Rmax - R) * (Z - Zmin) * (Zmax - Z)
        return psi_exact(RZ) + g * self.model(RZ)

    # ---- Grad–Shafranov operator Δ*ψ ----
    def _delta_star(self, RZ: torch.Tensor, create_graph_second: bool) -> torch.Tensor:
        RZ = RZ.to(device)
        if not RZ.requires_grad:
            RZ = RZ.requires_grad_(True)

        psi = self._psi_hat(RZ)

        grads = torch.autograd.grad(
            psi, RZ, grad_outputs=torch.ones_like(psi), create_graph=True
        )[0]
        psi_R = grads[:, 0:1]
        psi_Z = grads[:, 1:2]

        psi_RR = torch.autograd.grad(
            psi_R,
            RZ,
            grad_outputs=torch.ones_like(psi_R),
            create_graph=create_graph_second,
            retain_graph=True,
        )[0][:, 0:1]

        psi_ZZ = torch.autograd.grad(
            psi_Z,
            RZ,
            grad_outputs=torch.ones_like(psi_Z),
            create_graph=create_graph_second,
        )[0][:, 1:2]

        R = RZ[:, 0:1]
        R_safe = torch.clamp(R, min=1e-6)

        return psi_RR - psi_R / R_safe + psi_ZZ

    # ---- compute loss: returns (objective, raw J) ----
    def compute_loss(self, RZ_interior: torch.Tensor, create_graph_second: bool):
        RZ = RZ_interior.detach().clone().requires_grad_(True)
        res = self._delta_star(RZ, create_graph_second=create_graph_second)

        area = (Rmax - Rmin) * (Zmax - Zmin)
        J_raw = self.lambda_pde * (torch.mean(res**2) * area)  # >= 0
        J_obj = self._transform_objective(J_raw)
        return J_obj, J_raw.detach()

    # ---- diagnostics on a grid (CPU sync) ----
    def compute_pde_l2(self, n=60):
        Rs = np.linspace(Rmin, Rmax, n)
        Zs = np.linspace(Zmin, Zmax, n)
        RR, ZZ = np.meshgrid(Rs, Zs, indexing="xy")
        RZ = np.stack([RR.ravel(), ZZ.ravel()], axis=1).astype(np.float32)
        RZt = torch.from_numpy(RZ).to(device)

        res = self._delta_star(RZt, create_graph_second=False).detach().cpu().numpy().reshape(n, n)
        intZ = np.trapz(res**2, Zs, axis=0)
        integral = np.trapz(intZ, Rs, axis=0)
        return float(np.sqrt(integral))

    def compute_sol_l2(self, n=60):
        Rs = np.linspace(Rmin, Rmax, n)
        Zs = np.linspace(Zmin, Zmax, n)
        RR, ZZ = np.meshgrid(Rs, Zs, indexing="xy")
        RZ = np.stack([RR.ravel(), ZZ.ravel()], axis=1).astype(np.float32)

        u_true = psi_exact(RZ).reshape(n, n)

        RZt = torch.from_numpy(RZ).to(device)
        with torch.no_grad():
            u_pred = self._psi_hat(RZt).cpu().numpy().reshape(n, n)

        diff = u_pred - u_true
        intZ = np.trapz(diff**2, Zs, axis=0)
        integral = np.trapz(intZ, Rs, axis=0)
        return float(np.sqrt(integral))

    def compute_sol_rel_l2(self, n=60):
        Rs = np.linspace(Rmin, Rmax, n)
        Zs = np.linspace(Zmin, Zmax, n)
        RR, ZZ = np.meshgrid(Rs, Zs, indexing="xy")
        RZ = np.stack([RR.ravel(), ZZ.ravel()], axis=1).astype(np.float32)

        u_true = psi_exact(RZ).reshape(n, n)

        RZt = torch.from_numpy(RZ).to(device)
        with torch.no_grad():
            u_pred = self._psi_hat(RZt).cpu().numpy().reshape(n, n)

        diff = u_pred - u_true
        intZ_num = np.trapz(diff**2, Zs, axis=0)
        num = np.trapz(intZ_num, Rs, axis=0)

        intZ_den = np.trapz(u_true**2, Zs, axis=0)
        den = np.trapz(intZ_den, Rs, axis=0)

        return float(np.sqrt(num) / (np.sqrt(den) + self.rel_err_eps))

    def train(
        self,
        n_epochs=20000,
        n_collocation=4000,
        train_split=0.7,
        resample_every=500,
        adam_epochs=2000,
        verbose_freq=200,
        diag_grid_n=60,
        patience=500,
        min_delta=1e-8,
        moving_avg_window=20,
        scheduler_patience=300,
        scheduler_threshold=1e-4,
        scheduler_gamma=0.9,
        scheduler_min_lr=1e-6,
    ):
        print("\nTraining CFGS PINN: Δ*ψ = 0")
        print(f"Domain: R∈[{Rmin},{Rmax}], Z∈[{Zmin},{Zmax}]")
        if self.loss_transform == "root":
            print(f"Objective transform: root (n={self.root_n})  (eps={self.loss_eps:g})")
        else:
            print(f"Objective transform: {self.loss_transform}  (eps={self.loss_eps:g})")
        print(f"Optimizers: Adam ({adam_epochs} epochs) then SSBroyden")
        print("-" * 80)

        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be in (0,1).")
        if n_collocation < 2:
            raise ValueError("n_collocation must be >= 2.")
        if resample_every < 1:
            raise ValueError("resample_every must be >= 1.")
        if adam_epochs < 0 or adam_epochs >= n_epochs:
            raise ValueError("adam_epochs must be in [0, n_epochs-1].")

        n_train = int(n_collocation * train_split)
        n_train = min(max(n_train, 1), n_collocation - 1)

        def resample_block():
            R = torch.empty(n_collocation, 1, device=device).uniform_(Rmin, Rmax)
            Z = torch.empty(n_collocation, 1, device=device).uniform_(Zmin, Zmax)
            RZ = torch.cat([R, Z], dim=1)
            perm = torch.randperm(n_collocation, device=device)
            RZ = RZ[perm]
            return RZ[:n_train].detach().clone(), RZ[n_train:].detach().clone()

        RZ_train, RZ_val = resample_block()

        def make_plateau(opt):
            try:
                return optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=scheduler_gamma,
                    patience=scheduler_patience,
                    threshold=scheduler_threshold,
                    verbose=True,
                    min_lr=scheduler_min_lr,
                )
            except TypeError:
                return optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=scheduler_gamma,
                    patience=scheduler_patience,
                    threshold=scheduler_threshold,
                    min_lr=scheduler_min_lr,
                )

        sch_adam = make_plateau(self.adam)
        sch_ss = make_plateau(self.ssbroyden)

        self.best_state = None
        self.best_val_ma = float("inf")
        ma_buf = []
        epochs_no_improve = 0

        last_pde_l2 = np.nan
        last_sol_l2 = np.nan
        last_sol_rel_l2 = np.nan

        for epoch in range(1, n_epochs + 1):
            if epoch != 1 and ((epoch - 1) % resample_every == 0):
                RZ_train, RZ_val = resample_block()

            use_adam = epoch <= adam_epochs
            opt = self.adam if use_adam else self.ssbroyden
            sch = sch_adam if use_adam else sch_ss

            if use_adam:
                opt.zero_grad()
                J_obj, J_raw = self.compute_loss(RZ_train, create_graph_second=True)
                J_obj.backward()
                opt.step()
            else:
                holder = {}

                def closure():
                    opt.zero_grad()
                    J_obj_c, J_raw_c = self.compute_loss(RZ_train, create_graph_second=True)
                    holder["J_raw"] = J_raw_c
                    J_obj_c.backward()
                    return J_obj_c

                def loss_eval():
                    J_obj_e, _ = self.compute_loss(RZ_train, create_graph_second=False)
                    return J_obj_e

                J_obj = opt.step(closure, loss_eval)
                J_raw = holder["J_raw"]

            with torch.set_grad_enabled(True):
                val_obj, val_raw = self.compute_loss(RZ_val, create_graph_second=False)

            self.obj_train.append(float(J_obj.item()))
            self.obj_val.append(float(val_obj.item()))
            self.J_train.append(float(J_raw.item()))
            self.J_val.append(float(val_raw.item()))

            # early stopping on moving average of validation objective
            ma_buf.append(float(val_obj.item()))
            if len(ma_buf) > moving_avg_window:
                ma_buf.pop(0)
            val_ma = float(np.mean(ma_buf))

            if val_ma + min_delta < self.best_val_ma:
                self.best_val_ma = val_ma
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            sch.step(float(val_obj.item()))

            # diagnostics occasionally
            if epoch == 1 or (epoch % verbose_freq == 0):
                last_pde_l2 = self.compute_pde_l2(n=diag_grid_n)
                last_sol_l2 = self.compute_sol_l2(n=diag_grid_n)
                last_sol_rel_l2 = self.compute_sol_rel_l2(n=diag_grid_n)

            self.pde_l2.append(last_pde_l2)
            self.sol_l2.append(last_sol_l2)
            self.sol_rel_l2.append(last_sol_rel_l2)

            if epoch == 1 or (epoch % verbose_freq == 0):
                lr_now = opt.param_groups[0]["lr"]
                phase = "ADAM" if use_adam else "SSBROYDEN"
                print(
                    f"Epoch {epoch:6d} [{phase}] | "
                    f"obj={self.obj_train[-1]:.3e}, val_obj={self.obj_val[-1]:.3e} | "
                    f"J={self.J_train[-1]:.3e}, val_J={self.J_val[-1]:.3e} | "
                    f"pdeL2={last_pde_l2:.3e}, solL2={last_sol_l2:.3e}, relSolL2={last_sol_rel_l2:.3e} | "
                    f"lr={lr_now:.2e}"
                )

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val-MA improvement for {patience} epochs).")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        print("-" * 80)
        print(f"Done. Best val objective moving average: {self.best_val_ma:.6e}")

    def plot_results(self, n=80):
        Rs = np.linspace(Rmin, Rmax, n)
        Zs = np.linspace(Zmin, Zmax, n)
        RR, ZZ = np.meshgrid(Rs, Zs, indexing="xy")
        RZ = np.stack([RR.ravel(), ZZ.ravel()], axis=1).astype(np.float32)

        u_true = psi_exact(RZ).reshape(n, n)
        RZt = torch.from_numpy(RZ).to(device)
        with torch.no_grad():
            u_pred = self._psi_hat(RZt).cpu().numpy().reshape(n, n)
        abs_err = np.abs(u_pred - u_true)
        rel_err = abs_err / (np.abs(u_true) + self.rel_err_eps)

        fig, ax = plt.subplots(2, 3, figsize=(16, 9))

        im0 = ax[0, 0].imshow(u_true, origin="lower", extent=[Rmin, Rmax, Zmin, Zmax], aspect="auto")
        ax[0, 0].set_title("ψ_exact(R,Z)")
        plt.colorbar(im0, ax=ax[0, 0], fraction=0.046)

        im1 = ax[0, 1].imshow(u_pred, origin="lower", extent=[Rmin, Rmax, Zmin, Zmax], aspect="auto")
        ax[0, 1].set_title("ψ_PINN(R,Z)")
        plt.colorbar(im1, ax=ax[0, 1], fraction=0.046)

        im2 = ax[0, 2].imshow(abs_err, origin="lower", extent=[Rmin, Rmax, Zmin, Zmax], aspect="auto")
        ax[0, 2].set_title("|ψ_PINN - ψ_exact| (absolute)")
        plt.colorbar(im2, ax=ax[0, 2], fraction=0.046)

        im3 = ax[1, 0].imshow(rel_err, origin="lower", extent=[Rmin, Rmax, Zmin, Zmax], aspect="auto")
        ax[1, 0].set_title("|ψ_PINN - ψ_exact| / (|ψ_exact| + eps)")
        plt.colorbar(im3, ax=ax[1, 0], fraction=0.046)

        ax[1, 1].semilogy(self.obj_train, label="obj(train)")
        ax[1, 1].semilogy(self.obj_val, label="obj(val)")
        ax[1, 1].semilogy(self.J_train, "--", label="J(train)")
        ax[1, 1].semilogy(self.J_val, "--", label="J(val)")
        ax[1, 1].grid(True, alpha=0.3)
        ax[1, 1].legend()
        ax[1, 1].set_title("Objective/Loss curves")
        ax[1, 1].set_xlabel("Epoch")
        ax[1, 1].set_ylabel("Value")

        ax[1, 2].semilogy(self.pde_l2, label="||Δ*ψ||_L2 (abs)")
        ax[1, 2].semilogy(self.sol_l2, label="||ψ-ψ_exact||_L2 (abs)")
        ax[1, 2].semilogy(self.sol_rel_l2, label="||ψ-ψ_exact||_L2 / ||ψ_exact||_L2")
        ax[1, 2].grid(True, alpha=0.3)
        ax[1, 2].legend()
        ax[1, 2].set_title("Absolute and Relative Errors")
        ax[1, 2].set_xlabel("Epoch")
        ax[1, 2].set_ylabel("Value")

        for i, j in [(0, 0), (0, 1), (0, 2), (1, 0)]:
            ax[i, j].set_xlabel("R")
            ax[i, j].set_ylabel("Z")

        plt.tight_layout()
        plt.show()

    def save(self, path="../models/pinn_cfgs_ssbroyden.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to: {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Choose objective transform here:
    #   "root"     => (J + eps)^(1/n)   with integer n = root_n
    #   "sqrt"     => (J + eps)^(1/2)
    #   "log"      => log(J + eps)
    #   "identity" => J
    loss_transform = "log"
    root_n = 4          # <-- set n here (n=2 matches sqrt)
    loss_eps = 1e-12

    # If SSBroyden OOMs on GPU, set this True (H stored on CPU)
    ssbroyden_H_on_cpu = False

    model = NeuralNetwork(hidden_layers=(64, 64, 64), activation=nn.Tanh())
    print("\nNeural Network Architecture:\n")
    print(model, "\n")

    pinn = PINN_CFGS_Solver(
        model=model,
        lr=1e-3,
        lambda_pde=1.0,
        loss_transform=loss_transform,
        root_n=root_n,
        loss_eps=loss_eps,
        rel_err_eps=1e-12,
        ssbroyden_H_on_cpu=ssbroyden_H_on_cpu,
    )

    pinn.train(
        n_epochs=20000,
        n_collocation=4000,
        train_split=0.7,
        resample_every=500,
        adam_epochs=500,
        verbose_freq=200,
        diag_grid_n=60,
        patience=500,
        min_delta=1e-8,
        moving_avg_window=20,
        scheduler_patience=300,
        scheduler_threshold=1e-4,
        scheduler_gamma=0.9,
        scheduler_min_lr=1e-6,
    )

    pinn.plot_results(n=80)
    pinn.save("../models/pinn_cfgs_ssbroyden.pth")


if __name__ == "__main__":
    main()
