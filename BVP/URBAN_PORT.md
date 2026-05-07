# Urban-style port — change log

This document lists every behavioural change introduced by the
`*_urban.py` variants of the SSBroyden pipeline so the comparison with
the existing scripts is auditable.

The reference is Jorge Urbán's repository
(https://github.com/jorgeurban/self_scaled_algorithms_pinns), specifically
`modified_minimize.py`, `modified_optimize.py`, `Examples/AC/AC.py` and
`Examples/AC/AC_hparams.py`.

## New files

| Layer       | New file                                                | Pairs with                                  |
|-------------|---------------------------------------------------------|---------------------------------------------|
| Optimiser   | `optimizers/ssbroyden_urban.py`                         | `optimizers/ssbroyden.py`                   |
| 1D training | `one_d/pinn_bvpsolver_l2_SSBroyden_urban.py`            | `one_d/pinn_bvpsolver_l2_SSBroyden.py`      |
| 2D training | `two_d/pinn_ssbroyden_2d_urban.py`                      | `two_d/pinn_ssbroyden_2d.py` (problem reused via import) |

The existing files are untouched; you can run both variants on the same
problem and diff the logs / final metrics.

## What changed in the optimiser

| #  | Change | Why it matters |
|----|--------|----------------|
| 1  | **Strong Wolfe line search** (Armijo *and* curvature, with cubic-interpolated zoom) instead of pure Armijo backtracking. | BFGS theory needs the Wolfe-2 curvature condition to keep H positive-definite. Without it, `y·s ≤ 0` is common and forces frequent identity resets. |
| 2  | All seven update variants from Urbán et al. (2025) JCP: `BFGS`, `BFGS_scipy`, `SSBFGS_OL`, `SSBFGS_AB`, `SSBroyden1`, `SSBroyden2`, `SSBroyden3`. | The existing optimiser only has `bfgs`, `ssbfgs` and one SSBroyden flavour (matching SSBroyden2). |
| 3  | `tau_k` is **not capped at 1** by default (`tau_max=None`). | The existing code clamps to `[1e-6, 1.0]`, which disables the "scale H up" direction of self-scaling. |
| 4  | **`abs(a_k)` inside the sqrt** in the SSBroyden formulas, matching Urbán. | The existing code clamps the ratio to `≥ 0`, which silently degenerates to plain BFGS for negative `a_k`. |
| 5  | Optional `initial_scale=True` branch when `H == I`. | Urbán's `initial_scale and np.allclose(Hk, I)` block. Off by default. |
| 6  | `loss_and_grad(x_vec) -> (loss, grad)` API instead of `step(closure, loss_eval)`. | Wolfe needs gradient evaluations at trial points; the closure pattern does not expose those cheaply. |
| 7  | Dedicated `warm_start_from(H, cholesky_check=True)` to Cholesky-check user-supplied warm starts. | Mirrors Urbán's `try: cholesky(H0)` block in `AC.py`. |
| 8  | **float64 dense Hessian** by default. | Parity with Urbán's NumPy run; float32 is too noisy for the SSBroyden update. Network weights stay float32 unless you change them. |

## What changed in the training scripts

| #  | Change | Why it matters |
|----|--------|----------------|
| A  | `--loss-transform {identity,sqrt,log,boxcox}` flag controls the QN-phase objective `g(J)`. Adam still optimises raw `J`. | Urbán reshapes the loss landscape during QN; this is the `use_log` / `use_sqrt` knob in `AC_hparams.py`. The existing 1D script has no equivalent; the existing 2D script already supports this in-class but does not pair it with the optimiser changes. |
| B  | **Residual-adaptive sampling (RAD)** in the QN phase. Every `rad_resample_every` iterations a fresh batch is drawn from a pool of `rad_pool_size` uniform candidates with weight `(\|res\|^k1 / mean) + k2`. | Matches Urbán's `adaptive_rad` helper. Concentrates QN updates on high-residual regions. |
| C  | At each RAD restart, the dense `H` is **kept as a warm start, symmetrised, and Cholesky-checked**; identity reset only if Cholesky fails. | Mirrors Urbán's restart loop in `AC.py:507-535`. The existing pipeline carries `H` across blocks but never verifies positive-definiteness. |
| D  | Diagnostics expose `n_resets`, `n_ls_failures`, `n_func_evals`, last `alpha`, last `tau`, last `phi`, last gradient norm. | Makes it cheap to detect cases where the line search is failing or the self-scaling factor is saturating against a clamp. |

## CLI usage

### 1D BVP
```bash
# Headline Urban replication: SSBroyden2 + log-loss + RAD + warm-start.
python pinn_bvpsolver_l2_SSBroyden_urban.py \
    --variant SSBroyden2 --loss-transform log

# Compare against plain BFGS with identity loss (matches the BFGS script).
python pinn_bvpsolver_l2_SSBroyden_urban.py \
    --variant BFGS --loss-transform identity

# Saddle-avoiding SSBroyden1 + sqrt-loss + initial_scale.
python pinn_bvpsolver_l2_SSBroyden_urban.py \
    --variant SSBroyden1 --loss-transform sqrt --initial-scale
```

### 2D CFGS
```bash
# Headline replication. Default 30-neuron 1-layer net, 5000 total iters.
python pinn_ssbroyden_2d_urban.py \
    --variant SSBroyden2 --loss-transform log

# Pure replication of the existing pipeline behaviour with the new
# optimiser only (no loss transform, no RAD-driven restarts):
python pinn_ssbroyden_2d_urban.py \
    --variant SSBroyden2 --loss-transform identity \
    --rad-resample-every 100000  # effectively disables RAD
```

Run artefacts are written to `BVP/results/cfgs_urban_<variant>_<transform>_<tag>/`,
including `logs.npz` (raw and transformed losses, L^2 errors), `metadata.json`
(QN diagnostics + argv), and `results.png`.

## GPU notes

- Tested only on CPU here; the dense `H` is allocated on the parameter
  device by default. Use `--H-on-cpu` if `H_dtype × N²` does not fit
  alongside the autograd graph on the GPU.
- TF32 matmuls are explicitly disabled on CUDA so the Hessian update is
  not silently downcast to TF19 — Urbán runs in float64 on CPU and
  matching that is the whole point of the float64 default.
- The Wolfe line search performs up to `wolfe_max_ls` (default 25)
  forward+backward passes per QN step. On a 32×32×32 net (~3.2k params)
  expect ~5-10× the per-step cost of the Armijo path, but with far fewer
  identity resets — net wall-clock is usually similar or better.

## Side-by-side comparison protocol

To make the comparison clean, run the two variants on identical seeds,
identical Adam warmup, identical collocation budget. The only intended
difference is the QN behaviour. A fair triple would be:

```bash
SEED=7
for run in legacy urban; do
    if [ "$run" = legacy ]; then
        python one_d/pinn_bvpsolver_l2_SSBroyden.py
    else
        python one_d/pinn_bvpsolver_l2_SSBroyden_urban.py \
            --variant SSBroyden2 --loss-transform log --seed $SEED
    fi
done
```

The interesting comparison metrics are the final `||u'' - f||_{L^2}`
and `||u_NN - u_exact||_{L^2}` (1D), and the final relative `L^2` error
on the (q, mu) grid (2D).
