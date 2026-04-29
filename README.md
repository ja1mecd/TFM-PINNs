# TFM-PINNs

Code for the thesis *Analysis on Physics-Informed Neural Networks* (UC3M).

## Layout

```
TFM-PINNs/
├── BVP/                                   Boundary-value-problem solvers
│   ├── one_d/
│   │   ├── pinn_bvpsolver_l2.py           1-D baseline (Adam + scheduler, soft BCs)
│   │   ├── pinn_bvpsolver_l2_BFGS.py      1-D BFGS variant (Adam + L-BFGS)
│   │   ├── pinn_bvpsolver_l2_val_scheduler.py
│   │   │                                  1-D with validation split + ReduceLROnPlateau
│   │   └── architecture_sweep.py          Layer × neuron grid sweep (imports the baseline)
│   ├── two_d/
│   │   ├── pinn_ssbroyden_2d.py           Replicates Urbán+ (2025) §4.1 CFGS
│   │   └── pinn_bvpsolver2d_BFGS.py       Replicates Urbán+ (2025) §5 NLP
│   ├── optimizers/
│   │   └── ssbroyden.py                   Unified BFGS / SSBFGS / SSBroyden optimiser
│   ├── figures/                           Plots produced by the scripts
│   └── models/                            Trained checkpoints (.pth)
│
└── Interpolation/                         Function-interpolation experiments
    ├── pinn_interpolant_l2.py             L2-minimising PINN interpolant (single-config demo + shared building block)
    ├── error_table_pinn.py                Architecture x activation sweep, reports L∞ error (imports pinn_interpolant_l2)
    ├── figures/                           Error heatmaps per activation
    └── models/                            Trained checkpoints (.pth)
```

## Paper replication

Reference: Urbán, Stefanou & Pons (2025), *Self-scaled Broyden-family
quasi-Newton methods for Physics-Informed Neural Networks*,
J. Comp. Phys. 523, 113656.

| Script                          | Paper section                       |
|---------------------------------|-------------------------------------|
| `two_d/pinn_ssbroyden_2d.py`    | §4.1 Current-Free Grad–Shafranov    |
| `two_d/pinn_bvpsolver2d_BFGS.py`| §5   Non-linear Poisson (Liouville) |

Both scripts pull the quasi-Newton optimiser from
`BVP/optimizers/ssbroyden.py` (added to `sys.path` automatically).
User knobs exposed in each script's `main()`:

- `qn_variant ∈ {"bfgs", "ssbfgs", "ssbroyden"}` — selects the update
  formula. `"bfgs"` uses τ_k = φ_k = 1, `"ssbfgs"` adds self-scaling
  (eqs. 11–12 of the paper), `"ssbroyden"` is the full Broyden-family
  update (eqs. 13–23).
- `loss_transform ∈ {"identity", "sqrt", "log"}` — maps J → J, √(J+ε),
  log(J+ε) to reproduce the paper's loss-transform sweeps (eqs. 26–27).
- `qn_H_on_cpu` — keeps the dense Hessian approximation on CPU for
  memory-constrained GPUs.

Hyperparameters follow paper Tables 1 and 4 (1 × 30 / 5k iters for CFGS,
2 × 30 / 20k iters for NLP, with tanh and the reported batch sizes).

The 1-D scripts in `BVP/one_d/` are independent from this replication
effort — they are used alongside the interpolation experiments and for
standalone BVP studies in the thesis.

### Not yet covered

The paper also studies NLGS, 2DH, NLS, KdV, 1DB, AC, 3DNS and LDC. These
would need new scripts.

## Running

Most scripts are self-contained (`python <script>.py`). Scripts in
`BVP/one_d/` and `BVP/two_d/` write checkpoints to `../models/` relative
to their own folder — run them from their directory so the paths resolve.

Scripts with sibling-import dependencies:

- `BVP/one_d/architecture_sweep.py` imports from `pinn_bvpsolver_l2.py`
  (same folder). Run from `BVP/one_d/`.
- `Interpolation/error_table_pinn.py` imports from `pinn_interpolant_l2.py`
  (same folder). Run from `Interpolation/`. Sweep an activation with
  `python error_table_pinn.py --activation Tanh` (also `Sigmoid`, `ReLU`,
  `Softmax`); the heatmap is written to `figures/error_table_pinn_log_<activation>.png`.

## Dependencies

`numpy`, `torch`, `matplotlib`. No `requirements.txt` yet — add one if you
freeze the environment before submission.
