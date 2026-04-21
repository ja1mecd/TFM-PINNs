# TFM-PINNs

Code for the thesis *Analysis on Physics-Informed Neural Networks* (UC3M).

## Layout

```
TFM-PINNs/
├── BVP/                                   Boundary-value-problem solvers
│   ├── one_d/
│   │   ├── pinn_bvpsolver_l2.py           1-D baseline (Adam + L-BFGS, L2 loss)
│   │   ├── pinn_bvpsolver_l2_BFGS.py      BFGS-only variant
│   │   ├── pinn_bvpsolver_l2_val_scheduler.py
│   │   │                                  1-D with validation split + ReduceLROnPlateau
│   │   └── architecture_sweep.py          Layer × neuron grid sweep (imports the baseline)
│   ├── two_d/
│   │   ├── pinn_bvpsolver2d_BFGS.py       2-D BVP with Adam + BFGS
│   │   └── pinn_ssbroyden_2d.py           2-D Grad–Shafranov with SSBroyden optimiser
│   ├── optimizers/
│   │   └── ssbroyden.py                   Standalone Self-Scaled Broyden optimiser
│   ├── figures/                           Plots produced by the scripts
│   └── models/                            Trained checkpoints (.pth)
│
└── Interpolation/                         Function-interpolation experiments
    ├── pinn_interpolant_l2.py             L2-minimising PINN interpolant
    ├── pinn_interpolant_linf.py           L∞-minimising PINN interpolant
    ├── error_table_pinn.py                Architecture grid sweep (imports pinn_interpolant_l2)
    ├── figures/                           Error heatmaps per activation
    └── models/                            Trained checkpoints (.pth)
```

## Running

Most scripts are self-contained (`python <script>.py`). Scripts in
`BVP/one_d/` and `BVP/two_d/` write checkpoints to `../models/` relative
to their own folder — run them from their directory so the paths resolve.

Scripts with sibling-import dependencies:

- `BVP/one_d/architecture_sweep.py` imports from `pinn_bvpsolver_l2.py`
  (same folder). Run from `BVP/one_d/`.
- `Interpolation/error_table_pinn.py` imports from `pinn_interpolant_l2.py`
  (same folder). Run from `Interpolation/`.

## Dependencies

`numpy`, `torch`, `matplotlib`. No `requirements.txt` yet — add one if you
freeze the environment before submission.
