# Re-run guide — standardised optimiser protocol (Windows GPU machine)

All Chapter-4 BVP figure scripts now share **one** optimiser protocol:

- **Fixed handover at exactly 2000 Adam epochs**, then the quasi-Newton phase.
- **QN-phase early stopping** (urban-style relative-MA on validation J:
  `es_patience=300`, `es_window=20`, `es_min_delta=1e-4`; optional absolute
  `es_stop_loss`). It can never cut the Adam warm-up short, and runs end as
  soon as the QN phase converges, so wall-clock per run is now variable.
- **PDE / solution L2 diagnostics computed every epoch** (printing is still
  throttled by `verbose_freq`). This removes the old `sol_l2` staircase.

Total epoch counts are now *caps* (2000 Adam + a generous QN budget); early
stopping ends most runs well before the cap.

> NOTE: the convergence-plot aggregation (mean + IQR) is still the old
> linear-mean-on-log scheme — deliberately unchanged for now. It will be
> redesigned (median / geometric mean) once these re-run histories are in.
> The raw per-epoch, per-seed histories are saved to `raw_histories.npz` in
> every run's output folder, so the plots can be regenerated without retraining.

Run the two batches below in two terminals **in parallel**. Each batch is
balanced so the two heaviest sweeps (Helmholtz Box-Cox, CFGS Box-Cox) run on
separate terminals. Adjust `--seeds`, `--lambdas`, problem params, etc. to
match your previous invocations where noted.

---

## Terminal A

```bat
REM --- 1D optimiser comparison (Fig 4.3) ---
cd BVP\one_d
python optimiser_comparison_1d.py

REM --- 1D Box-Cox fine sweep, delayed SSBroyden engagement ---
REM   (set --engage-threshold / --lambdas to match your previous run)
python boxcox_sweep_1d_finegrained.py --engage-threshold 1.0

REM --- 2D Poisson on the unit square ---
cd ..\two_d
python pinn_poisson_2d_unitsquare.py

REM --- 2D Helmholtz Box-Cox sweep (heavy) ---
python boxcox_sweep_2d_helmholtz.py
```

## Terminal B

```bat
REM --- 2D CFGS Box-Cox sweep (heavy) ---
cd BVP\two_d
python boxcox_sweep_2d_cfgs.py

REM --- 2D CFGS single SSBroyden run (Fig cfgs-ssbroyden) ---
python pinn_ssbroyden_2d_urban.py

REM --- 2D nonlinear Grad-Shafranov (Fig nlp-ssbroyden) ---
python pinn_nlgs_2d.py
```

---

## Useful overrides

- Disable QN early stopping (run the full fixed budget):
  `--no-early-stop` (scripts with a CLI) or pass `early_stop=False`.
- Tighten / loosen early stopping: `--es-patience`, `--es-min-delta`,
  `--es-window`, `--es-stop-loss` (where exposed).
- Force a different warm-up length: `--adam-epochs N` / `--adam-warmup N`.
- `pinn_nlgs_2d.py` has no CLI; edit the `pinn.train(...)` call in its
  `__main__` if you need different settings (currently 2000 Adam, 10000 cap).

After both terminals finish, report back the output folder names (or just the
`summary_table.txt` / `raw_histories.npz` paths) and we'll redo the figures.
