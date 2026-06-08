# Re-run guide — standardised 3×32 network (Windows GPU machine)

**All BVP benchmarks now use one network: 3 hidden layers × 32 units, Tanh.**
The script defaults had drifted to the smaller Urban-paper nets (Helmholtz
2×20, CFGS 1×30, NLP 2×30); they have been standardised to `(32,32,32)`. The
2D Box-Cox summaries now log `net=...` in `summary_table.txt` — check that line
reads `net=32x32x32` on every run before trusting it.

Optimiser protocol is unchanged: fixed handover at exactly 2000 Adam epochs →
quasi-Newton phase, QN-phase early stopping (urban-style relative-MA on
validation J), PDE/solution L2 diagnostics every epoch. Total epoch counts are
*caps*; early stopping ends most runs before the cap. Per-epoch, per-seed
histories are saved to `raw_histories.npz` in each run folder, so figures can
be regenerated without retraining.

---

## What needs re-running (architecture changed → results invalid)

| Thesis figure / table | Script (command) | Old net |
|---|---|---|
| `fig:nlp-ssbroyden` (Helmholtz "low" single run, §helmholtz-benchmark) | `python pinn_helmholtz_2d.py` | 2×20 |
| `fig:cfgs-ssbroyden` (CFGS single run) | `python pinn_ssbroyden_2d_urban.py` | 1×30 |
| `fig:helmholtz-2d-boxcox-sweep` + `tab:helmholtz-2d-boxcox-summary` | `python boxcox_sweep_2d_helmholtz.py` | 2×20 |
| `fig:cfgs-boxcox-sweep` + `tab:cfgs-boxcox-summary` | `python boxcox_sweep_2d_cfgs.py` | 1×30 |
| (new) 1D Box-Cox at the 2D protocol | `python boxcox_sweep_1d_2darch.py` | 2×20 |

### Already 3×32 — NO re-run needed
- `fig:bvp-1d-optimisers` (Fig 4.3) ← `optimiser_comparison_1d.py`
- `fig:bvp-1d-boxcox-sweep` ← `boxcox_sweep_1d_finegrained.py --engage-threshold 1.0`
- `fig:poisson-2d-results` ← `pinn_poisson_2d_unitsquare.py`

> The identity-benchmark scripts (`pinn_helmholtz_2d.py`,
> `pinn_ssbroyden_2d_urban.py`) carry their thesis settings in code / as CLI
> defaults: Helmholtz uses config `"low"` `(a1,a2)=(1,4), k=1`, `ssbroyden`,
> `identity`; CFGS uses `--variant SSBroyden2 --loss-transform identity`. Run
> them with **no extra flags** to reproduce the thesis runs.
>
> `pinn_nlgs_2d.py` (nonlinear Grad–Shafranov) is **not** a thesis figure — do
> not run it for the thesis (the old guide listed it here by mistake; the NLP
> figure is the Helmholtz single run above).

---

Run the two batches below in two terminals **in parallel**. The two heaviest
sweeps (Helmholtz Box-Cox, CFGS Box-Cox) are on separate terminals. Start each
terminal from the repo root (the folder containing `BVP\`).

## Terminal A

```bat
cd BVP\two_d

REM --- 2D Helmholtz Box-Cox sweep (heavy) -> fig:helmholtz-2d-boxcox-sweep ---
python boxcox_sweep_2d_helmholtz.py

REM --- 2D Helmholtz single identity run -> fig:nlp-ssbroyden ---
python pinn_helmholtz_2d.py
```

## Terminal B

```bat
cd BVP\two_d

REM --- 2D CFGS Box-Cox sweep (heavy) -> fig:cfgs-boxcox-sweep ---
python boxcox_sweep_2d_cfgs.py

REM --- 2D CFGS single identity run -> fig:cfgs-ssbroyden ---
python pinn_ssbroyden_2d_urban.py
```

## When a terminal frees up (heavy; the new 1D experiment)

```bat
cd BVP\one_d

REM 1D Box-Cox at the 2D protocol: 3x32, 10k collocation, 10k-epoch cap,
REM 2000 Adam -> SSBroyden, 11 lambdas x 3 seeds. Emits both the
REM unconditional and the success-conditioned figure.
python boxcox_sweep_1d_2darch.py
```

---

## Useful overrides

- Disable QN early stopping (run the full fixed budget): `--no-early-stop`
  (scripts with a CLI) or `early_stop=False`.
- Tighten / loosen early stopping: `--es-patience`, `--es-min-delta`,
  `--es-window`, `--es-stop-loss` (where exposed).
- Change the network for a one-off check: `--hidden 7 7 7 ...` is **not** what
  we want here — leave the default `32 32 32` for every thesis run.
- The two identity-benchmark scripts: `pinn_helmholtz_2d.py` keys off the
  in-code `config_name/qn_variant/loss_transform` knobs near the top of
  `main()`; `pinn_ssbroyden_2d_urban.py` is CLI-driven (`--variant`,
  `--loss-transform`, `--seed`).

## After the runs finish

1. Report the new output-folder names (or paste each `summary_table.txt`).
   Confirm every header shows `net=32x32x32`.
2. The thesis `\includegraphics` paths point at the **old** timestamped folders
   — they'll need repointing to the new ones (the `\IfFileExists` guard shows a
   placeholder until then, so the build won't break).
3. `tab:helmholtz-2d-boxcox-summary` and `tab:cfgs-boxcox-summary` numbers will
   change and need updating from the new `summary_table.txt`.
