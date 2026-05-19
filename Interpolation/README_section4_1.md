# Section 4.1 interpolation study — how to reproduce

Full sweep: 4 activations x 35 cells (L in 1..7, W in 5,10,20,40,80) x 20 seeds
= 2800 trainings. Run on the GPU box from this directory:

    cd TFM-PINNs/Interpolation
    python run_interpolation_study.py

Outputs:
- `figures/error_table_pinn_log_<activation>.png` — mean log10 e_inf heatmaps
  (the paths the thesis \includegraphics already points at). Cells whose
  ensemble failed to train render gray and are labelled "fail".
- `results/error_table_pinn_<activation>.json` — raw per-seed records
  (linf, l2, train_time_s, epochs_run) plus aggregates. This is the
  reproducible source of truth for the section 4.1 numbers.
- `results/error_table_pinn_<activation>.partial.json` — per-cell
  checkpoint written during the run; safe to delete after completion.
- `thesis/tables/interpolation_summary.tex` — TFM-4 Table 4.1 analog
  (best (L,W) per activation, L-inf and L2 mean +- std, failed-cell
  count, mean time per cell). Resolved relative to the script via an
  absolute default path, so it lands in the repo's thesis/tables/.

Seeds: 42..61 (20). Single precision (float32), machine eps ~1.19e-7,
recorded in each JSON as `machine_eps`. A seed/cell that raises during
training is recorded with linf=l2=inf (epochs_run=0) so one failure
never aborts the 2800-run sweep.

Resilience: per-seed exceptions are isolated; per-cell partial
checkpoints let an interrupted run resume context. cuDNN is left in
default (non-deterministic) mode by design — the 20-seed ensemble is
for variance estimation, not bitwise reproducibility.

Smoke test (fast, from the repo root):

    python3 -m pytest TFM-PINNs/Interpolation/tests/ -v

Single dissected run (unchanged, the "one fully-worked example" figure):

    python3 pinn_interpolant_l2.py

Next step after the full run: rewrite thesis section 4.1 prose against
the regenerated numbers in results/*.json and the new summary table.
