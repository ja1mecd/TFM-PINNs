@echo off
REM ===========================================================================
REM  URBAN ENGINE + INITIAL HESSIAN SCALING (Urban-faithful re-run)
REM  fig:bvp-1d-optimisers / Table 4.4  <-  optimiser_comparison_1d.py
REM
REM  Re-run of the 1D BVP k=4 optimiser comparison with Urban's Oren-Luenberger
REM  initial Hessian scaling ENABLED (--initial-scale). The legacy urban run
REM  (bvp1d_k4_optim_compare_urban_20260613_152100) had it OFF, which is why
REM  SSBroyden only marginally beat BFGS/SSBFGS instead of dominating as in
REM  Urban 2025. The optimiser fix in ssbroyden_urban.py re-applies the scaling
REM  after every mid-run H-reset (not just step 0), which is what keeps the
REM  self-scaled iteration in-basin on the stiff k=4 residual.
REM
REM  Everything else matches the thesis protocol exactly: 20 seeds (42-61),
REM  5000-epoch budget, 2000-epoch fixed Adam warm-up, 3x32 Tanh, QN early
REM  stopping. Expectation: SSBroyden's success count rises well above the
REM  9/20 of the unscaled run and clears more seeds than BFGS/SSBFGS.
REM
REM  HEAVY run (20 seeds x 4 pipelines x float64 Hessian + Wolfe). To split
REM  across terminals:
REM     python optimiser_comparison_1d.py --engine urban --initial-scale --seeds 42 43 44 45 46 47 48 49 50 51
REM     python optimiser_comparison_1d.py --engine urban --initial-scale --seeds 52 53 54 55 56 57 58 59 60 61
REM  then merge with merge_optim_runs.py.
REM
REM  Output: BVP\results\bvp1d_k4_optim_compare_urban_iscale_<timestamp>\
REM ===========================================================================
cd /d "%~dp0"

pushd BVP\one_d
python optimiser_comparison_1d.py --engine urban --initial-scale ^
  --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  1D optimiser comparison (urban + initial scaling, 20 seeds) done.
echo  New folder: BVP\results\bvp1d_k4_optim_compare_urban_iscale_*\
echo  Compare SSBroyden success count vs the unscaled urban run.
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
