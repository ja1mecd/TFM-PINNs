@echo off
REM ===========================================================================
REM  URBAN ENGINE (float64 dense Hessian + strong-Wolfe line search)
REM  fig:bvp-1d-optimisers / Table 4.4  <-  optimiser_comparison_1d.py
REM
REM  20-seed (42-61) optimiser comparison on the 1D BVP, k=4, 5000-epoch budget,
REM  2000-epoch fixed Adam warm-up, 3x32 Tanh, QN early stopping -- the exact
REM  thesis protocol. The ONLY change vs the inhouse baseline
REM  (bvp1d_k4_optim_compare_merged_20seeds) is the QN core, swapped to Urban
REM  float64 + strong Wolfe via --engine urban.
REM
REM  This is the HEAVIEST run (20 seeds x 4 pipelines x float64 Hessian + Wolfe).
REM  To parallelise across several terminals, split the seed list, e.g.:
REM     python optimiser_comparison_1d.py --engine urban --seeds 42 43 44 45 46 47 48 49 50 51
REM     python optimiser_comparison_1d.py --engine urban --seeds 52 53 54 55 56 57 58 59 60 61
REM  then merge the two result dirs with merge_optim_runs.py (as the inhouse
REM  20-seed figure was assembled). This bat runs all 20 in one process.
REM
REM  Output: BVP\results\bvp1d_k4_optim_compare_urban_<timestamp>\
REM ===========================================================================
cd /d "%~dp0"

pushd BVP\one_d
python optimiser_comparison_1d.py --engine urban ^
  --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  1D optimiser comparison (urban engine, 20 seeds) done.
echo  New folder: BVP\results\bvp1d_k4_optim_compare_urban_*\
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
