@echo off
REM ===========================================================================
REM  URBAN ENGINE (float64 dense Hessian + strong-Wolfe line search)
REM  2D Poisson optimiser comparison  <-  optimiser_comparison_2d_poisson.py
REM
REM  Sibling of run_urban_optim_compare_1d.bat, on the 2D Poisson benchmark
REM  (u_xx + u_yy = -2 pi^2 sin(pi x) sin(pi y), hard box ansatz, 3x32 Tanh).
REM  Four pipelines (Adam, Adam->BFGS, Adam->SSBFGS, Adam->SSBroyden), 20 seeds
REM  (42-61), 10000-epoch budget, 2000-epoch fixed Adam warm-up, QN early
REM  stopping -- the standardised 2D protocol.
REM
REM  KEY DIFFERENCE vs the 1D run: --engine urban with Urban's initial Hessian
REM  scaling ON (the default here). The optimiser fix in ssbroyden_urban.py
REM  re-applies the Oren-Luenberger scaling after every mid-run H-reset, not
REM  just on step 0, which is what Urban's headline SSBroyden relies on. The
REM  1D run was made before this fix (initial_scale OFF); do not compare the
REM  two blindly.
REM
REM  Figure uses the chapter-wide 0.01 success criterion and portrait geometry,
REM  so the output PNG drops straight into the thesis.
REM
REM  This is a HEAVY, GPU-bound run (20 seeds x 4 pipelines x float64 Hessian +
REM  Wolfe + 2D Laplacian backward). To parallelise across terminals, split the
REM  seed list, e.g.:
REM     python optimiser_comparison_2d_poisson.py --seeds 42 43 44 45 46 47 48 49 50 51
REM     python optimiser_comparison_2d_poisson.py --seeds 52 53 54 55 56 57 58 59 60 61
REM
REM  Output: BVP\results\poisson2d_optim_compare_urban_<timestamp>\
REM ===========================================================================
cd /d "%~dp0"

pushd BVP\two_d
python optimiser_comparison_2d_poisson.py --engine urban ^
  --success-rel-l2-threshold 0.01 --portrait ^
  --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  2D Poisson optimiser comparison (urban engine, 20 seeds) done.
echo  New folder: BVP\results\poisson2d_optim_compare_urban_*\
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
