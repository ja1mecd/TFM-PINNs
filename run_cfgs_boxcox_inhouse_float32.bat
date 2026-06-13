@echo off
REM CFGS (current-free Grad-Shafranov) Box-Cox sweep, rerun on the IN-HOUSE
REM quasi-Newton backend so the cross-regime comparison against Helmholtz is
REM apples-to-apples.
REM
REM Why: the reported CFGS results used a different optimiser stack from every
REM other benchmark in the thesis. CFGS ran on the Urban solver
REM (ssbroyden_urban.py) with a float64 inverse-Hessian and a strong-Wolfe line
REM search, while Helmholtz / Poisson / vacuum-GS / 1D all ran on the in-house
REM ssbroyden.py with a float32 inverse-Hessian and Armijo backtracking. That
REM made the headline contrast (small lambda helps CFGS, hurts Helmholtz)
REM confounded with precision + line search + implementation.
REM
REM This run changes ONLY the quasi-Newton core. The Grad-Shafranov operator,
REM hard ansatz, RAD resampling, collocation count, handover, early stopping,
REM budget and seeds are all identical to the float64 run -- the new
REM --qn-backend inhouse flag swaps in the float32 + Armijo ssbroyden.py
REM optimiser, exactly matching the Helmholtz stack.
REM
REM Compare the output against the existing float64 run:
REM   BVP\results\cfgs_urban_SSBroyden2_boxcox_finesweep_20260608_211555\
REM Two outcomes, both informative:
REM   - small-lambda benefit SURVIVES at float32+Armijo  -> regime claim robust,
REM     the precision/line-search confound was harmless.
REM   - it COLLAPSES (no deep J, small lambda stops helping) -> the CFGS result
REM     was substantially a float64/Wolfe artefact, and the thesis needs it.
REM
REM Everything is left at the sweep defaults so it matches the float64 run:
REM   variant SSBroyden2 (-> in-house "ssbroyden"), lambda {0,0.1,...,1.0},
REM   5000-epoch cap, 2000 Adam warm-up, 1000 collocation, 3x32 Tanh,
REM   RAD resample 500, seeds 42 43 44.
REM
REM Output lands in BVP\results\cfgs_inhouse_SSBroyden2_boxcox_finesweep_*/

cd /d "%~dp0"

pushd BVP\two_d
python boxcox_sweep_2d_cfgs.py --qn-backend inhouse
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  CFGS Box-Cox sweep (in-house float32 + Armijo) completed.
echo  New folder in BVP\results\:
echo    - cfgs_inhouse_SSBroyden2_boxcox_finesweep_*/
echo  Compare against the float64 run:
echo    - cfgs_urban_SSBroyden2_boxcox_finesweep_20260608_211555/
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
