@echo off
REM ===========================================================================
REM  URBAN ENGINE (float64 dense Hessian + strong-Wolfe line search)
REM  fig:poisson-2d-results  <-  pinn_poisson_2d_unitsquare.py
REM
REM  Same standardised protocol as the thesis inhouse baseline
REM  (poisson2d_unitsquare_ssbroyden_identity_*); the ONLY change is the QN
REM  core, swapped from in-house float32 + Armijo to Urban float64 + strong
REM  Wolfe via --engine urban. Run this in its own terminal.
REM
REM  Output: BVP\results\poisson2d_unitsquare_ssbroyden_urban_<timestamp>\
REM ===========================================================================
cd /d "%~dp0"

pushd BVP\two_d
python pinn_poisson_2d_unitsquare.py --engine urban
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  Poisson 2D (urban engine) done.
echo  New folder: BVP\results\poisson2d_unitsquare_ssbroyden_urban_*\
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
