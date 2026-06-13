@echo off
REM ===========================================================================
REM  URBAN ENGINE (float64 dense Hessian + strong-Wolfe line search)
REM  fig:helmholtz-2d-boxcox-sweep  <-  boxcox_sweep_2d_helmholtz.py
REM
REM  Box-Cox lambda sweep on 2D Helmholtz (a1,a2)=(1,4), k=1. Same standardised
REM  protocol and default lambda grid as the thesis inhouse baseline
REM  (helmholtz2d_a1_4_k1_boxcox_finesweep_ssbroyden_*); the ONLY change is the
REM  QN core, swapped to Urban float64 + strong Wolfe via --engine urban.
REM  Run in its own terminal. This is the longest of the sweeps.
REM
REM  Output: BVP\results\helmholtz2d_a1_4_k1_boxcox_finesweep_ssbroydenurban_<timestamp>\
REM ===========================================================================
cd /d "%~dp0"

pushd BVP\two_d
python boxcox_sweep_2d_helmholtz.py --engine urban
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  Helmholtz Box-Cox sweep (urban engine) done.
echo  New folder: BVP\results\helmholtz2d_a1_4_k1_boxcox_finesweep_ssbroydenurban_*\
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
