@echo off
REM ===========================================================================
REM  URBAN ENGINE (float64 dense Hessian + strong-Wolfe line search)
REM  fig:vacuum-gs-results  <-  pinn_vacuum_gs_2d.py
REM
REM  Same standardised protocol as the thesis inhouse baseline
REM  (vacuum_gs_ssbroyden_identity_*); the ONLY change is the QN core, swapped
REM  to Urban float64 + strong Wolfe via --engine urban. Run in its own terminal.
REM
REM  Output: BVP\results\vacuum_gs_ssbroyden_urban_<timestamp>\
REM ===========================================================================
cd /d "%~dp0"

pushd BVP\two_d
python pinn_vacuum_gs_2d.py --engine urban
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  Vacuum Grad-Shafranov 2D (urban engine) done.
echo  New folder: BVP\results\vacuum_gs_ssbroyden_urban_*\
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
