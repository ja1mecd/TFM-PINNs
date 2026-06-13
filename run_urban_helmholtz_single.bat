@echo off
REM ===========================================================================
REM  URBAN ENGINE (float64 dense Hessian + strong-Wolfe line search)
REM  fig:nlp-ssbroyden  <-  pinn_helmholtz_2d.py  (config "low": a1=1,a2=4,k=1)
REM
REM  Single-run Helmholtz figure. Same protocol as the thesis inhouse baseline
REM  (helmholtz_low_ssbroyden_identity_*); the ONLY change is the QN core,
REM  swapped to Urban float64 + strong Wolfe via --engine urban. Own terminal.
REM
REM  Output: BVP\results\helmholtz_low_ssbroyden_urban_identity_<timestamp>\
REM ===========================================================================
cd /d "%~dp0"

pushd BVP\two_d
python pinn_helmholtz_2d.py --engine urban
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  Helmholtz single run (urban engine) done.
echo  New folder: BVP\results\helmholtz_low_ssbroyden_urban_identity_*\
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
