@echo off
REM ===========================================================================
REM  URBAN ENGINE (float64 dense Hessian + strong-Wolfe line search)
REM  fig:bvp-1d-boxcox-sweep  <-  boxcox_sweep_1d_2darch.py
REM
REM  Box-Cox lambda sweep on the 1D BVP, k=4, SSBroyden, 3x32 Tanh. Same
REM  protocol and default lambda grid as the thesis inhouse baseline
REM  (bvp1d_k4_boxcox_2darch_ssbroyden_*); the ONLY change is the QN core,
REM  swapped to Urban float64 + strong Wolfe via --engine urban. Own terminal.
REM
REM  Note: under --engine urban the line search is strong Wolfe by construction,
REM  so the --line-search flag is inert for the QN core.
REM
REM  Output: BVP\results\bvp1d_k4_boxcox_2darch_urban_ssbroyden_ls-*_<timestamp>\
REM ===========================================================================
cd /d "%~dp0"

pushd BVP\one_d
python boxcox_sweep_1d_2darch.py --wavenumber 4 --qn-variant ssbroyden --engine urban
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  1D Box-Cox sweep (2D arch, urban engine) done.
echo  New folder: BVP\results\bvp1d_k4_boxcox_2darch_urban_ssbroyden_*\
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
