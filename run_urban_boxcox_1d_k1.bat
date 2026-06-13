@echo off
REM ===========================================================================
REM  URBAN ENGINE (float64 dense Hessian + strong-Wolfe line search)
REM  1D BVP Box-Cox sweep at k=1 (easier wavenumber) with --engine urban.
REM  Companion to run_urban_boxcox_1d_2darch.bat (which is k=4).
REM
REM  Output: BVP\results\bvp1d_k1_boxcox_2darch_urban_ssbroyden_ls-armijo_<ts>\
REM ===========================================================================
cd /d "%~dp0"

pushd BVP\one_d
python boxcox_sweep_1d_2darch.py --wavenumber 1 --qn-variant ssbroyden --engine urban
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  1D k=1 Box-Cox sweep (urban engine) done.
echo  New folder: BVP\results\bvp1d_k1_boxcox_2darch_urban_ssbroyden_*\
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
