@echo off
REM Box-Cox sweep on the 1D BVP at k=1 (-u'' = pi^2 sin(pi x), u = sin(pi x)),
REM under the same 2D protocol as the existing k=4 run:
REM   3x32 Tanh, 10 000 collocation, 10 000 epochs (2000 Adam -> SSBroyden),
REM   seeds 42/43/44, lambda in {0, 0.1, ..., 1.0}.
REM Output goes to BVP\results\bvp1d_k1_boxcox_2darch_ssbroyden_<timestamp>\.

cd /d "%~dp0"

call :run_one_args BVP\one_d boxcox_sweep_1d_2darch.py "--wavenumber 1" || goto :fail

echo.
echo ============================================================
echo  1D Box-Cox k=1 sweep completed successfully.
echo  Look in BVP\results\bvp1d_k1_boxcox_2darch_ssbroyden_*\ for:
echo    - boxcox_sweep_1d_2darch.png
echo    - boxcox_sweep_1d_2darch_conditioned.png
echo    - summary_table.txt
echo    - raw_histories.npz
echo ============================================================
exit /b 0

:run_one_args
echo.
echo ============================================================
echo  Running %~1\%~2 %~3
echo ============================================================
pushd "%~1"
python "%~2" %~3
set RC=%ERRORLEVEL%
popd
exit /b %RC%

:fail
echo.
echo *** Script failed. Aborting. ***
exit /b 1
