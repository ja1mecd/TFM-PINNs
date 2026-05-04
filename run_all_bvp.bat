@echo off
REM Run all BVP scripts (1D + 2D) sequentially.
REM Stops at the first failure (set %ERRORLEVEL% non-zero).

cd /d "%~dp0"

call :run_one BVP\one_d pinn_bvpsolver_l2.py             || goto :fail
call :run_one BVP\one_d pinn_bvpsolver_l2_BFGS.py        || goto :fail
call :run_one BVP\one_d pinn_ssbroyden_1d.py             || goto :fail
call :run_one BVP\one_d boxcox_sweep_1d.py               || goto :fail
call :run_one BVP\one_d architecture_sweep.py            || goto :fail
call :run_one BVP\two_d pinn_bvpsolver2d_BFGS.py         || goto :fail
call :run_one BVP\two_d pinn_ssbroyden_2d.py             || goto :fail

echo.
echo ============================================================
echo  All scripts completed successfully.
echo  Figures: BVP\one_d\figures\, BVP\one_d\architecture_heatmap*.png
echo  Results: BVP\results\, BVP\models\
echo ============================================================
exit /b 0

:run_one
echo.
echo ============================================================
echo  Running %~1\%~2
echo ============================================================
pushd "%~1"
python "%~2"
set RC=%ERRORLEVEL%
popd
exit /b %RC%

:fail
echo.
echo *** Script failed. Aborting. ***
exit /b 1
