@echo off
REM Resume run_thesis_followup.bat from step 4. Steps 1-3 (hessian_spectrum,
REM boxcox_diagnostic, optimiser_comparison) already completed; their result
REM directories are under BVP\results\ and do not need to be regenerated.

cd /d "%~dp0"

call :run_one BVP\one_d boxcox_sweep_1d_finegrained.py                            || goto :fail
call :run_one_args BVP\one_d boxcox_sweep_1d_finegrained.py "--engage-threshold 1.0" || goto :fail
call :run_one BVP\two_d pinn_poisson_2d_unitsquare.py                             || goto :fail
call :run_one BVP\two_d boxcox_sweep_2d_helmholtz.py                              || goto :fail

echo.
echo ============================================================
echo  Resume (steps 4-7) completed successfully.
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
