@echo off
REM Run the follow-up experiments to close the gaps identified in the
REM comparison with TFM-4 and address the Box-Cox interpretation in the
REM 1D BVP results of section 4.2.3.
REM
REM Order is chosen so that cheap diagnostic / sanity-check scripts run
REM first; the long sweeps are at the end. Stops at the first failure.

cd /d "%~dp0"

REM --- Cheap diagnostics --------------------------------------------------

call :run_one BVP\one_d hessian_spectrum_diagnostic.py             || goto :fail
call :run_one BVP\one_d boxcox_diagnostic.py                       || goto :fail

REM --- 1D follow-ups ------------------------------------------------------

call :run_one BVP\one_d optimiser_comparison_1d.py                 || goto :fail
call :run_one BVP\one_d boxcox_sweep_1d_finegrained.py             || goto :fail
REM Same sweep, but with delayed Box-Cox engagement once J<1.0:
call :run_one_args BVP\one_d boxcox_sweep_1d_finegrained.py "--engage-threshold 1.0" || goto :fail

REM --- 2D follow-ups (the regime where Box-Cox is supposed to help) -------

call :run_one BVP\two_d pinn_poisson_2d_unitsquare.py              || goto :fail
call :run_one BVP\two_d boxcox_sweep_2d_helmholtz.py               || goto :fail

echo.
echo ============================================================
echo  Follow-up suite completed successfully.
echo  Look in BVP\results\ for the per-run sub-directories.
echo  Key figures:
echo    - hessian_spectrum/hessian_spectrum_empirical.png
echo    - boxcox_diagnostic/boxcox_diagnostic.png
echo    - bvp1d_*_optim_compare_*/optimiser_comparison.png
echo    - bvp1d_*_boxcox_finesweep_*/boxcox_sweep_finegrained.png
echo    - bvp1d_*_boxcox_finesweep_delayed_*/boxcox_sweep_finegrained.png
echo    - poisson2d_unitsquare_*/poisson2d_results.png
echo    - helmholtz2d_*_boxcox_finesweep_*/boxcox_sweep_2d_helmholtz.png
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
