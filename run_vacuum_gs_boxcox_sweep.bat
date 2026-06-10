@echo off
REM Box-Cox sweep on the vacuum Grad-Shafranov (psi = R^2) validation problem
REM (thesis sec. 4.3.2). Probe to confirm whether this well-conditioned instance
REM of the current-free Grad-Shafranov operator is stagnation-limited like the
REM CFGS benchmark (sec. 4.3.4), i.e. whether small lambda helps monotonically.
REM
REM 11 exponents x 3 seeds, standardised protocol: fixed 2000-epoch Adam warm-up
REM -> SSBroyden, Box-Cox engaged from the start of the QN phase, QN-phase early
REM stopping. Heavier than the single validation run; expect it to take a while.

cd /d "%~dp0"

call :run_one BVP\two_d boxcox_sweep_2d_vacuum_gs.py || goto :fail

echo.
echo ============================================================
echo  Vacuum-GS Box-Cox sweep completed.
echo  New folder in BVP\results\:
echo    - vacuum_gs_ssbroyden_boxcox_finesweep_*/
echo        boxcox_sweep_2d_vacuum_gs.png
echo        summary_table.txt
echo        raw_histories.npz
echo  Send the folder name back so the result can be read into the thesis.
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
