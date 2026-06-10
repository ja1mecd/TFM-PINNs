@echo off
REM Chapter 4.3 follow-up runs (Windows GPU machine).
REM
REM Closes two gaps:
REM   1. Fig 4.5 (Poisson): rerun so the figure carries the new solution-L2 +
REM      residual-L2 convergence panel and saves the field into raw_histories.npz.
REM   2. Sec 4.3.2 (vacuum Grad-Shafranov, psi = R^2): first run of the new
REM      validation solver on the box R in [1,2], Z in [-1,1].
REM
REM Fig 4.6 (Helmholtz) does NOT need a rerun: it was regenerated from the saved
REM history.npz/fields.npz by two_d\regenerate_helmholtz_figure.py.
REM
REM Both runs use the standardised 3x32 Tanh network, fixed 2000-epoch Adam
REM warm-up -> SSBroyden, identity loss, QN-phase early stopping. Stops at the
REM first failure.

cd /d "%~dp0"

call :run_one BVP\two_d pinn_poisson_2d_unitsquare.py   || goto :fail
call :run_one BVP\two_d pinn_vacuum_gs_2d.py            || goto :fail

echo.
echo ============================================================
echo  Chapter 4.3 follow-up runs completed successfully.
echo  New per-run sub-directories in BVP\results\:
echo    - poisson2d_unitsquare_ssbroyden_identity_*/poisson2d_results.png
echo    - vacuum_gs_ssbroyden_identity_*/vacuum_gs_results.png  (+ summary.json)
echo.
echo  Send the new timestamped folder names back so the thesis figure paths
echo  and the Sec 4.3.2 numbers/table can be filled in.
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
