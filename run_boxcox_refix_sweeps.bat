@echo off
REM Re-run the 2D Box-Cox sweeps with the FIXED (offset-free) Box-Cox transform.
REM
REM Background: the old transform g(J) = (J^lambda - 1)/lambda carries a
REM -1/lambda constant. It is mathematically inert (zero gradient) but in float32
REM it pins the objective near -1/lambda, so once J falls below the ULP (~1.2e-7
REM near 1) the tiny residual is rounded away in the SSBroyden line-search value
REM comparison and the QN phase stalls near lambda=1. The transform is now the
REM offset-free g(J) = (J+eps)^lambda / lambda (identical gradient and rank-one
REM Hessian term, positive-valued, float32-resolvable).
REM
REM Expected: Helmholtz barely moves (it floors at ~5e-6, above the ULP), but
REM CFGS's lambda=1 end should drop from ~8e-7 toward ~1e-11, reshaping the
REM lambda curve. The vacuum-GS sweep becomes meaningful for the first time.
REM
REM Each sweep is 11 lambdas x 3 seeds -> heavy. Stops at the first failure.

cd /d "%~dp0"

call :run_one BVP\two_d boxcox_sweep_2d_cfgs.py       || goto :fail
call :run_one BVP\two_d boxcox_sweep_2d_helmholtz.py  || goto :fail
call :run_one BVP\two_d boxcox_sweep_2d_vacuum_gs.py  || goto :fail

echo.
echo ============================================================
echo  Box-Cox re-fix sweeps completed. New folders in BVP\results\:
echo    - cfgs_urban_SSBroyden2_boxcox_finesweep_*/
echo    - helmholtz2d_a1_4_k1_boxcox_finesweep_*/   (name per script)
echo    - vacuum_gs_ssbroyden_boxcox_finesweep_*/
echo  Each has summary_table.txt + boxcox_sweep_2d_*.png + raw_histories.npz.
echo  Send the three folder names back to compare against the thesis tables.
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
