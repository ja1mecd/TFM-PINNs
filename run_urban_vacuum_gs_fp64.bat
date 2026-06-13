@echo off
REM ===========================================================================
REM  URBAN ENGINE + FLOAT64 NETWORK
REM  fig:vacuum-gs-results / tab:vacuum-gs-summary  <-  pinn_vacuum_gs_2d.py
REM
REM  Why this run exists:
REM  The standard urban run keeps a float64 inverse-Hessian but a FLOAT32
REM  network. The vacuum GS exact solution psi = R^2 lies exactly in the hard
REM  ansatz space, so a float32 network reaches it to the single-precision
REM  floor and the solution-error metric reads exactly 0 across every seed
REM  (BVP\results\vacuum_gs_ssbroyden_urban_identity_20260613_152052\). The
REM  residual J is fine; only the solution error is floored.
REM
REM  --fp64 runs the network and residual in double precision (torch default
REM  dtype + a dtype-matched evaluation grid), so the solution error resolves
REM  below ~1e-7 and the table gets a real number instead of "<1e-7".
REM
REM  Same standardised protocol otherwise (3x32 Tanh, 2000 Adam warm-up,
REM  10000-epoch cap with QN early stopping, seeds 42/43/44, identity loss).
REM
REM  Output: BVP\results\vacuum_gs_ssbroyden_urban_fp64_identity_<timestamp>\
REM ===========================================================================
cd /d "%~dp0"

pushd BVP\two_d
python pinn_vacuum_gs_2d.py --engine urban --fp64
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  Vacuum Grad-Shafranov 2D (urban + fp64) done.
echo  New folder: BVP\results\vacuum_gs_ssbroyden_urban_fp64_*\
echo  Check summary.json: final_sol_l2_mean should now be nonzero.
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
