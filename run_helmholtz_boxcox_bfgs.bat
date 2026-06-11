@echo off
REM Box-Cox sweeps on the 2D Helmholtz benchmark, BOTH optimizer variants, with
REM the corrected solver (transform now gated to the QN phase only -- identity
REM during the Adam warm-up, matching CFGS/vacuum-GS and the thesis protocol).
REM
REM Why both: the earlier Helmholtz sweeps applied the transform during Adam too,
REM which throttles the Adam phase for small lambda (g'=J^{lambda-1} is small at
REM J>>1) and confounds "small lambda hurts". With that fixed we must re-run the
REM SSBroyden sweep as well, not just add BFGS -- otherwise the BFGS-vs-SSBroyden
REM contrast is not on equal footing.
REM
REM   - SSBroyden (self-scaled): expect transform ~redundant, as on CFGS.
REM   - BFGS (not self-scaled):  does small lambda help (redundancy was the SSB
REM     story) or still hurt (intrinsic anisotropy -> stronger claim)?
REM
REM Same dyadic lambda grid as CFGS (dense near 0, negative tail for the turnover,
REM 0.75 anchor). These are two heavy sweeps; run them in two terminals if you
REM want them in parallel (split the two python lines below).

cd /d "%~dp0"

pushd BVP\two_d
python boxcox_sweep_2d_helmholtz.py --qn-variant ssbroyden --lambdas -1 -0.5 -0.25 -0.125 0 0.125 0.25 0.5 0.75 1
set RC=%ERRORLEVEL%
if not "%RC%"=="0" ( popd & goto :fail )
python boxcox_sweep_2d_helmholtz.py --qn-variant bfgs --lambdas -1 -0.5 -0.25 -0.125 0 0.125 0.25 0.5 0.75 1
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  Helmholtz Box-Cox sweeps (ssbroyden + bfgs) completed.
echo  New folders in BVP\results\:
echo    - helmholtz2d_a1_4_k1_boxcox_finesweep_ssbroyden_*/
echo    - helmholtz2d_a1_4_k1_boxcox_finesweep_bfgs_*/
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
