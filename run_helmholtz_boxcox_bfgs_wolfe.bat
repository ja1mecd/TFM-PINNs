@echo off
REM Box-Cox sweep on 2D Helmholtz (a1,a2)=(1,4) with PLAIN BFGS under the
REM Urban et al. (2025) protocol -- the faithful replication of their Fig. 6
REM setup, where BFGS+log/sqrt beats BFGS+identity by ~2 orders of magnitude.
REM
REM What changed vs the earlier BFGS sweep (and why it should now reproduce
REM their result):
REM   1. --qn-line-search strong_wolfe : curvature-condition line search
REM      (cubic interpolation, steps can exceed 1). Guarantees y.s > 0 so the
REM      inverse Hessian stays positive definite and is NEVER reset -- the
REM      old Armijo search reset H to identity on every curvature failure,
REM      which under log/sqrt (gradient ~ J^(lambda-1)) degraded BFGS to
REM      gradient descent.
REM   2. Urban Adam phase: 5000 epochs, lr0=5e-3 with exp decay 0.98^(t/1000),
REM      beta1=0.99, eps=1e-20 (their 2DH config) instead of 2000 @ 1e-3.
REM   3. Net 20x20 (their 2DH(1,4) row in Table 4) instead of the thesis
REM      standard 3x32. DELIBERATE deviation -- this is a replication run,
REM      not part of the standardised thesis sweep series.
REM   4. Budget 15000 total (5000 Adam + up to 10000 QN). Their Fig. 6 runs
REM      20k QN, but the J vs J_log separation is clear well before 10k QN.
REM      --es-patience 1000 keeps controlled early stopping without cutting
REM      transformed runs during their slow first phase (old default 300
REM      stopped runs after ~1300 QN steps).
REM   5. --n-collocation 12500 with the 0.8 train split = 10000 training
REM      points, matching their batch size.
REM
REM Lambda grid: same 10 values as the 2026-06-11 SSBroyden fine sweep, so the
REM two tables are directly comparable. Expect ~1.5-2x cost per QN epoch vs
REM Armijo (Wolfe needs gradient evaluations in the line search). If runtime
REM is a problem, trim the negative lambdas first -- the SSBroyden sweep
REM showed they only continue the monotone trend.
REM
REM Results land in BVP\results\helmholtz2d_a1_4_k1_boxcox_finesweep_bfgswolfe_*/

cd /d "%~dp0"

pushd BVP\two_d
python boxcox_sweep_2d_helmholtz.py ^
  --qn-variant bfgs ^
  --qn-line-search strong_wolfe ^
  --adam-schedule urban_exp ^
  --adam-beta1 0.99 ^
  --adam-eps 1e-20 ^
  --lr 5e-3 ^
  --hidden 20 20 ^
  --epochs 15000 ^
  --adam-epochs 5000 ^
  --n-collocation 12500 ^
  --es-patience 1000 ^
  --patience 1000 ^
  --lambdas -1 -0.5 -0.25 -0.125 0 0.125 0.25 0.5 0.75 1
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  Helmholtz Box-Cox BFGS (strong Wolfe, Urban protocol)
echo  sweep completed. New folder in BVP\results\:
echo    - helmholtz2d_a1_4_k1_boxcox_finesweep_bfgswolfe_*/
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
