@echo off
REM Box-Cox sweep on CFGS with BFGS (a NON-self-scaled quasi-Newton).
REM
REM This is the experiment that should reproduce + extend Urban et al. Table 2.
REM Their loss-transform result (sqrt/log improve accuracy ~1 order) is measured
REM on BFGS, NOT on the self-scaled SSBroyden. With a self-scaled method the
REM transform is redundant (our SSBroyden2 sweep is flat across lambda); with
REM plain BFGS the transform's Hessian rescaling g'(J) = J^{lambda-1} has a real
REM job to do, so small lambda should genuinely help.
REM
REM Uses the FIXED offset-free Box-Cox transform, so BFGS+log no longer stalls on
REM the float32 line-search precision issue. lambda=0 is log, lambda=0.5 is the
REM sqrt-equivalent, so this 11-point sweep contains Urban's two points and the
REM interpolation between them.
REM
REM 11 lambdas x 3 seeds. Pair the result with the existing SSBroyden2 sweep
REM (cfgs_urban_SSBroyden2_boxcox_finesweep_20260611_130536) for the contrast.

REM Lambda grid: dyadic spacing dense near 0 (where the transform turns from weak
REM to strong and where Urban's optimum sat), extended into NEGATIVE lambda to
REM find the help->hurt turnover (more amplification g'=J^{lambda-1} eventually
REM over-amplifies and destabilises the line search), plus one anchor at 0.75
REM because identity->sqrt is a real ~8x effect for BFGS (Urban Table 2).
REM Expect very negative lambda to throw line-search failures -- that is signal.

cd /d "%~dp0"

pushd BVP\two_d
python boxcox_sweep_2d_cfgs.py --variant BFGS --lambdas -1 -0.5 -0.25 -0.125 0 0.125 0.25 0.5 0.75 1
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" goto :fail

echo.
echo ============================================================
echo  CFGS BFGS Box-Cox sweep completed.
echo  New folder in BVP\results\:
echo    - cfgs_urban_BFGS_boxcox_finesweep_*/
echo        summary_table.txt + boxcox_sweep_2d_cfgs.png + raw_histories.npz
echo  Send the folder name back to read the lambda curve.
echo ============================================================
exit /b 0

:fail
echo.
echo *** Script failed (exit %RC%). ***
exit /b 1
