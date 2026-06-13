@echo off
REM ===========================================================================
REM  Line-search ablation of the 1D BVP Box-Cox sweep at k=1.
REM
REM  Question this answers
REM  ---------------------
REM  Every Box-Cox sweep we run picks identity (lambda=1) and buries log
REM  (lambda=0), the opposite of Urban et al., who report log/sqrt helping.
REM  Hypothesis: a loss transform only helps when the UNTRANSFORMED baseline
REM  STALLS. Our self-scaled Broyden + Armijo stack does not starve, so the
REM  gradient-inflating transforms (low lambda) have nothing to rescue and
REM  only their downside (an exploding gradient that corrupts the Broyden
REM  secant update) shows up -> identity wins. Urban's transforms help because
REM  their baseline stalls in the small-residual regime.
REM
REM  Test: hold EVERYTHING fixed (3x32 Tanh, 10k collocation, 2000 Adam ->
REM  SSBroyden, seeds 42/43/44, lambda 0..1) and sweep ONLY the line-search
REM  strength, from Urban-faithful strong-Wolfe down to no line search at all.
REM  Prediction: as the line search weakens, identity stalls and the
REM  identity-minus-log gap flips sign -- log/sqrt overtake identity, exactly
REM  reproducing the published behaviour inside our own pipeline.
REM
REM  Four regimes (strong -> crippled), each a full 11-lambda x 3-seed sweep:
REM    1. strong_wolfe                 Urban-faithful upper bound; baseline
REM                                    never starves -> identity should win big.
REM    2. armijo                       The historical default (= the existing
REM                                    k=1 run); control, identity wins.
REM    3. armijo --max-ls 3 --no-reset-on-fail
REM                                    Crippled Armijo: barely backtracks and no
REM                                    H->identity safety reset -> baseline weak.
REM    4. none                         Fixed unit step, no line search at all;
REM                                    most crippled -> baseline stalls hardest.
REM
REM  Each regime writes its own timestamped folder, tagged with the line-search
REM  regime in the name:
REM    BVP\results\bvp1d_k1_boxcox_2darch_ssbroyden_ls-<tag>_<timestamp>\
REM      - boxcox_sweep_1d_2darch.png
REM      - boxcox_sweep_1d_2darch_conditioned.png
REM      - summary_table.txt   (header records line_search=<tag>)
REM      - raw_histories.npz
REM
REM  Compare the "Final residual vs lambda" / "Solution error vs lambda" panels
REM  across the four folders: watch the minimum march from lambda~0.7 (strong)
REM  toward lambda~0 (crippled).
REM
REM  Cost note: 4 sweeps x 33 runs x up to 10k epochs. QN-phase early stopping
REM  ends most runs early, but budget for an overnight run. Drop a regime by
REM  commenting out its call below if you want a faster first pass.
REM ===========================================================================

cd /d "%~dp0"

call :run BVP\one_d boxcox_sweep_1d_2darch.py "--wavenumber 1 --line-search strong_wolfe" || goto :fail
call :run BVP\one_d boxcox_sweep_1d_2darch.py "--wavenumber 1 --line-search armijo" || goto :fail
call :run BVP\one_d boxcox_sweep_1d_2darch.py "--wavenumber 1 --line-search armijo --max-ls 3 --no-reset-on-fail" || goto :fail
call :run BVP\one_d boxcox_sweep_1d_2darch.py "--wavenumber 1 --line-search none" || goto :fail

echo.
echo ============================================================
echo  1D Box-Cox k=1 line-search ablation completed.
echo  Four folders in BVP\results\:
echo    - bvp1d_k1_boxcox_2darch_ssbroyden_ls-strong_wolfe_*\
echo    - bvp1d_k1_boxcox_2darch_ssbroyden_ls-armijo_*\
echo    - bvp1d_k1_boxcox_2darch_ssbroyden_ls-armijo-maxls3-noreset_*\
echo    - bvp1d_k1_boxcox_2darch_ssbroyden_ls-none_*\
echo  Compare the residual/solution-vs-lambda panels across the four.
echo ============================================================
exit /b 0

:run
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
echo *** Script failed (exit %RC%). Aborting remaining regimes. ***
exit /b 1
