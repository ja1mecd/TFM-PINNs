@echo off
REM Resume run_thesis_followup.bat from step 4. Steps 1-3 (hessian_spectrum,
REM boxcox_diagnostic, optimiser_comparison) already completed; do not re-run.
REM
REM IMPORTANT — survival across SSH/RDP drops:
REM     start /B cmd /c run_thesis_followup_resume.bat > resume.log 2>&1
REM otherwise a network blip will kill the parent shell and orphan the work.
REM Even WITH start /B, every (lambda, seed) pair from now on is checkpointed
REM under <RESUME_DIR>\pairs\, so re-launching this same .bat picks up
REM exactly where it left off.

cd /d "%~dp0"

REM ---- Persistent output dirs that the sweep scripts will checkpoint into ----
set RESUME_DIR_4A=..\results\bvp1d_k4_boxcox_finesweep_ssbroyden_resumed
set RESUME_DIR_4B=..\results\bvp1d_k4_boxcox_finesweep_delayed_ssbroyden_resumed

REM ============================================================================
REM Step 4a — fine-grained Box-Cox sweep, identity-engagement schedule.
REM
REM Prior run died after (lambda=0.4, seed=45) but produced NO disk
REM checkpoints (per-pair persistence was added afterwards). The two
REM invocations below split the remaining work so that already-done pairs are
REM not re-run unnecessarily:
REM     1) (lambda=0.4, seed=46): the only seed of lambda=0.4 not yet finished
REM     2) (lambda=0.5..1.0, seeds 42..46): fully untouched lambdas
REM Both target the same --resume-dir, so per-pair .npz files accumulate in
REM one place and a third interruption inside this resume run will be caught
REM by the new checkpointing layer.
REM ============================================================================

call :run_one_args BVP\one_d boxcox_sweep_1d_finegrained.py ^
    "--resume-dir %RESUME_DIR_4A% --lambdas 0.4 --seeds 46"                                           || goto :fail

call :run_one_args BVP\one_d boxcox_sweep_1d_finegrained.py ^
    "--resume-dir %RESUME_DIR_4A% --lambdas 0.5 0.6 0.7 0.8 0.9 1.0 --seeds 42 43 44 45 46"           || goto :fail

REM (Optional) re-do the lost pairs (lambda=0.0..0.3 all seeds and lambda=0.4
REM seeds 42..45). Uncomment if you want the final figure to include them;
REM otherwise the sweep will plot 26 pairs over 7 lambdas instead of 55 over 11:
REM call :run_one_args BVP\one_d boxcox_sweep_1d_finegrained.py ^
REM     "--resume-dir %RESUME_DIR_4A% --lambdas 0.0 0.1 0.2 0.3 --seeds 42 43 44 45 46"   || goto :fail
REM call :run_one_args BVP\one_d boxcox_sweep_1d_finegrained.py ^
REM     "--resume-dir %RESUME_DIR_4A% --lambdas 0.4 --seeds 42 43 44 45"                  || goto :fail

REM ============================================================================
REM Step 4b — same sweep but with delayed Box-Cox engagement (J<1.0 trigger).
REM ============================================================================

call :run_one_args BVP\one_d boxcox_sweep_1d_finegrained.py ^
    "--resume-dir %RESUME_DIR_4B% --engage-threshold 1.0"                                             || goto :fail

REM ============================================================================
REM Steps 5, 6 — 2D experiments (no per-pair checkpointing yet, run as-is).
REM ============================================================================

call :run_one BVP\two_d pinn_poisson_2d_unitsquare.py                                                 || goto :fail
call :run_one BVP\two_d boxcox_sweep_2d_helmholtz.py                                                  || goto :fail

echo.
echo ============================================================
echo  Resume completed. Per-pair checkpoints in:
echo    %RESUME_DIR_4A%\pairs\
echo    %RESUME_DIR_4B%\pairs\
echo  Aggregated figures + summary tables next to them.
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
