@echo off
REM Extension of the Table 4.4 / Fig 4.3 optimiser comparison from 5 to 20
REM seeds: runs the 15 NEW seeds 47-61 on all four pipelines with the exact
REM thesis protocol (k=4, 5000-epoch budget, 2000-epoch fixed Adam warm-up,
REM 3x32 Tanh, QN early stopping) -- all script defaults, only --seeds set.
REM The existing 5-seed run (seeds 42-46, bvp1d_k4_optim_compare_20260521_175309)
REM is NOT rerun; merge afterwards with:
REM   python BVP\one_d\merge_optim_runs.py <dir_42-46> <dir_47-61> --portrait

cd /d "%~dp0"

pushd BVP\one_d
python optimiser_comparison_1d.py --seeds 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61
set RC=%ERRORLEVEL%
popd

if %RC% neq 0 (
    echo.
    echo *** optimiser_comparison_1d.py failed. ***
    exit /b 1
)

echo.
echo ============================================================
echo  Done. New run directory under BVP\results\:
echo    bvp1d_k4_optim_compare_^<timestamp^>
echo  Copy it back next to the seed 42-46 run and merge with
echo  merge_optim_runs.py to regenerate Table 4.4 / Fig 4.3.
echo ============================================================
exit /b 0
