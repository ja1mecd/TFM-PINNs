"""Unit tests for the median-based aggregates of optimiser_comparison_1d."""

import numpy as np
import pytest

from optimiser_comparison_1d import (
    PipelineResult,
    SeedRun,
    median_iqr,
    write_summary,
)


@pytest.mark.unit
def test_median_iqr_odd_count():
    med, q1, q3 = median_iqr([1.0, 2.0, 3.0, 4.0, 100.0])
    assert med == 3.0
    assert q1 == 2.0
    assert q3 == 4.0


@pytest.mark.unit
def test_median_is_robust_to_one_outlier():
    clean = median_iqr([1.0, 2.0, 3.0])[0]
    spiked = median_iqr([1.0, 2.0, 3000.0])[0]
    assert clean == spiked == 2.0


@pytest.mark.unit
def test_median_iqr_rejects_empty():
    with pytest.raises(ValueError):
        median_iqr([])


def _run(seed: int, J: float, sol: float, rel: float) -> SeedRun:
    return SeedRun(
        seed=seed,
        J_val_history=np.array([J]),
        sol_l2_history=np.array([sol]),
        final_J_val=J,
        final_pde_l2=J,
        final_sol_l2=sol,
        final_sol_rel_l2=rel,
    )


@pytest.mark.unit
def test_write_summary_reports_median_and_min(tmp_path):
    # Three successes, one a shallow outlier: the median must ignore it
    # and the min must pick the deepest run.
    result = PipelineResult(
        pipeline="adam_ssbroyden",
        seeds=(
            _run(1, J=1e-5, sol=1e-5, rel=1.4e-5),
            _run(2, J=3e-5, sol=2e-5, rel=2.8e-5),
            _run(3, J=1e-2, sol=2e-3, rel=2.8e-3),   # shallow success
            _run(4, J=1e-7, sol=3e-6, rel=4.2e-6),   # deepest run
            _run(5, J=7e3, sol=1.8, rel=2.5),        # failure
        ),
    )
    out = tmp_path / "summary.txt"
    write_summary(
        results=(result,), out_path=str(out), k=4.0,
        total_epochs=-1, adam_warmup=2000, seeds=(1, 2, 3, 4, 5),
        rel_l2_threshold=0.01,
    )
    text = out.read_text()
    assert "median" in text
    # median J over the four successes = (1e-5 + 3e-5)/2 = 2e-5
    assert "2.0000e-05" in text
    # deepest residual must be reported
    assert "1.0000e-07" in text
    # success count
    assert "4/5" in text
