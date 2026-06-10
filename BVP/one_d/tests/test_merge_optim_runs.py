"""Unit tests for merge_optim_runs.merge_results."""

import numpy as np
import pytest

from optimiser_comparison_1d import PipelineResult, SeedRun
from merge_optim_runs import merge_results


def _run(seed: int, val: float = 1.0) -> SeedRun:
    return SeedRun(
        seed=seed,
        J_val_history=np.array([val, val / 2]),
        sol_l2_history=np.array([val, val / 4]),
        final_J_val=val / 2,
        final_pde_l2=val,
        final_sol_l2=val / 4,
        final_sol_rel_l2=val / 3,
    )


def _result(pipeline: str, seeds: tuple[int, ...]) -> PipelineResult:
    return PipelineResult(
        pipeline=pipeline, seeds=tuple(_run(s) for s in seeds)
    )


@pytest.mark.unit
def test_merges_seeds_per_pipeline_in_order():
    a = (_result("adam", (42, 43)), _result("adam_bfgs", (42, 43)))
    b = (_result("adam", (47, 48)), _result("adam_bfgs", (47, 48)))
    merged = merge_results(a, b)
    assert [r.pipeline for r in merged] == ["adam", "adam_bfgs"]
    assert [s.seed for s in merged[0].seeds] == [42, 43, 47, 48]
    assert [s.seed for s in merged[1].seeds] == [42, 43, 47, 48]


@pytest.mark.unit
def test_rejects_duplicate_seeds():
    a = (_result("adam", (42, 43)),)
    b = (_result("adam", (43, 44)),)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        merge_results(a, b)


@pytest.mark.unit
def test_rejects_mismatched_pipelines():
    a = (_result("adam", (42,)),)
    b = (_result("adam_bfgs", (47,)),)
    with pytest.raises(ValueError, match="[Pp]ipeline"):
        merge_results(a, b)


@pytest.mark.unit
def test_three_way_merge():
    a = (_result("adam", (42,)),)
    b = (_result("adam", (47,)),)
    c = (_result("adam", (50,)),)
    merged = merge_results(a, b, c)
    assert [s.seed for s in merged[0].seeds] == [42, 47, 50]


@pytest.mark.unit
def test_merge_preserves_run_data_immutably():
    a = (_result("adam", (42,)),)
    b = (_result("adam", (47,)),)
    merged = merge_results(a, b)
    # Originals untouched
    assert [s.seed for s in a[0].seeds] == [42]
    assert [s.seed for s in b[0].seeds] == [47]
    # Same SeedRun objects carried over, not copies with altered data
    assert merged[0].seeds[0] is a[0].seeds[0]
    assert merged[0].seeds[1] is b[0].seeds[0]
