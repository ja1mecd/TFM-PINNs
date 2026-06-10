import math

import pytest

from activation_stats_bvp import (
    METRICS,
    PRIMARY_METRIC,
    BVPCellResult,
    BVPSweepResult,
    aggregate,
    load_json,
    save_json,
    to_latex_summary,
)


def _cells_two_by_one():
    # Grid: layers=[1], neurons=[5]; two seeds.
    return [
        BVPCellResult(layers=1, neurons=5, seed=42, sol_linf=1e-2,
                      sol_rel_l2=2e-2, residual_l2=5.0,
                      train_time_s=1.0, epochs_run=100),
        BVPCellResult(layers=1, neurons=5, seed=43, sol_linf=3e-2,
                      sol_rel_l2=4e-2, residual_l2=7.0,
                      train_time_s=3.0, epochs_run=120),
    ]


@pytest.mark.unit
def test_aggregate_means_and_std():
    sweep = aggregate(
        activation="Tanh", layers=[1], neurons=[5],
        cells=_cells_two_by_one(),
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
        wavenumber=1.0,
    )
    assert isinstance(sweep, BVPSweepResult)
    assert sweep.wavenumber == pytest.approx(1.0)
    assert sweep.sol_linf_mean[0][0] == pytest.approx(2e-2)
    # population std of {1e-2, 3e-2} = 1e-2
    assert sweep.sol_linf_std[0][0] == pytest.approx(1e-2)
    assert sweep.sol_rel_l2_mean[0][0] == pytest.approx(3e-2)
    assert sweep.residual_l2_mean[0][0] == pytest.approx(6.0)
    assert sweep.time_mean[0][0] == pytest.approx(2.0)
    assert sweep.n_failed[0][0] == 0
    assert sweep.activation == "Tanh"


@pytest.mark.unit
def test_primary_metric_drives_failure_flag():
    # sol_linf mean ~ 0.55 -> log10 > -0.5 -> failed, regardless of others.
    cells = [
        BVPCellResult(layers=1, neurons=5, seed=42, sol_linf=0.5,
                      sol_rel_l2=0.9, residual_l2=160.0,
                      train_time_s=1.0, epochs_run=50),
        BVPCellResult(layers=1, neurons=5, seed=43, sol_linf=0.6,
                      sol_rel_l2=0.9, residual_l2=160.0,
                      train_time_s=1.0, epochs_run=50),
    ]
    sweep = aggregate(
        activation="Softmax", layers=[1], neurons=[5], cells=cells,
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    assert PRIMARY_METRIC == "sol_linf"
    assert sweep.n_failed[0][0] == 1


@pytest.mark.unit
def test_timing_ignores_nan_from_failed_run():
    cells = [
        BVPCellResult(layers=1, neurons=5, seed=42, sol_linf=float("inf"),
                      sol_rel_l2=float("inf"), residual_l2=float("inf"),
                      train_time_s=float("nan"), epochs_run=0),
        BVPCellResult(layers=1, neurons=5, seed=43, sol_linf=1e-2,
                      sol_rel_l2=2e-2, residual_l2=5.0,
                      train_time_s=4.0, epochs_run=120),
    ]
    sweep = aggregate(
        activation="ReLU", layers=[1], neurons=[5], cells=cells,
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    # only the finite 4.0 contributes
    assert sweep.time_mean[0][0] == pytest.approx(4.0)


@pytest.mark.unit
def test_json_round_trip(tmp_path):
    sweep = aggregate(
        activation="Tanh", layers=[1], neurons=[5],
        cells=_cells_two_by_one(),
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    path = tmp_path / "sweep.json"
    save_json(sweep, str(path))
    loaded = load_json(str(path))
    assert loaded == sweep


@pytest.mark.unit
def test_metrics_registry_attrs_exist_on_sweep():
    sweep = aggregate(
        activation="Tanh", layers=[1], neurons=[5],
        cells=_cells_two_by_one(),
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    for meta in METRICS.values():
        assert hasattr(sweep, meta["mean_attr"])
        assert hasattr(sweep, meta["std_attr"])


@pytest.mark.unit
def test_latex_summary_has_table_and_best_cell():
    sweep = aggregate(
        activation="Tanh", layers=[1], neurons=[5],
        cells=_cells_two_by_one(),
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    tex = to_latex_summary([sweep])
    # Split into two narrower tables (the combined 7-column table is too wide).
    assert tex.count(r"\begin{table}") == 2 and tex.count(r"\end{table}") == 2
    assert r"\label{tab:bvp-activation-summary}" in tex
    assert r"\label{tab:bvp-activation-summary-diag}" in tex
    assert "Tanh" in tex
    assert "(1, 5)" in tex  # only cell, so it is the best
    assert r"(\pi)^2\sin(\pi x)" in tex  # k=1 caption drops the unit coefficient


@pytest.mark.unit
def test_load_defaults_wavenumber_for_legacy_json(tmp_path):
    # A pre-wavenumber JSON (field absent) must still load, defaulting to k=1.
    import json
    from dataclasses import asdict
    sweep = aggregate(
        activation="Tanh", layers=[1], neurons=[5],
        cells=_cells_two_by_one(),
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
        wavenumber=1.0,
    )
    payload = asdict(sweep)
    del payload["wavenumber"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))
    assert load_json(str(path)).wavenumber == pytest.approx(1.0)


@pytest.mark.unit
def test_with_rell2_failures_majority_rule():
    from activation_stats_bvp import with_rell2_failures

    def cell(seed, rel):
        return BVPCellResult(layers=1, neurons=5, seed=seed, sol_linf=rel,
                             sol_rel_l2=rel, residual_l2=1.0,
                             train_time_s=1.0, epochs_run=50)

    # 3 of 4 seeds below 1e-2: strict majority succeeds -> not failed.
    ok = aggregate(
        activation="Tanh", layers=[1], neurons=[5],
        cells=[cell(42, 1e-4), cell(43, 5e-3), cell(44, 9e-3), cell(45, 0.5)],
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    assert with_rell2_failures(ok).n_failed[0][0] == 0

    # 2 of 4 (exactly half) succeed: no strict majority -> failed.
    tie = aggregate(
        activation="Tanh", layers=[1], neurons=[5],
        cells=[cell(42, 1e-4), cell(43, 5e-3), cell(44, 0.5), cell(45, 0.9)],
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    assert with_rell2_failures(tie).n_failed[0][0] == 1

    # A new object is returned; the input sweep is left untouched.
    assert with_rell2_failures(tie) is not tie
