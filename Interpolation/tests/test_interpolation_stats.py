import math

import pytest

from interpolation_stats import CellResult, SweepResult, aggregate


def _cells_two_by_one():
    # Grid: layers=[1], neurons=[5]; two seeds.
    return [
        CellResult(layers=1, neurons=5, seed=42, linf=1e-2, l2=2e-3,
                   train_time_s=1.0, epochs_run=100),
        CellResult(layers=1, neurons=5, seed=43, linf=3e-2, l2=4e-3,
                   train_time_s=3.0, epochs_run=120),
    ]


@pytest.mark.unit
def test_aggregate_means_and_std():
    sweep = aggregate(
        activation="Tanh",
        layers=[1],
        neurons=[5],
        cells=_cells_two_by_one(),
        failure_log_threshold=-0.5,
        machine_eps=1.1920929e-7,
    )
    assert isinstance(sweep, SweepResult)
    assert sweep.linf_mean[0][0] == pytest.approx(2e-2)
    # population std of {1e-2, 3e-2} = 1e-2
    assert sweep.linf_std[0][0] == pytest.approx(1e-2)
    assert sweep.l2_mean[0][0] == pytest.approx(3e-3)
    assert sweep.time_mean[0][0] == pytest.approx(2.0)
    assert sweep.n_failed[0][0] == 0
    assert sweep.activation == "Tanh"
    assert sweep.machine_eps == pytest.approx(1.1920929e-7)


@pytest.mark.unit
def test_aggregate_flags_failed_cell():
    cells = [
        CellResult(layers=1, neurons=5, seed=42, linf=1.7, l2=1.0,
                   train_time_s=1.0, epochs_run=50),
        CellResult(layers=1, neurons=5, seed=43, linf=1.6, l2=1.0,
                   train_time_s=1.0, epochs_run=50),
    ]
    sweep = aggregate(
        activation="Softmax", layers=[1], neurons=[5], cells=cells,
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    # log10(mean linf) = log10(1.65) > -0.5  -> failed
    assert sweep.n_failed[0][0] == 1
    assert math.isfinite(sweep.linf_mean[0][0])
