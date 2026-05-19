import math

import pytest

from interpolation_stats import CellResult, SweepResult, aggregate
from interpolation_stats import save_json, load_json
from interpolation_stats import to_latex_summary


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


@pytest.mark.unit
def test_aggregate_handles_zero_linf_without_crashing():
    cells = [
        CellResult(layers=1, neurons=5, seed=42, linf=0.0, l2=0.0,
                   train_time_s=1.0, epochs_run=10),
        CellResult(layers=1, neurons=5, seed=43, linf=0.0, l2=0.0,
                   train_time_s=1.0, epochs_run=10),
    ]
    sweep = aggregate(
        activation="Tanh", layers=[1], neurons=[5], cells=cells,
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    # mean linf == 0 -> clamped to machine_eps; log10(1.19e-7) < -0.5
    # so the cell is NOT flagged as failed.
    assert sweep.n_failed[0][0] == 0


@pytest.mark.unit
def test_json_round_trip(tmp_path):
    sweep = aggregate(
        activation="ReLU", layers=[1], neurons=[5],
        cells=_cells_two_by_one(),
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    path = tmp_path / "relu.json"
    save_json(sweep, str(path))
    loaded = load_json(str(path))
    assert loaded == sweep


@pytest.mark.unit
def test_load_json_raises_valueerror_on_corrupt_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json")
    with pytest.raises(ValueError, match="Failed to load SweepResult"):
        load_json(str(bad))


@pytest.mark.unit
def test_latex_summary_has_row_per_activation():
    tanh = aggregate(
        activation="Tanh", layers=[1], neurons=[5],
        cells=_cells_two_by_one(),
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    relu = aggregate(
        activation="ReLU", layers=[1], neurons=[5],
        cells=_cells_two_by_one(),
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    tex = to_latex_summary([tanh, relu])
    assert r"\begin{tabular}" in tex
    assert "Tanh" in tex and "ReLU" in tex
    # best cell for the 1x1 grid is (L=1, W=5)
    assert "(1, 5)" in tex
    # mean+-std rendered with \pm and scientific notation
    assert r"\pm" in tex
    # _cells_two_by_one(): linf {1e-2,3e-2} -> mean 2e-2 std 1e-2;
    #   l2 {2e-3,4e-3} -> mean 3e-3 std 1e-3. Shared-exponent, math mode.
    assert r"$(2.00 \pm 1.00)\times 10^{-2}$" in tex   # linf cell
    assert r"$(3.00 \pm 1.00)\times 10^{-3}$" in tex   # l2 cell
    assert "2.00" in tex       # mean time/cell column
    assert "0/1" in tex        # failed / total cells
    assert tex.count(r"\\") >= 3  # header row + two data rows


@pytest.mark.unit
def test_latex_summary_escapes_activation_name():
    sw = aggregate(
        activation="Leaky_ReLU", layers=[1], neurons=[5],
        cells=_cells_two_by_one(),
        failure_log_threshold=-0.5, machine_eps=1.1920929e-7,
    )
    tex = to_latex_summary([sw])
    assert r"Leaky\_ReLU" in tex
    assert "Leaky_ReLU" not in tex.replace(r"Leaky\_ReLU", "")


@pytest.mark.unit
def test_fmt_pm_is_math_mode_and_handles_nonfinite():
    from interpolation_stats import _fmt_pm
    s = _fmt_pm(2e-2, 1e-2)
    assert s.startswith("$") and s.endswith("$")     # math-mode delimited
    assert r"\times 10^{-2}" in s
    assert "e-0" not in s                              # no raw sci-notation
    assert _fmt_pm(float("inf"), float("inf")) == r"$\mathrm{n/a}$"
    assert _fmt_pm(float("nan"), 0.0) == r"$\mathrm{n/a}$"
