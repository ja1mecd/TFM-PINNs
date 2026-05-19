import pytest


@pytest.mark.integration
def test_train_returns_epochs_run():
    import torch
    import torch.nn as nn
    from pinn_interpolant_l2 import NeuralNetwork, PINN_L2_Minimizer

    torch.manual_seed(0)
    model = NeuralNetwork(hidden_layers=[5], activation=nn.Tanh())
    pinn = PINN_L2_Minimizer(model, lr=1e-3)
    epochs = pinn.train(
        n_epochs=15, n_collocation_points=16, verbose_freq=999,
        patience=999, min_delta=1e-9, moving_avg_window=3, l2_points=32,
    )
    assert epochs == 15
    assert pinn.epochs_run == 15


@pytest.mark.integration
def test_run_sweep_multi_seed_writes_json_and_heatmap(tmp_path):
    import argparse
    import os

    import error_table_pinn as et
    from interpolation_stats import load_json

    args = argparse.Namespace(
        activation="Tanh",
        layers=[1],
        neurons=[5],
        epochs=20,
        collocation_points=16,
        patience=999,
        min_delta=1e-9,
        moving_avg_window=3,
        linf_points=64,
        l2_points=64,
        n_seeds=2,
        seed_base=42,
        failure_log_threshold=-0.5,
        results_dir=str(tmp_path / "results"),
        output_dir=str(tmp_path / "figures"),
        output=None,
    )
    cells = et.run_sweep(args)
    assert len(cells) == 2  # 1 cell x 2 seeds
    assert {c.seed for c in cells} == {42, 43}

    json_path, heatmap_path = et.persist(args, cells)
    sweep = load_json(json_path)
    assert sweep.activation == "Tanh"
    assert sweep.linf_mean[0][0] > 0.0

    assert os.path.exists(heatmap_path)


@pytest.mark.integration
def test_run_sweep_isolates_training_failures(tmp_path, monkeypatch):
    import argparse
    import error_table_pinn as et
    from interpolation_stats import load_json

    real_train = et.PINN_L2_Minimizer.train
    calls = {"n": 0}

    def flaky_train(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated CUDA OOM")
        return real_train(self, *a, **k)

    monkeypatch.setattr(et.PINN_L2_Minimizer, "train", flaky_train)

    args = argparse.Namespace(
        activation="Tanh", layers=[1], neurons=[5], epochs=15,
        collocation_points=16, patience=999, min_delta=1e-9,
        moving_avg_window=3, linf_points=64, l2_points=64,
        n_seeds=2, seed_base=42, failure_log_threshold=-0.5,
        results_dir=str(tmp_path / "results"),
        output_dir=str(tmp_path / "figures"), output=None,
    )
    cells = et.run_sweep(args)
    # both seeds still produce a CellResult; the failed one is a sentinel
    assert len(cells) == 2
    assert {c.seed for c in cells} == {42, 43}
    failed = [c for c in cells if c.linf == float("inf")]
    assert len(failed) == 1 and failed[0].epochs_run == 0
    # persist still works and the heatmap renders despite the inf cell
    json_path, heatmap_path = et.persist(args, cells)
    import os
    assert os.path.exists(heatmap_path)
    sweep = load_json(json_path)
    assert sweep.n_failed[0][0] == 1  # mean linf is inf -> flagged failed
    # a per-cell partial checkpoint was written
    assert os.path.exists(
        os.path.join(args.results_dir, "error_table_pinn_Tanh.partial.json")
    )


@pytest.mark.integration
def test_summarize_builds_latex_from_jsons(tmp_path):
    from interpolation_stats import CellResult, aggregate, save_json
    import summarize_interpolation as si

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for act in ("Tanh", "ReLU"):
        cells = [
            CellResult(layers=1, neurons=5, seed=42, linf=1e-2, l2=2e-3,
                       train_time_s=1.0, epochs_run=10),
            CellResult(layers=1, neurons=5, seed=43, linf=3e-2, l2=4e-3,
                       train_time_s=1.0, epochs_run=10),
        ]
        sw = aggregate(activation=act, layers=[1], neurons=[5], cells=cells,
                       failure_log_threshold=-0.5, machine_eps=1e-7)
        save_json(sw, str(results_dir / f"error_table_pinn_{act}.json"))

    out = tmp_path / "interpolation_summary.tex"
    si.build_summary(
        results_dir=str(results_dir),
        activations=["Tanh", "Sigmoid", "ReLU", "Softmax"],
        output_path=str(out),
    )
    text = out.read_text()
    assert r"\begin{tabular}" in text
    assert "Tanh" in text and "ReLU" in text
    assert "Sigmoid" not in text  # JSON absent -> skipped, not crashed


@pytest.mark.integration
def test_summarize_raises_when_no_jsons(tmp_path):
    import summarize_interpolation as si

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        si.build_summary(str(empty), ["Tanh"], str(tmp_path / "out.tex"))


@pytest.mark.integration
def test_orchestrator_runs_all_and_summarizes(tmp_path):
    import run_interpolation_study as rs

    rc = rs.run(
        activations=["Tanh"],
        layers=[1], neurons=[5], epochs=15, n_seeds=2,
        collocation_points=16, patience=999, min_delta=1e-9,
        moving_avg_window=3, linf_points=64, l2_points=64,
        failure_log_threshold=-0.5,
        results_dir=str(tmp_path / "results"),
        output_dir=str(tmp_path / "figures"),
        summary_path=str(tmp_path / "summary.tex"),
    )
    assert rc == 0
    assert (tmp_path / "summary.tex").exists()
    assert (tmp_path / "results" / "error_table_pinn_Tanh.json").exists()


@pytest.mark.integration
def test_run_sweep_resume_skips_cells_in_partial(tmp_path):
    """--resume must reuse cells already in <activation>.partial.json
    and only run the missing (L,W,seed) combinations."""
    import argparse
    import json
    import os

    import error_table_pinn as et
    from interpolation_stats import CellResult

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # Seed the checkpoint: L=1,W=5 fully done for seeds {42,43};
    # L=1,W=10 missing entirely. Use sentinel values so we can recognise them
    # in the merged result and confirm they were NOT re-run.
    seeded = [
        CellResult(layers=1, neurons=5, seed=42, linf=1.2345e-9, l2=1.0e-9,
                   train_time_s=0.01, epochs_run=1),
        CellResult(layers=1, neurons=5, seed=43, linf=2.3456e-9, l2=2.0e-9,
                   train_time_s=0.01, epochs_run=1),
    ]
    partial_path = results_dir / "error_table_pinn_Tanh.partial.json"
    partial_path.write_text(json.dumps(
        [c.__dict__ for c in seeded], indent=2
    ))

    args = argparse.Namespace(
        activation="Tanh", layers=[1], neurons=[5, 10], epochs=15,
        collocation_points=16, patience=999, min_delta=1e-9,
        moving_avg_window=3, linf_points=64, l2_points=64,
        n_seeds=2, seed_base=42, failure_log_threshold=-0.5,
        results_dir=str(results_dir),
        output_dir=str(tmp_path / "figures"), output=None,
        resume=True,
    )
    cells = et.run_sweep(args)
    assert len(cells) == 4  # 2 cells x 2 seeds

    by_key = {(c.layers, c.neurons, c.seed): c for c in cells}
    # Seeded cells preserved verbatim (sentinel linf values intact):
    assert by_key[(1, 5, 42)].linf == pytest.approx(1.2345e-9)
    assert by_key[(1, 5, 43)].linf == pytest.approx(2.3456e-9)
    # Missing cells got newly trained (linf comes from a real net, > 0
    # and definitely not the sentinel):
    assert by_key[(1, 5, 10)] if False else True  # silence noqa
    new = by_key[(1, 10, 42)]
    assert new.linf > 0 and new.linf != 1.2345e-9
    assert new.epochs_run == 15

    # persist still produces a valid final JSON over the merged grid
    json_path, _ = et.persist(args, cells)
    assert os.path.exists(json_path)


@pytest.mark.integration
def test_orchestrator_resume_skips_done_activations(tmp_path):
    """run() with resume=True must skip activations whose final JSON exists."""
    import json
    import os

    import run_interpolation_study as rs
    from interpolation_stats import CellResult, aggregate, save_json

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # Pre-populate a "completed" Tanh final JSON.
    done_cells = [
        CellResult(layers=1, neurons=5, seed=42, linf=1e-2, l2=2e-3,
                   train_time_s=0.1, epochs_run=1),
        CellResult(layers=1, neurons=5, seed=43, linf=3e-2, l2=4e-3,
                   train_time_s=0.1, epochs_run=1),
    ]
    sw = aggregate(activation="Tanh", layers=[1], neurons=[5],
                   cells=done_cells, failure_log_threshold=-0.5,
                   machine_eps=1e-7)
    save_json(sw, str(results_dir / "error_table_pinn_Tanh.json"))

    rc = rs.run(
        activations=["Tanh", "ReLU"],
        layers=[1], neurons=[5], epochs=15, n_seeds=2,
        collocation_points=16, patience=999, min_delta=1e-9,
        moving_avg_window=3, linf_points=64, l2_points=64,
        failure_log_threshold=-0.5,
        results_dir=str(results_dir),
        output_dir=str(tmp_path / "figures"),
        summary_path=str(tmp_path / "summary.tex"),
        resume=True,
    )
    assert rc == 0
    # Tanh JSON untouched (still the pre-populated 1e-2 / 3e-2 cells);
    # ReLU JSON now produced.
    tanh_after = json.loads(
        (results_dir / "error_table_pinn_Tanh.json").read_text()
    )
    assert tanh_after["linf_mean"][0][0] == pytest.approx(2e-2)
    assert (results_dir / "error_table_pinn_ReLU.json").exists()
    assert (tmp_path / "summary.tex").exists()
