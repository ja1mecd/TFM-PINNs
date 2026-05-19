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
