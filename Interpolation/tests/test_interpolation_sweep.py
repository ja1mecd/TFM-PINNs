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

    import os
    assert os.path.exists(heatmap_path)
