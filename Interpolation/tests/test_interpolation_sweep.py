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
