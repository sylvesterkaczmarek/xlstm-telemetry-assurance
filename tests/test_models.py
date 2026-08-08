import torch

from xlstm_telemetry_assurance.models import LSTMForecaster, XLSTMForecaster


def test_lstm_shapes_and_positive_scale():
    model = LSTMForecaster(channels=3, hidden_size=8)
    mean, std = model(torch.randn(4, 12, 6))
    assert mean.shape == (4, 3)
    assert std.shape == (4, 3)
    assert torch.all(std > 0)


def test_xlstm_shapes_and_positive_scale():
    model = XLSTMForecaster(channels=3, hidden_size=8)
    mean, std = model(torch.randn(4, 12, 6))
    assert mean.shape == (4, 3)
    assert std.shape == (4, 3)
    assert torch.all(std > 0)


def test_xlstm_forward_is_finite():
    model = XLSTMForecaster(channels=3, hidden_size=8)
    mean, std = model(torch.randn(4, 24, 6) * 10)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
