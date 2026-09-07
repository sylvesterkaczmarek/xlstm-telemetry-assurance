import math

import numpy as np
import pytest
import torch

from xlstm_telemetry_assurance.models import LSTMForecaster
from xlstm_telemetry_assurance.training import _nll, predict, train_model


def _data():
    return np.zeros((4, 3, 2), dtype=np.float32), np.zeros((4, 1), dtype=np.float32)


def test_nll_rejects_target_broadcasting():
    with pytest.raises(ValueError, match="same nonempty"):
        _nll(torch.zeros(3, 1), torch.ones(3, 1), torch.zeros(3, 4))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_nll_rejects_invalid_scales(scale):
    with pytest.raises((ValueError, FloatingPointError)):
        _nll(torch.zeros(2, 1), torch.full((2, 1), scale), torch.zeros(2, 1))


def test_nll_large_scale_does_not_overflow_variance():
    actual = _nll(torch.zeros(1, 1), torch.full((1, 1), 1e30), torch.zeros(1, 1))
    assert float(actual) == pytest.approx(math.log(1e30) + 0.5 * math.log(2 * math.pi), rel=1e-6)


def test_nonfinite_targets_fail_before_corrupting_model():
    model = LSTMForecaster(1, 2)
    x, y = _data()
    y[0, 0] = np.nan
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="finite"):
        train_model(model, x, y, epochs=1)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("bad_data", ["empty", "bad_x_dimensions", "bad_y_dimensions", "mismatched_samples", "integer"])
def test_training_rejects_malformed_data(bad_data):
    model = LSTMForecaster(1, 2)
    x, y = _data()
    if bad_data == "empty":
        x, y = x[:0], y[:0]
    elif bad_data == "bad_x_dimensions":
        x = x[:, 0]
    elif bad_data == "bad_y_dimensions":
        y = y[:, 0]
    elif bad_data == "mismatched_samples":
        y = y[:2]
    else:
        x = x.astype(np.int64)
    with pytest.raises(ValueError):
        train_model(model, x, y, epochs=1)


@pytest.mark.parametrize("settings", [{"epochs": -1}, {"epochs": True}, {"epochs": 1.5}, {"lr": 0}, {"lr": float("inf")}])
def test_training_rejects_invalid_settings(settings):
    x, y = _data()
    with pytest.raises(ValueError):
        train_model(LSTMForecaster(1, 2), x, y, **settings)


def test_double_precision_model_accepts_float32_arrays_and_prediction_preserves_modes():
    model = LSTMForecaster(1, 2).double()
    model.train()
    model.head.eval()
    x, y = _data()
    loss = train_model(model, x, y, epochs=1)
    model.head.eval()
    modes = [module.training for module in model.modules()]
    mean, std = predict(model, x)
    assert len(loss) == 1 and math.isfinite(loss[0])
    assert mean.dtype == np.float64 and std.dtype == np.float64
    assert [module.training for module in model.modules()] == modes


def test_prediction_rejects_nonfinite_output_and_restores_mode():
    model = LSTMForecaster(1, 2)
    model.train()
    with torch.no_grad():
        model.head.proj.bias.fill_(float("nan"))
    x, _ = _data()
    with pytest.raises(FloatingPointError, match="finite"):
        predict(model, x)
    assert model.training


@pytest.mark.parametrize("has_buffer", [False, True])
def test_prediction_supports_parameterless_baselines(has_buffer):
    class PersistenceForecaster(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if has_buffer:
                self.register_buffer("scale", torch.ones(1, dtype=torch.float64))

        def forward(self, x):
            mean = x[:, -1, :1]
            return mean, torch.ones_like(mean)

    x, _ = _data()
    mean, std = predict(PersistenceForecaster(), x)
    np.testing.assert_array_equal(mean, x[:, -1, :1])
    np.testing.assert_array_equal(std, np.ones((4, 1)))
    assert mean.dtype == (np.float64 if has_buffer else np.float32)
