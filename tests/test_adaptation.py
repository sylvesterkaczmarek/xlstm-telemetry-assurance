import numpy as np
import torch

from xlstm_telemetry_assurance.models import LSTMForecaster
from xlstm_telemetry_assurance.training import guarded_adaptation, train_model


def test_guarded_adaptation_rolls_back_on_guard_degradation():
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(64, 8, 6)).astype(np.float32)
    y = x[:, -1, :3].copy()
    model = LSTMForecaster(channels=3, hidden_size=8)
    train_model(model, x, y, epochs=2)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    bad_y = y + 8.0
    result = guarded_adaptation(model, x[:32], bad_y[:32], x[32:], y[32:], steps=8, lr=0.05, tolerance=0.0)
    assert not result.accepted
    for key, value in model.state_dict().items():
        assert torch.allclose(value, before[key])
