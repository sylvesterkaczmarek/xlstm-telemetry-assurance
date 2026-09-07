import numpy as np
import pytest
import torch
from torch import nn

from xlstm_telemetry_assurance.models import LSTMForecaster
from xlstm_telemetry_assurance.training import guarded_adaptation, train_model


class SmallForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.BatchNorm1d(2)
        self.head = nn.Linear(2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.training_calls = 0
        self.fail_on = None

    def forward(self, x):
        features = self.body(x[:, -1])
        if self.head.training:
            self.training_calls += 1
            if self.training_calls == self.fail_on:
                raise RuntimeError("candidate failed after an update")
        mean = self.head(features)
        return mean, torch.full_like(mean, 0.1)


def _small_data():
    return np.ones((4, 2, 2), dtype=np.float32), np.zeros((4, 1), dtype=np.float32)


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


def test_negative_guard_loss_accepts_unchanged_model():
    model = SmallForecaster()
    x, y = _small_data()
    result = guarded_adaptation(model, x, y, x, y, steps=0)
    assert result.guard_loss_before < 0
    assert result.guard_loss_after == result.guard_loss_before
    assert result.accepted


@pytest.mark.parametrize("degradation, accepted", [(0.005, True), (0.03, False)])
def test_negative_nll_tolerance_allows_only_small_degradation(monkeypatch, degradation, accepted):
    from xlstm_telemetry_assurance import training

    losses = iter([-1.0, -1.0 + degradation])
    monkeypatch.setattr(training, "_loss", lambda *args: next(losses))
    model = SmallForecaster()
    x, y = _small_data()
    result = guarded_adaptation(model, x, y, x, y, steps=0, tolerance=0.01)
    assert result.accepted is accepted


@pytest.mark.parametrize("outcome", ["accepted", "rejected", "exception"])
def test_adaptation_preserves_modes_gradient_flags_and_gradients(outcome):
    model = SmallForecaster()
    model.train()
    model.body.eval()
    model.head.weight.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    gradients = [parameter.grad for parameter in model.parameters()]
    states = {name: value.clone() for name, value in model.state_dict().items()}
    x, y = _small_data()
    if outcome == "exception":
        model.fail_on = 2
        with pytest.raises(RuntimeError, match="candidate failed"):
            guarded_adaptation(model, x, y + 2, x, y, steps=3)
    else:
        target = y if outcome == "accepted" else y + 2
        result = guarded_adaptation(model, x, target, x, y, steps=3, tolerance=0)
        assert result.accepted is (outcome == "accepted")
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    for parameter, gradient in zip(model.parameters(), gradients):
        assert parameter.grad is gradient
        assert torch.equal(parameter.grad, torch.full_like(parameter, 0.25))
    if outcome != "accepted":
        for name, value in model.state_dict().items():
            assert torch.equal(value, states[name])


def test_head_only_adaptation_does_not_update_body_running_statistics():
    model = SmallForecaster()
    model.train()
    before = {name: value.clone() for name, value in model.body.state_dict().items()}
    x, y = _small_data()
    guarded_adaptation(model, x, y + 0.05, x, y, steps=2, tolerance=100)
    for name, value in model.body.state_dict().items():
        assert torch.equal(value, before[name])


def test_nonfinite_candidate_restores_state(monkeypatch):
    model = SmallForecaster()
    x, y = _small_data()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    original_step = torch.optim.Adam.step

    def overflowing_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("inf"))
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", overflowing_step)
    with pytest.raises(FloatingPointError, match="nonfinite parameters"):
        guarded_adaptation(model, x, y, x, y, steps=1)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("settings", [{"steps": -1}, {"steps": True}, {"lr": 0}, {"tolerance": -0.1}, {"tolerance": float("nan")}])
def test_invalid_adaptation_settings_leave_model_untouched(settings):
    model = SmallForecaster()
    x, y = _small_data()
    with pytest.raises(ValueError):
        guarded_adaptation(model, x, y, x, y, **settings)
    assert model.training
    assert all(parameter.requires_grad for parameter in model.parameters())
