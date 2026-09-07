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


def test_first_step_is_independent_of_input_gate_magnitude():
    from xlstm_telemetry_assurance.models import StabilizedSLSTMCell

    cell = StabilizedSLSTMCell(1, 1).double()
    with torch.no_grad():
        cell.x_proj.weight.zero_()
        cell.h_proj.weight.zero_()
    x = torch.zeros(1, 1, dtype=torch.float64)
    expected = 0.5 * torch.tanh(torch.tensor(1.0, dtype=torch.float64))
    for input_log in (-1000.0, -100.0, 0.0, 1000.0):
        with torch.no_grad():
            cell.x_proj.bias.copy_(torch.tensor([input_log, 0.0, 1.0, 0.0]))
        hidden, state = cell(x, cell.initial_state(1, x.device, x.dtype))
        torch.testing.assert_close(hidden.squeeze(), expected)
        assert state[2].item() == 1.0
        assert all(torch.isfinite(value).all() for value in state)


def test_stabilised_cell_matches_unscaled_recurrence_and_gradients():
    from copy import deepcopy

    from xlstm_telemetry_assurance.models import StabilizedSLSTMCell

    torch.manual_seed(91)
    cell = StabilizedSLSTMCell(3, 4).double()
    reference = deepcopy(cell)
    inputs = (torch.randn(2, 17, 3, dtype=torch.float64) * 0.25).requires_grad_()
    reference_inputs = inputs.detach().clone().requires_grad_()
    state = cell.initial_state(2, inputs.device, inputs.dtype)
    actual = []
    expected = []
    h = torch.zeros(2, 4, dtype=torch.float64)
    c, n = torch.zeros_like(h), torch.zeros_like(h)
    for step in range(inputs.shape[1]):
        hidden, state = cell(inputs[:, step], state)
        actual.append(hidden)
        i_log, f_log, z_raw, o_raw = (
            reference.x_proj(reference_inputs[:, step]) + reference.h_proj(h)
        ).chunk(4, dim=-1)
        c = torch.exp(f_log) * c + torch.exp(i_log) * torch.tanh(z_raw)
        n = torch.exp(f_log) * n + torch.exp(i_log)
        h = torch.sigmoid(o_raw) * c / n
        expected.append(h)
    actual, expected = torch.stack(actual), torch.stack(expected)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=1e-9, atol=1e-11)
    for actual_parameter, expected_parameter in zip(cell.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=1e-9, atol=1e-11)


def test_stabilised_cell_long_sequence_has_finite_states_and_gradients():
    from xlstm_telemetry_assurance.models import StabilizedSLSTMCell

    torch.manual_seed(92)
    cell = StabilizedSLSTMCell(3, 4)
    with torch.no_grad():
        cell.x_proj.bias[:4].copy_(torch.tensor([-100.0, 100.0, -40.0, 40.0]))
        cell.x_proj.bias[4:8].copy_(torch.tensor([2.0, -2.0, 1.0, -1.0]))
    inputs = (torch.randn(2, 512, 3) * 10).requires_grad_()
    state = cell.initial_state(2, inputs.device, inputs.dtype)
    outputs = []
    for step in range(inputs.shape[1]):
        hidden, state = cell(inputs[:, step], state)
        outputs.append(hidden)
        assert all(torch.isfinite(value).all() for value in state)
        assert torch.all(state[2] >= 1)
        assert torch.all(torch.abs(hidden) <= 1)
    torch.stack(outputs).square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in cell.parameters())


def test_forecasters_reject_invalid_dimensions():
    import pytest

    for model_class in (LSTMForecaster, XLSTMForecaster):
        for invalid in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="channels"):
                model_class(channels=invalid)
            with pytest.raises(ValueError, match="hidden_size"):
                model_class(channels=3, hidden_size=invalid)
        model = model_class(channels=3, hidden_size=8)
        for shape in ((0, 12, 6), (4, 0, 6), (4, 12, 3), (12, 6)):
            with pytest.raises(ValueError, match="shape"):
                model(torch.empty(shape))
