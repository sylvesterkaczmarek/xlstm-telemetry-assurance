from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _positive_size(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _forecast_input(x: torch.Tensor, channels: int) -> None:
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] != channels * 2:
        raise ValueError("input must have nonempty shape (batch, time, 2 * channels)")


class GaussianHead(nn.Module):
    def __init__(self, hidden_size: int, channels: int, residual: bool = True) -> None:
        super().__init__()
        _positive_size("channels", channels)
        _positive_size("hidden_size", hidden_size)
        self.channels = channels
        self.residual = residual
        self.proj = nn.Linear(hidden_size, channels * 2)

    def forward(self, hidden: torch.Tensor, last_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.proj(hidden)
        delta, raw_scale = raw.chunk(2, dim=-1)
        mean = last_value + delta if self.residual else delta
        std = F.softplus(raw_scale) + 0.05
        return mean, std


class LSTMForecaster(nn.Module):
    def __init__(self, channels: int, hidden_size: int = 32) -> None:
        super().__init__()
        _positive_size("channels", channels)
        _positive_size("hidden_size", hidden_size)
        self.channels = channels
        self.lstm = nn.LSTM(input_size=channels * 2, hidden_size=hidden_size, batch_first=True)
        self.head = GaussianHead(hidden_size, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _forecast_input(x, self.channels)
        output, _ = self.lstm(x)
        hidden = output[:, -1]
        last_value = x[:, -1, : self.channels]
        return self.head(hidden, last_value)


class StabilizedSLSTMCell(nn.Module):
    """Compact sLSTM-inspired cell with stabilized exponential gates.

    This keeps the core xLSTM idea of exponential input/forget gates with a
    normalizer and log-space stabilizer while staying small enough for an
    inspectable benchmark. It is not a reimplementation of the full xLSTM
    reference library.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        _positive_size("input_size", input_size)
        _positive_size("hidden_size", hidden_size)
        self.hidden_size = hidden_size
        self.x_proj = nn.Linear(input_size, hidden_size * 4)
        self.h_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        h, c, n, m = state
        i_log, f_log, z_raw, o_raw = (self.x_proj(x) + self.h_proj(h)).chunk(4, dim=-1)
        z = torch.tanh(z_raw)
        o = torch.sigmoid(o_raw)

        m_new = torch.maximum(i_log, f_log + m)
        i = torch.exp(i_log - m_new)
        f = torch.exp(f_log + m - m_new)
        c_new = f * c + i * z
        n_new = f * n + i
        # From the empty state, at least one stabilised gate is exactly one,
        # so the normaliser is >= 1. Artificial gate/denominator floors would
        # change the recurrence and suppress legitimate small input gates.
        normalized = c_new / n_new
        h_new = o * normalized
        return h_new, (h_new, c_new, n_new, m_new)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        _positive_size("batch_size", batch_size)
        zeros = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        # No previous memory contributes to the first step. This makes its
        # stabiliser equal to i_log for every finite input-gate value.
        neg_inf = torch.full_like(zeros, -torch.inf)
        return zeros, zeros, zeros, neg_inf


class XLSTMForecaster(nn.Module):
    def __init__(self, channels: int, hidden_size: int = 32) -> None:
        super().__init__()
        _positive_size("channels", channels)
        _positive_size("hidden_size", hidden_size)
        self.channels = channels
        self.input_norm = nn.LayerNorm(channels * 2)
        self.cell = StabilizedSLSTMCell(channels * 2, hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.head = GaussianHead(hidden_size, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _forecast_input(x, self.channels)
        x_norm = self.input_norm(x)
        state = self.cell.initial_state(x.shape[0], x.device, x.dtype)
        hidden = state[0]
        for step in range(x.shape[1]):
            hidden, state = self.cell(x_norm[:, step], state)
        hidden = self.post_norm(hidden)
        last_value = x[:, -1, : self.channels]
        return self.head(hidden, last_value)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
