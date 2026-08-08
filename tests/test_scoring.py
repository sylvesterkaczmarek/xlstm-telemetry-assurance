import numpy as np
import torch
from torch import nn

from xlstm_telemetry_assurance.benchmark import _score_stream


class ZeroForecaster(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        mean = torch.zeros((batch, self.channels), dtype=x.dtype, device=x.device)
        std = torch.ones_like(mean)
        return mean, std


def test_fault_scoring_uses_observed_measurement_not_clean_counterfactual():
    clean = np.zeros((12, 1), dtype=np.float32)
    observed = clean.copy()
    observed[7, 0] = 8.0
    missing = np.zeros_like(clean, dtype=bool)
    fault_mask = np.zeros(len(clean), dtype=bool)
    fault_mask[7] = True

    metrics = _score_stream(
        ZeroForecaster(channels=1),
        clean=clean,
        observed=observed,
        missing=missing,
        fault_mask=fault_mask,
        mean=np.zeros(1, dtype=np.float32),
        std=np.ones(1, dtype=np.float32),
        threshold=2.0,
        seq_len=3,
    )

    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 1.0
