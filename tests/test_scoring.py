import numpy as np
import torch
from torch import nn

from xlstm_telemetry_assurance.benchmark import _scenario_row, _score_stream


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


def test_scenario_row_uses_metrics_from_exact_fault_scenario():
    common = {
        "domain": "spacecraft",
        "model": "lstm",
        "seed": 11,
        "parameters": 123,
        "latency_ms": 0.5,
    }
    clean_metrics = {
        "rmse": 1.0,
        "coverage_90": 0.91,
        "gaussian_nll": 2.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "false_positive_rate": 0.01,
    }
    fault_metrics = {
        "rmse": 4.0,
        "coverage_90": 0.61,
        "gaussian_nll": 5.0,
        "f1": 0.7,
        "precision": 0.8,
        "recall": 0.6,
        "false_positive_rate": 0.12,
    }

    clean_row = _scenario_row(common, "clean", clean_metrics, include_detection_metrics=False)
    fault_row = _scenario_row(common, "drift", fault_metrics, include_detection_metrics=True)

    assert fault_row["rmse"] == fault_metrics["rmse"] != clean_row["rmse"]
    assert fault_row["coverage_90"] == fault_metrics["coverage_90"] != clean_row["coverage_90"]
    assert fault_row["gaussian_nll"] == fault_metrics["gaussian_nll"] != clean_row["gaussian_nll"]
    assert fault_row["false_alarm_rate"] == fault_metrics["false_positive_rate"] != clean_row["false_alarm_rate"]
    assert fault_row["f1"] == fault_metrics["f1"]
    assert fault_row["precision"] == fault_metrics["precision"]
    assert fault_row["recall"] == fault_metrics["recall"]
    assert clean_row["f1"] == ""
    assert clean_row["precision"] == ""
    assert clean_row["recall"] == ""
