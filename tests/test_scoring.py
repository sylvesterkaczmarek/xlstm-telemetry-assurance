import numpy as np
import torch
from torch import nn

from xlstm_telemetry_assurance.benchmark import (
    FAULTS,
    VALUE_ONLY_FAULTS,
    _scenario_row,
    _score_stream,
    _summarize_model_rows,
    _timing_metadata,
)


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
        "window_inference_latency_ms": 0.5,
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
    assert "latency_ms" not in fault_row
    assert fault_row["window_inference_latency_ms"] == 0.5


def test_timing_metadata_states_host_window_scope_and_limitations():
    metadata = _timing_metadata(24)
    assert metadata["metric"] == "window_inference_latency_ms"
    assert metadata["device"] == "host_cpu"
    assert metadata["scope"] == "complete_model_forward_pass_for_one_input_window"
    assert metadata["input_window_length"] == 24
    assert metadata["hardware_dependent"] is True
    assert metadata["spacecraft_or_robot_realtime_timing_measured"] is False
    assert metadata["wcet_measured"] is False


def _row(scenario, f1, nll, *, latency=0.4, accepted=""):
    return {
        "domain": "spacecraft",
        "model": "lstm",
        "seed": 11,
        "scenario": scenario,
        "rmse": 1.0 if scenario == "clean" else 2.0,
        "coverage_90": 0.9 if scenario == "clean" else 0.8,
        "gaussian_nll": nll,
        "f1": "" if scenario in {"clean", "adaptation"} else f1,
        "precision": "",
        "recall": "",
        "false_alarm_rate": 0.01 if scenario == "clean" else "",
        "parameters": 10,
        "window_inference_latency_ms": latency,
        "adaptation_accepted": accepted,
    }


def test_summary_separates_packet_loss_value_faults_and_mixed_and_propagates_nll():
    rows = [_row("clean", 0.0, 1.0), _row("adaptation", 0.0, "", accepted=True)]
    f1_values = {
        "packet_loss": 0.90,
        "spike": 0.10,
        "stuck": 0.20,
        "drift": 0.30,
        "regime_shift": 0.40,
        "mixed": 0.50,
    }
    for index, fault in enumerate(FAULTS, start=1):
        rows.append(_row(fault, f1_values[fault], 1.0 + index))

    summary = _summarize_model_rows(rows)
    expected_value_mean = np.mean([f1_values[fault] for fault in VALUE_ONLY_FAULTS])

    assert summary["packet_loss_f1_mean"] == f1_values["packet_loss"]
    assert summary["value_fault_f1_mean"] == expected_value_mean
    assert summary["mixed_fault_f1_mean"] == f1_values["mixed"]
    assert summary["fault_f1_mean"] == np.mean(list(f1_values.values()))
    assert summary["per_fault_f1_mean"] == f1_values
    assert summary["clean_gaussian_nll_mean"] == 1.0
    assert summary["fault_gaussian_nll_mean"] == np.mean([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert summary["window_inference_latency_ms_mean"] == 0.4
