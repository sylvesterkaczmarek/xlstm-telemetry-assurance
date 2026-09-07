import numpy as np
import pytest
import torch
from torch import nn

from xlstm_telemetry_assurance.benchmark import (
    FAULTS,
    VALUE_ONLY_FAULTS,
    _scenario_row,
    _adaptation_windows,
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


def test_adaptation_and_guard_buffers_share_no_raw_timestamps():
    timeline = np.arange(100, dtype=np.float32).reshape(-1, 1)
    adapt_x, adapt_y, guard_x, guard_y = _adaptation_windows(timeline, timeline, 8)
    adaptation_times = set(adapt_x.ravel()) | set(adapt_y.ravel())
    guard_times = set(guard_x.ravel()) | set(guard_y.ravel())
    assert not adaptation_times & guard_times
    assert max(adaptation_times) == 69
    assert min(guard_times) == 70
    assert guard_y[0, 0] == 78


@pytest.mark.parametrize("reading,flag", [(np.nan, False), (np.inf, False), (999.0, True)])
def test_unavailable_target_has_missingness_score_without_placeholder_residual(reading, flag):
    clean = np.zeros((12, 1), dtype=np.float32)
    observed = clean.copy()
    observed[7, 0] = reading
    missing = np.zeros_like(clean, dtype=bool)
    missing[7, 0] = flag
    mask = np.zeros(len(clean), dtype=bool)
    mask[7] = True
    arguments = dict(model=ZeroForecaster(1), clean=clean, observed=observed, missing=missing,
                     fault_mask=mask, mean=np.zeros(1), std=np.ones(1), seq_len=3)
    assert _score_stream(**arguments, threshold=2)["f1"] == 1
    assert _score_stream(**arguments, threshold=5)["recall"] == 0


def _complete_cohort(seed=11):
    rows = [_row("clean", 0, 1), _row("adaptation", 0, "", accepted=False)]
    rows += [_row(fault, 0.5, 1) for fault in FAULTS]
    return [{**row, "seed": seed} for row in rows]


def test_summary_uses_sample_sd_and_distinct_seed_count():
    rows = _complete_cohort(11) + _complete_cohort(29)
    rows[8]["rmse"] = 3.0
    result = _summarize_model_rows(rows)
    assert result["clean_rmse_mean"] == 2
    assert result["clean_rmse_std"] == pytest.approx(np.sqrt(2))
    assert result["seed_count"] == 2
    assert result["adaptation_accept_rate"] == 0


def test_summary_rejects_duplicate_incomplete_and_invalid_cohorts():
    rows = _complete_cohort()
    for malformed in ([], rows + [rows[0]], rows[:-1], rows + _complete_cohort(29)[:-1]):
        with pytest.raises(ValueError):
            _summarize_model_rows(malformed)
    for key, value in (("rmse", float("nan")), ("domain", "robotics")):
        with pytest.raises(ValueError):
            _summarize_model_rows([{**rows[0], key: value}, *rows[1:]])
    with pytest.raises(ValueError, match="must be a boolean"):
        _summarize_model_rows([rows[0], {**rows[1], "adaptation_accepted": "False"}, *rows[2:]])


def test_failed_benchmark_preserves_previous_evidence(tmp_path, monkeypatch):
    from xlstm_telemetry_assurance import benchmark

    output = tmp_path / "reference"
    output.mkdir()
    names = ("metrics.csv", "summary.json", "fault_detection_f1.png", "run_environment.json")
    for name in names:
        (output / name).write_bytes(b"previous checked evidence")

    def fail(staging, smoke=False):
        (staging / "run_environment.json").write_text("incomplete candidate run")
        raise FloatingPointError("failed training")

    monkeypatch.setattr(benchmark, "_run_benchmark", fail)
    with pytest.raises(FloatingPointError, match="failed training"):
        benchmark.run_benchmark(output)
    assert all((output / name).read_bytes() == b"previous checked evidence" for name in names)
