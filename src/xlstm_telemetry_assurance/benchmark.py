from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import build_windows, generate_clean_telemetry, inject_fault, prepare_observed_inputs, standardize
from .metrics import binary_metrics, coverage, gaussian_nll, rmse
from .models import LSTMForecaster, XLSTMForecaster, count_parameters
from .provenance import configure_deterministic_execution, write_run_environment
from .training import guarded_adaptation, predict, train_model

DEFAULT_SEEDS = [11, 29, 47]
DOMAINS = ["spacecraft", "robotics"]
FAULTS = ["packet_loss", "spike", "stuck", "drift", "regime_shift", "mixed"]
VALUE_ONLY_FAULTS = ["spike", "stuck", "drift", "regime_shift"]


def _timing_metadata(sequence_length: int) -> dict:
    return {
        "metric": "window_inference_latency_ms",
        "device": "host_cpu",
        "scope": "complete_model_forward_pass_for_one_input_window",
        "input_window_length": sequence_length,
        "hardware_dependent": True,
        "spacecraft_or_robot_realtime_timing_measured": False,
        "wcet_measured": False,
    }


def _benchmark_config(smoke: bool) -> dict:
    sequence_length = 16 if smoke else 24
    return {
        "smoke": smoke,
        "seeds": [11] if smoke else list(DEFAULT_SEEDS),
        "domains": list(DOMAINS),
        "faults": list(FAULTS),
        "value_only_faults": list(VALUE_ONLY_FAULTS),
        "sequence_length": sequence_length,
        "train_length": 520 if smoke else 1050,
        "eval_length": 300 if smoke else 700,
        "epochs": 2 if smoke else 10,
        "hidden_size": 20 if smoke else 32,
        "training_learning_rate": 0.004 if smoke else 0.003,
        "calibration_quantile": 0.99,
        "missingness_score_boost": 4.0,
        "adaptation_steps": 4 if smoke else 10,
        "adaptation_learning_rate": 0.015,
        "adaptation_tolerance": 0.01,
        "window_inference_latency_repeats": 120,
        "timing": _timing_metadata(sequence_length),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _window_inference_latency_ms(model: torch.nn.Module, x: torch.Tensor, repeats: int = 120) -> float:
    """Measure one complete host-CPU forward pass for one input window."""
    model.eval()
    sample = x[:1]
    with torch.no_grad():
        for _ in range(10):
            model(sample)
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / repeats


def _score_stream(
    model: torch.nn.Module,
    clean: np.ndarray,
    observed: np.ndarray,
    missing: np.ndarray,
    fault_mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    seq_len: int,
) -> dict[str, float]:
    inputs, observed_std = prepare_observed_inputs(observed, missing, mean, std)
    clean_std = standardize(clean, mean, std)
    x, y_observed, indices = build_windows(inputs, observed_std, seq_len)
    _, y_clean, _ = build_windows(inputs, clean_std, seq_len)
    pred_mean, pred_std = predict(model, x)

    # Runtime anomaly scoring uses only information actually available to the
    # system. The clean counterfactual is retained only for benchmark metrics.
    standardized_residual = np.abs(y_observed - pred_mean) / np.maximum(pred_std, 1e-4)
    target_missing = missing[indices]
    available = ~target_missing
    residual_sum = np.sum(standardized_residual * available, axis=1)
    available_count = np.maximum(np.sum(available, axis=1), 1)
    scores = residual_sum / available_count

    # Missing measurements are explicit telemetry faults. Their forward-filled
    # values are excluded from the residual and missingness is scored directly.
    miss_step = target_missing.any(axis=1).astype(np.float32)
    scores = scores + 4.0 * miss_step

    labels = fault_mask[indices].astype(np.int64)
    preds = (scores > threshold).astype(np.int64)
    metrics = binary_metrics(labels, preds)
    metrics["rmse"] = rmse(y_clean, pred_mean)
    metrics["coverage_90"] = coverage(y_clean, pred_mean, pred_std, z=1.6448536269514722)
    metrics["gaussian_nll"] = gaussian_nll(y_clean, pred_mean, pred_std)
    return metrics


def _scenario_row(
    common: dict,
    scenario: str,
    metrics: dict[str, float],
    *,
    include_detection_metrics: bool,
) -> dict:
    """Build one benchmark row from the metrics for that exact scenario."""
    return {
        **common,
        "scenario": scenario,
        "rmse": metrics["rmse"],
        "coverage_90": metrics["coverage_90"],
        "gaussian_nll": metrics["gaussian_nll"],
        "f1": metrics["f1"] if include_detection_metrics else "",
        "precision": metrics["precision"] if include_detection_metrics else "",
        "recall": metrics["recall"] if include_detection_metrics else "",
        "false_alarm_rate": metrics["false_positive_rate"],
        "adaptation_accepted": "",
    }


def _calibrate_threshold(model, clean, mean, std, seq_len):
    missing = np.zeros_like(clean, dtype=bool)
    inputs, clean_std = prepare_observed_inputs(clean, missing, mean, std)
    x, y, _ = build_windows(inputs, clean_std, seq_len)
    pred_mean, pred_std = predict(model, x)
    scores = np.mean(np.abs(y - pred_mean) / np.maximum(pred_std, 1e-4), axis=1)
    return float(np.quantile(scores, 0.99))


def _adaptation_check(model, domain, seed, mean, std, seq_len, smoke=False):
    length = 360 if smoke else 720
    clean = generate_clean_telemetry(domain, length=length, seed=seed + 700)
    observed, missing, _ = inject_fault(clean, "drift", seed=seed + 900)
    inputs, clean_std = prepare_observed_inputs(observed, missing, mean, std)
    clean_target = standardize(clean, mean, std)
    x, _, _ = build_windows(inputs, clean_std, seq_len)
    _, y, _ = build_windows(inputs, clean_target, seq_len)
    split = int(len(x) * 0.7)
    adapt_x, adapt_y = x[:split], y[:split]
    guard_x, guard_y = x[split:], y[split:]
    result = guarded_adaptation(
        model,
        adapt_x,
        adapt_y,
        guard_x,
        guard_y,
        steps=4 if smoke else 10,
        lr=0.015,
        tolerance=0.01,
    )
    return result


def run_benchmark(output: Path, smoke: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    config = _benchmark_config(smoke)
    deterministic_settings = configure_deterministic_execution()
    write_run_environment(output / "run_environment.json", config, deterministic_settings)

    seeds = config["seeds"]
    seq_len = config["sequence_length"]
    train_length = config["train_length"]
    eval_length = config["eval_length"]
    epochs = config["epochs"]
    hidden = config["hidden_size"]

    rows: list[dict] = []

    for domain in DOMAINS:
        for seed in seeds:
            _seed_everything(seed)
            train_clean = generate_clean_telemetry(domain, train_length, seed)
            mean = train_clean.mean(axis=0)
            std = train_clean.std(axis=0) + 1e-6
            missing = np.zeros_like(train_clean, dtype=bool)
            train_inputs, train_std = prepare_observed_inputs(train_clean, missing, mean, std)
            train_x, train_y, _ = build_windows(train_inputs, train_std, seq_len)

            calibration = generate_clean_telemetry(domain, eval_length, seed + 100)
            test_clean = generate_clean_telemetry(domain, eval_length, seed + 200)

            for model_name, factory in [
                ("lstm", lambda: LSTMForecaster(channels=train_clean.shape[1], hidden_size=hidden)),
                ("xlstm", lambda: XLSTMForecaster(channels=train_clean.shape[1], hidden_size=hidden)),
            ]:
                _seed_everything(seed)
                model = factory()
                train_model(model, train_x, train_y, epochs=epochs, lr=config["training_learning_rate"])
                threshold = _calibrate_threshold(model, calibration, mean, std, seq_len)

                clean_obs = test_clean.copy()
                clean_missing = np.zeros_like(test_clean, dtype=bool)
                clean_mask = np.zeros(len(test_clean), dtype=bool)
                clean_metrics = _score_stream(
                    model,
                    test_clean,
                    clean_obs,
                    clean_missing,
                    clean_mask,
                    mean,
                    std,
                    threshold,
                    seq_len,
                )
                window_latency = _window_inference_latency_ms(
                    model,
                    torch.from_numpy(train_x[:16]),
                    repeats=config["window_inference_latency_repeats"],
                )
                common = {
                    "domain": domain,
                    "model": model_name,
                    "seed": seed,
                    "parameters": count_parameters(model),
                    "window_inference_latency_ms": window_latency,
                }
                rows.append(
                    _scenario_row(
                        common,
                        "clean",
                        clean_metrics,
                        include_detection_metrics=False,
                    )
                )

                for fault in FAULTS:
                    observed, fault_missing, fault_mask = inject_fault(test_clean, fault, seed=seed + 300)
                    metrics = _score_stream(
                        model,
                        test_clean,
                        observed,
                        fault_missing,
                        fault_mask,
                        mean,
                        std,
                        threshold,
                        seq_len,
                    )
                    rows.append(
                        _scenario_row(
                            common,
                            fault,
                            metrics,
                            include_detection_metrics=True,
                        )
                    )

                adaptation = _adaptation_check(model, domain, seed, mean, std, seq_len, smoke=smoke)
                rows.append(
                    {
                        **common,
                        "scenario": "adaptation",
                        "rmse": "",
                        "coverage_90": "",
                        "gaussian_nll": "",
                        "f1": "",
                        "precision": "",
                        "recall": "",
                        "false_alarm_rate": "",
                        "adaptation_accepted": adaptation.accepted,
                    }
                )

    fieldnames = [
        "domain",
        "model",
        "seed",
        "scenario",
        "rmse",
        "coverage_90",
        "gaussian_nll",
        "f1",
        "precision",
        "recall",
        "false_alarm_rate",
        "parameters",
        "window_inference_latency_ms",
        "adaptation_accepted",
    ]
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows, timing_metadata=config["timing"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot_fault_f1(rows, output / "fault_detection_f1.png")
    return summary


def _summarize_model_rows(selected: list[dict]) -> dict:
    clean = [r for r in selected if r["scenario"] == "clean"]
    faults = [r for r in selected if r["scenario"] in FAULTS]
    packet_loss = [r for r in selected if r["scenario"] == "packet_loss"]
    value_faults = [r for r in selected if r["scenario"] in VALUE_ONLY_FAULTS]
    mixed = [r for r in selected if r["scenario"] == "mixed"]
    adaptations = [r for r in selected if r["scenario"] == "adaptation"]
    per_fault_f1 = {
        fault: float(np.mean([float(r["f1"]) for r in selected if r["scenario"] == fault]))
        for fault in FAULTS
    }
    return {
        "clean_rmse_mean": float(np.mean([float(r["rmse"]) for r in clean])),
        "clean_rmse_std": float(np.std([float(r["rmse"]) for r in clean])),
        "clean_coverage_90_mean": float(np.mean([float(r["coverage_90"]) for r in clean])),
        "clean_gaussian_nll_mean": float(np.mean([float(r["gaussian_nll"]) for r in clean])),
        "clean_gaussian_nll_std": float(np.std([float(r["gaussian_nll"]) for r in clean])),
        "fault_gaussian_nll_mean": float(np.mean([float(r["gaussian_nll"]) for r in faults])),
        "fault_f1_mean": float(np.mean([float(r["f1"]) for r in faults])),
        "packet_loss_f1_mean": float(np.mean([float(r["f1"]) for r in packet_loss])),
        "value_fault_f1_mean": float(np.mean([float(r["f1"]) for r in value_faults])),
        "mixed_fault_f1_mean": float(np.mean([float(r["f1"]) for r in mixed])),
        "per_fault_f1_mean": per_fault_f1,
        "false_alarm_rate_mean": float(np.mean([float(r["false_alarm_rate"]) for r in clean])),
        "parameters": int(clean[0]["parameters"]),
        "window_inference_latency_ms_mean": float(
            np.mean([float(r["window_inference_latency_ms"]) for r in clean])
        ),
        "adaptation_accept_rate": float(
            np.mean([1.0 if r["adaptation_accepted"] else 0.0 for r in adaptations])
        ),
    }


def _summarize(rows: list[dict], timing_metadata: dict | None = None) -> dict:
    summary: dict = {
        "_metadata": {
            "schema_version": 2,
            "timing": timing_metadata,
            "fault_reporting": {
                "packet_loss": "uses an explicit missingness signal and is not purely residual-based detection",
                "value_faults": list(VALUE_ONLY_FAULTS),
                "mixed": "contains both explicit missingness and value corruption and is reported separately",
            },
        }
    }
    for domain in DOMAINS:
        summary[domain] = {}
        for model in ["lstm", "xlstm"]:
            selected = [r for r in rows if r["domain"] == domain and r["model"] == model]
            summary[domain][model] = _summarize_model_rows(selected)
    return summary


def _plot_fault_f1(rows: list[dict], path: Path) -> None:
    labels = []
    packet_values = []
    value_values = []
    mixed_values = []
    for domain in DOMAINS:
        for model in ["lstm", "xlstm"]:
            selected = [r for r in rows if r["domain"] == domain and r["model"] == model]
            labels.append(f"{domain}\n{model}")
            packet_values.append(
                float(np.mean([float(r["f1"]) for r in selected if r["scenario"] == "packet_loss"]))
            )
            value_values.append(
                float(np.mean([float(r["f1"]) for r in selected if r["scenario"] in VALUE_ONLY_FAULTS]))
            )
            mixed_values.append(float(np.mean([float(r["f1"]) for r in selected if r["scenario"] == "mixed"])))

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, packet_values, width, label="Packet loss (explicit missingness)")
    ax.bar(x, value_values, width, label="Value-only fault mean")
    ax.bar(x + width, mixed_values, width, label="Mixed fault")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Fault-detection F1")
    ax.set_ylim(0, 1)
    ax.set_title("Fault detection by evidence type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the xLSTM telemetry assurance benchmark.")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run_benchmark(args.output, smoke=args.smoke)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
