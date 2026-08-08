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
from .training import guarded_adaptation, predict, train_model

DEFAULT_SEEDS = [11, 29, 47]
DOMAINS = ["spacecraft", "robotics"]
FAULTS = ["packet_loss", "spike", "stuck", "drift", "regime_shift", "mixed"]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _latency_ms(model: torch.nn.Module, x: torch.Tensor, repeats: int = 120) -> float:
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
    x, _, indices = build_windows(inputs, observed_std, seq_len)
    _, y_clean, _ = build_windows(inputs, clean_std, seq_len)
    pred_mean, pred_std = predict(model, x)
    scores = np.mean(np.abs(y_clean - pred_mean) / np.maximum(pred_std, 1e-4), axis=1)
    miss_step = missing[indices].any(axis=1).astype(np.float32)
    scores = scores + 4.0 * miss_step
    labels = fault_mask[indices].astype(np.int64)
    preds = (scores > threshold).astype(np.int64)
    metrics = binary_metrics(labels, preds)
    metrics["rmse"] = rmse(y_clean, pred_mean)
    metrics["coverage_90"] = coverage(y_clean, pred_mean, pred_std, z=1.6448536269514722)
    metrics["gaussian_nll"] = gaussian_nll(y_clean, pred_mean, pred_std)
    return metrics


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
    seeds = [11] if smoke else DEFAULT_SEEDS
    seq_len = 16 if smoke else 24
    train_length = 520 if smoke else 1050
    eval_length = 300 if smoke else 700
    epochs = 2 if smoke else 10
    hidden = 20 if smoke else 32

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
                train_model(model, train_x, train_y, epochs=epochs, lr=0.004 if smoke else 0.003)
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
                latency = _latency_ms(model, torch.from_numpy(train_x[:16]))
                common = {
                    "domain": domain,
                    "model": model_name,
                    "seed": seed,
                    "parameters": count_parameters(model),
                    "latency_ms": latency,
                }
                rows.append(
                    {
                        **common,
                        "scenario": "clean",
                        "rmse": clean_metrics["rmse"],
                        "coverage_90": clean_metrics["coverage_90"],
                        "f1": "",
                        "precision": "",
                        "recall": "",
                        "false_alarm_rate": clean_metrics["false_positive_rate"],
                        "adaptation_accepted": "",
                    }
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
                        {
                            **common,
                            "scenario": fault,
                            "rmse": clean_metrics["rmse"],
                            "coverage_90": clean_metrics["coverage_90"],
                            "f1": metrics["f1"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "false_alarm_rate": clean_metrics["false_positive_rate"],
                            "adaptation_accepted": "",
                        }
                    )

                adaptation = _adaptation_check(model, domain, seed, mean, std, seq_len, smoke=smoke)
                rows.append(
                    {
                        **common,
                        "scenario": "adaptation",
                        "rmse": "",
                        "coverage_90": "",
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
        "f1",
        "precision",
        "recall",
        "false_alarm_rate",
        "parameters",
        "latency_ms",
        "adaptation_accepted",
    ]
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot_fault_f1(rows, output / "fault_detection_f1.png")
    return summary


def _summarize(rows: list[dict]) -> dict:
    summary: dict = {}
    for domain in DOMAINS:
        summary[domain] = {}
        for model in ["lstm", "xlstm"]:
            selected = [r for r in rows if r["domain"] == domain and r["model"] == model]
            clean = [r for r in selected if r["scenario"] == "clean"]
            faults = [r for r in selected if r["scenario"] in FAULTS]
            mixed = [r for r in selected if r["scenario"] == "mixed"]
            adaptations = [r for r in selected if r["scenario"] == "adaptation"]
            summary[domain][model] = {
                "clean_rmse_mean": float(np.mean([float(r["rmse"]) for r in clean])),
                "clean_rmse_std": float(np.std([float(r["rmse"]) for r in clean])),
                "clean_coverage_90_mean": float(np.mean([float(r["coverage_90"]) for r in clean])),
                "fault_f1_mean": float(np.mean([float(r["f1"]) for r in faults])),
                "mixed_fault_f1_mean": float(np.mean([float(r["f1"]) for r in mixed])),
                "false_alarm_rate_mean": float(np.mean([float(r["false_alarm_rate"]) for r in clean])),
                "parameters": int(clean[0]["parameters"]),
                "latency_ms_mean": float(np.mean([float(r["latency_ms"]) for r in clean])),
                "adaptation_accept_rate": float(np.mean([1.0 if r["adaptation_accepted"] else 0.0 for r in adaptations])),
            }
    return summary


def _plot_fault_f1(rows: list[dict], path: Path) -> None:
    labels = []
    values = []
    for domain in DOMAINS:
        for model in ["lstm", "xlstm"]:
            faults = [
                float(r["f1"])
                for r in rows
                if r["domain"] == domain and r["model"] == model and r["scenario"] in FAULTS
            ]
            labels.append(f"{domain}\n{model}")
            values.append(float(np.mean(faults)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Mean fault-detection F1")
    ax.set_ylim(0, 1)
    ax.set_title("Residual-based assurance under controlled telemetry faults")
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
