from __future__ import annotations

import numpy as np


def rmse(target: np.ndarray, mean: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - mean) ** 2)))


def gaussian_nll(target: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    var = np.maximum(std**2, 1e-8)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (target - mean) ** 2 / var))


def coverage(target: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = 1.6448536269514722) -> float:
    lower = mean - z * std
    upper = mean + z * std
    return float(np.mean((target >= lower) & (target <= upper)))


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    tp = int(np.sum(labels & predictions))
    fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions))
    tn = int(np.sum(~labels & ~predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }
