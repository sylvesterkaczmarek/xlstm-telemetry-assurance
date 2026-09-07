from __future__ import annotations

from numbers import Real

import numpy as np


def _real_array(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0 or values.size == 0 or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty real numeric array")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _forecast_arrays(
    target: np.ndarray, mean: np.ndarray, std: np.ndarray | None = None
) -> tuple[np.ndarray, ...]:
    target = _real_array("target", target)
    mean = _real_array("mean", mean)
    if mean.shape != target.shape:
        raise ValueError("target and mean must have the same shape")
    if std is None:
        return target, mean
    std = _real_array("std", std)
    if std.shape != target.shape:
        raise ValueError("target, mean and std must have the same shape")
    if not np.all(std > 0):
        raise ValueError("std must be strictly positive")
    return target, mean, std


def _stable_mean(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    return float(scale * np.mean(values / scale)) if scale else 0.0


def rmse(target: np.ndarray, mean: np.ndarray) -> float:
    target, mean = _forecast_arrays(target, mean)
    try:
        with np.errstate(over="raise", invalid="raise"):
            residual = target - mean
            scale = float(np.max(np.abs(residual)))
            return float(scale * np.sqrt(np.mean((residual / scale) ** 2))) if scale else 0.0
    except FloatingPointError as exc:
        raise ValueError("RMSE calculation exceeded the supported numeric range") from exc


def gaussian_nll(target: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    target, mean, std = _forecast_arrays(target, mean, std)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            residual = (target - mean) / std
            # Avoid squaring std, which can overflow even for representable NLL.
            terms = np.log(std) + 0.5 * np.log(2 * np.pi) + (residual / np.sqrt(2.0)) ** 2
            return _stable_mean(terms)
    except FloatingPointError as exc:
        raise ValueError("Gaussian NLL calculation exceeded the supported numeric range") from exc


def coverage(target: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = 1.6448536269514722) -> float:
    target, mean, std = _forecast_arrays(target, mean, std)
    if isinstance(z, (bool, np.bool_)) or not isinstance(z, Real) or not np.isfinite(z) or z < 0:
        raise ValueError("z must be finite and nonnegative")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return float(np.mean(np.abs((target - mean) / std) <= z))
    except FloatingPointError as exc:
        raise ValueError("coverage calculation exceeded the supported numeric range") from exc


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    for name, values in (("labels", labels), ("predictions", predictions)):
        if values.ndim == 0 or values.size == 0 or values.dtype.kind not in "biuf":
            raise ValueError(f"{name} must be a nonempty binary array")
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(f"{name} must contain only 0 or 1")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
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
