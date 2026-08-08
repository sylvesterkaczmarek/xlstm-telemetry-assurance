import numpy as np

from xlstm_telemetry_assurance.metrics import binary_metrics, coverage, rmse


def test_rmse_zero_for_identical_arrays():
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    assert rmse(values, values) == 0.0


def test_binary_metrics_known_case():
    labels = np.array([0, 1, 1, 0])
    predictions = np.array([0, 1, 0, 1])
    metrics = binary_metrics(labels, predictions)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_coverage_is_bounded():
    target = np.zeros((10, 2), dtype=np.float32)
    mean = np.zeros_like(target)
    std = np.ones_like(target)
    value = coverage(target, mean, std)
    assert 0.0 <= value <= 1.0
