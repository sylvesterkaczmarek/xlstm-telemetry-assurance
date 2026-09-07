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


def test_gaussian_nll_matches_known_density():
    from xlstm_telemetry_assurance.metrics import gaussian_nll

    result = gaussian_nll(np.array([0.0, 2.0]), np.zeros(2), np.array([1.0, 2.0]))
    expected = 0.5 * np.log(2 * np.pi) + 0.5 * np.log(2.0) + 0.25
    np.testing.assert_allclose(result, expected)


def test_metrics_avoid_representable_float32_overflow():
    from xlstm_telemetry_assurance.metrics import gaussian_nll

    target = np.array([1e20], dtype=np.float32)
    mean = np.zeros_like(target)
    np.testing.assert_allclose(rmse(target, mean), float(target[0]))
    np.testing.assert_allclose(gaussian_nll(target, mean, target), np.log(float(target[0])) + 0.5 * np.log(2 * np.pi) + 0.5)
    assert coverage(target, mean, target) == 1.0


def test_rmse_avoids_float64_square_overflow():
    np.testing.assert_allclose(rmse(np.array([1e200, -1e200]), np.zeros(2)), 1e200)


def test_forecast_metrics_reject_broadcasting_and_invalid_arrays():
    import pytest

    from xlstm_telemetry_assurance.metrics import gaussian_nll

    metrics = [rmse, lambda y, m: gaussian_nll(y, m, np.ones_like(y)), lambda y, m: coverage(y, m, np.ones_like(y))]
    for metric in metrics:
        with pytest.raises(ValueError, match="same shape"):
            metric(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
        for invalid in (np.array([]), np.array([np.nan]), np.array([np.inf]), np.array([1j]), np.array(["1"])):
            with pytest.raises(ValueError):
                metric(invalid, invalid)


def test_uncertainty_metrics_reject_invalid_scale():
    import pytest

    from xlstm_telemetry_assurance.metrics import gaussian_nll

    for metric in (gaussian_nll, coverage):
        for std in (np.array([0.0]), np.array([-1.0]), np.array([np.nan]), np.ones((1, 1))):
            with pytest.raises(ValueError):
                metric(np.zeros(1), np.zeros(1), std)


def test_coverage_checks_width_and_includes_boundaries():
    import pytest

    target = np.array([-2.0, 0.0, 2.0, 3.0])
    assert coverage(target, np.zeros(4), np.ones(4), z=2.0) == 0.75
    assert coverage(target, np.zeros(4), np.ones(4), z=0.0) == 0.25
    for z in (-1, np.nan, np.inf, True, "1"):
        with pytest.raises(ValueError, match="z must"):
            coverage(target, np.zeros(4), np.ones(4), z=z)


def test_binary_metrics_reject_broadcasting_and_invalid_labels():
    import pytest

    with pytest.raises(ValueError, match="same shape"):
        binary_metrics(np.array([[0], [1]]), np.array([0, 1]))
    for invalid in (np.array([]), np.array([np.nan]), np.array([2]), np.array([-1]), np.array(["1"])):
        with pytest.raises(ValueError):
            binary_metrics(invalid, np.ones_like(invalid))
        with pytest.raises(ValueError):
            binary_metrics(np.ones_like(invalid), invalid)
    assert binary_metrics(np.array([False, True]), np.array([False, True]))["f1"] == 1.0


def test_unrepresentable_metrics_fail_explicitly():
    import pytest

    from xlstm_telemetry_assurance.metrics import gaussian_nll

    with pytest.raises(ValueError, match="numeric range"):
        rmse(np.array([1e308]), np.array([-1e308]))
    with pytest.raises(ValueError, match="numeric range"):
        gaussian_nll(np.array([1e308]), np.zeros(1), np.ones(1))
