import numpy as np
import pytest

from xlstm_telemetry_assurance.data import (
    build_windows, generate_clean_telemetry, inject_fault, prepare_observed_inputs, standardize,
)


def test_generators_are_deterministic():
    first = generate_clean_telemetry("spacecraft", 120, 5)
    second = generate_clean_telemetry("spacecraft", 120, 5)
    assert np.array_equal(first, second)


def test_packet_loss_preserves_explicit_missingness():
    clean = generate_clean_telemetry("robotics", 160, 3)
    observed, missing, fault_mask = inject_fault(clean, "packet_loss", 7)
    assert missing.any()
    assert fault_mask.any()
    mean = clean.mean(axis=0)
    std = clean.std(axis=0) + 1e-6
    inputs, _ = prepare_observed_inputs(observed, missing, mean, std)
    channels = clean.shape[1]
    assert inputs.shape[1] == channels * 2
    assert np.isfinite(inputs).all()
    assert np.array_equal(inputs[:, channels:].astype(bool), missing)


def test_masked_finite_placeholder_is_excluded_from_forward_fill():
    values = np.array([[2.0], [999.0], [np.nan], [4.0]], dtype=np.float32)
    missing = np.array([[False], [True], [False], [False]])
    inputs, targets = prepare_observed_inputs(values, missing, np.zeros(1), np.ones(1))
    np.testing.assert_array_equal(targets[:, 0], [2, 2, 2, 4])
    np.testing.assert_array_equal(inputs[:, 1], [0, 1, 1, 0])
    assert values[1, 0] == 999 and np.isnan(values[2, 0])
    assert not missing[2, 0]


def test_nonfinite_readings_are_explicitly_missing_and_leading_gaps_use_zero():
    values = np.array([[np.inf, np.nan], [-np.inf, 3], [4, 5]], dtype=np.float32)
    inputs, _ = prepare_observed_inputs(values, np.zeros_like(values, dtype=bool), np.zeros(2), np.ones(2))
    np.testing.assert_array_equal(inputs[:, :2], [[0, 0], [0, 3], [4, 5]])
    np.testing.assert_array_equal(inputs[:, 2:], [[1, 1], [1, 0], [0, 0]])


@pytest.mark.parametrize("missing", [np.zeros((4, 1)), np.zeros((4, 2), dtype=bool)])
def test_preprocessing_rejects_malformed_missing_masks(missing):
    with pytest.raises(ValueError, match="boolean matrix matching"):
        prepare_observed_inputs(np.ones((4, 1)), missing, np.zeros(1), np.ones(1))


@pytest.mark.parametrize("std", [np.array([0.0]), np.array([-1.0]), np.array([np.nan]), np.ones((1, 1))])
def test_standardization_rejects_invalid_scale(std):
    with pytest.raises(ValueError):
        standardize(np.ones((4, 1)), np.zeros(1), std)


def test_windows_keep_one_step_target_alignment_and_reject_short_or_mismatched_data():
    inputs = np.arange(8, dtype=np.float32).reshape(-1, 1)
    x, y, indices = build_windows(inputs, inputs + 100, 3)
    np.testing.assert_array_equal(x[0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(y[:, 0], [103, 104, 105, 106, 107])
    np.testing.assert_array_equal(indices, np.arange(3, 8))
    for seq_len in (0, -1, 8, True):
        with pytest.raises(ValueError):
            build_windows(inputs, inputs, seq_len)
    with pytest.raises(ValueError, match="timeline lengths"):
        build_windows(inputs, inputs[:-1], 3)


@pytest.mark.parametrize("length", [1, 2, 5, 10])
def test_mixed_fault_handles_short_timelines(length):
    clean = np.ones((length, 6), dtype=np.float32)
    observed, missing, fault = inject_fault(clean, "mixed", 11)
    assert observed.shape == missing.shape == clean.shape
    assert fault.shape == (length,)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_generator_rejects_invalid_length(length):
    with pytest.raises(ValueError, match="positive integer"):
        generate_clean_telemetry("spacecraft", length, 11)
