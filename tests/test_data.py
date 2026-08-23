import numpy as np

from xlstm_telemetry_assurance.data import generate_clean_telemetry, inject_fault, prepare_observed_inputs


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
