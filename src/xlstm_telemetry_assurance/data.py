from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TelemetryBatch:
    clean: np.ndarray
    observed: np.ndarray
    missing: np.ndarray
    fault_mask: np.ndarray
    channel_names: tuple[str, ...]


SPACECRAFT_CHANNELS = (
    "battery_soc",
    "bus_voltage",
    "temperature",
    "wheel_speed",
    "pointing_error",
    "payload_current",
)

ROBOTICS_CHANNELS = (
    "joint_position",
    "joint_velocity",
    "motor_current",
    "motor_temperature",
    "vibration",
    "tool_load",
)


def generate_clean_telemetry(domain: str, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if domain == "spacecraft":
        return _spacecraft(length, rng)
    if domain == "robotics":
        return _robotics(length, rng)
    raise ValueError(f"Unknown domain: {domain}")


def channel_names(domain: str) -> tuple[str, ...]:
    if domain == "spacecraft":
        return SPACECRAFT_CHANNELS
    if domain == "robotics":
        return ROBOTICS_CHANNELS
    raise ValueError(f"Unknown domain: {domain}")


def _spacecraft(length: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(length, dtype=np.float32)
    orbit = 2.0 * np.pi * t / 96.0
    sun = np.clip(np.sin(orbit), 0.0, None)
    eclipse = (sun < 0.08).astype(np.float32)
    payload = ((t.astype(int) // 64) % 4 == 1).astype(np.float32)
    slew = ((t.astype(int) % 120) > 88).astype(np.float32) * ((t.astype(int) % 120) < 101).astype(np.float32)

    soc = np.empty(length, dtype=np.float32)
    temp = np.empty(length, dtype=np.float32)
    wheel = np.empty(length, dtype=np.float32)
    soc[0] = 0.72
    temp[0] = 18.0
    wheel[0] = 2200.0
    for i in range(1, length):
        charge = 0.006 * sun[i] - 0.0028 - 0.002 * payload[i] - 0.0012 * slew[i]
        soc[i] = np.clip(soc[i - 1] + charge + rng.normal(0, 0.0007), 0.2, 0.98)
        target_temp = 13.0 + 10.0 * sun[i] + 5.0 * payload[i] + 2.0 * slew[i]
        temp[i] = temp[i - 1] + 0.075 * (target_temp - temp[i - 1]) + rng.normal(0, 0.12)
        wheel_target = 2100 + 950 * slew[i] + 130 * np.sin(orbit[i] * 0.5)
        wheel[i] = wheel[i - 1] + 0.16 * (wheel_target - wheel[i - 1]) + rng.normal(0, 20)

    bus_voltage = 26.5 + 3.4 * soc - 0.25 * eclipse - 0.38 * payload + rng.normal(0, 0.045, length)
    pointing = 0.015 + 0.000025 * np.abs(wheel - 2200) + 0.11 * slew + rng.normal(0, 0.004, length)
    payload_current = 0.65 + 1.9 * payload + 0.22 * sun + 0.35 * slew + rng.normal(0, 0.045, length)
    return np.column_stack([soc, bus_voltage, temp, wheel, pointing, payload_current]).astype(np.float32)


def _robotics(length: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(length, dtype=np.float32)
    phase = t.astype(int) % 180
    mode = np.zeros(length, dtype=np.int64)
    mode[(phase >= 30) & (phase < 90)] = 1
    mode[(phase >= 90) & (phase < 135)] = 2
    mode[phase >= 135] = 3

    command = np.zeros(length, dtype=np.float32)
    command[mode == 1] = 0.75 * np.sin(2 * np.pi * t[mode == 1] / 45.0)
    command[mode == 2] = 0.34 * np.sin(2 * np.pi * t[mode == 2] / 70.0)
    command[mode == 3] = 0.92 * np.sin(2 * np.pi * t[mode == 3] / 32.0)

    pos = np.empty(length, dtype=np.float32)
    vel = np.empty(length, dtype=np.float32)
    temp = np.empty(length, dtype=np.float32)
    pos[0], vel[0], temp[0] = 0.0, 0.0, 29.0
    for i in range(1, length):
        err = command[i] - pos[i - 1]
        vel[i] = 0.76 * vel[i - 1] + 0.22 * err + rng.normal(0, 0.012)
        pos[i] = pos[i - 1] + vel[i] + rng.normal(0, 0.003)
        effort = abs(err) + 0.35 * abs(vel[i]) + 0.45 * (mode[i] == 3)
        temp[i] = temp[i - 1] + 0.035 * (28.0 + 8.0 * effort - temp[i - 1]) + rng.normal(0, 0.05)

    error = command - pos
    current = 1.2 + 3.0 * np.abs(error) + 0.7 * np.abs(vel) + 0.9 * (mode == 3) + rng.normal(0, 0.05, length)
    vibration = 0.06 + 0.22 * np.abs(vel) + 0.13 * (mode == 3) + 0.08 * np.abs(np.sin(t / 4.0)) + rng.normal(0, 0.012, length)
    tool_load = 0.15 + 0.8 * (mode == 2) + 1.3 * (mode == 3) + 0.12 * np.abs(error) + rng.normal(0, 0.035, length)
    return np.column_stack([pos, vel, current, temp, vibration, tool_load]).astype(np.float32)


def inject_fault(clean: np.ndarray, fault: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = clean.copy()
    missing = np.zeros_like(clean, dtype=bool)
    fault_mask = np.zeros(clean.shape[0], dtype=bool)
    length, channels = clean.shape
    start = int(length * 0.58)
    duration = max(8, int(length * 0.12))
    stop = min(length, start + duration)

    if fault == "packet_loss":
        ch = 1
        missing[start:stop:2, ch] = True
        observed[missing] = np.nan
        fault_mask[start:stop:2] = True
    elif fault == "spike":
        ch = min(4, channels - 1)
        spike_stop = min(length, start + max(5, duration // 5))
        scale = float(clean[:, ch].std() + 1e-6)
        observed[start:spike_stop, ch] += 6.0 * scale
        fault_mask[start:spike_stop] = True
    elif fault == "stuck":
        ch = 2
        observed[start:stop, ch] = observed[start, ch]
        fault_mask[start:stop] = True
    elif fault == "drift":
        ch = 0
        scale = float(clean[:, ch].std() + 1e-6)
        drift = np.linspace(0.0, 3.5 * scale, stop - start, dtype=np.float32)
        observed[start:stop, ch] += drift
        fault_mask[start:stop] = True
    elif fault == "regime_shift":
        shift = np.linspace(0.6, 1.4, channels, dtype=np.float32) * clean.std(axis=0)
        observed[start:stop] += shift
        fault_mask[start:stop] = True
    elif fault == "mixed":
        ch = 1
        missing[start : start + duration // 3 : 2, ch] = True
        observed[missing] = np.nan
        second = start + duration // 3
        third = start + (2 * duration) // 3
        spike_ch = min(4, channels - 1)
        observed[second:third, spike_ch] += 4.5 * (clean[:, spike_ch].std() + 1e-6)
        drift_ch = 0
        observed[third:stop, drift_ch] += np.linspace(
            0.0,
            3.0 * (clean[:, drift_ch].std() + 1e-6),
            stop - third,
            dtype=np.float32,
        )
        fault_mask[start:stop] = True
    else:
        raise ValueError(f"Unknown fault: {fault}")

    return observed, missing, fault_mask


def forward_fill(values: np.ndarray) -> np.ndarray:
    filled = values.copy()
    for ch in range(filled.shape[1]):
        last = 0.0
        valid_seen = False
        for i in range(filled.shape[0]):
            if np.isfinite(filled[i, ch]):
                last = float(filled[i, ch])
                valid_seen = True
            else:
                filled[i, ch] = last if valid_seen else 0.0
    return filled


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def prepare_observed_inputs(
    observed: np.ndarray,
    missing: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    filled = forward_fill(observed)
    standardized = standardize(filled, mean, std)
    inputs = np.concatenate([standardized, missing.astype(np.float32)], axis=1)
    return inputs.astype(np.float32), standardized


def build_windows(inputs: np.ndarray, targets: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, indices = [], [], []
    for i in range(seq_len, len(inputs)):
        xs.append(inputs[i - seq_len : i])
        ys.append(targets[i])
        indices.append(i)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(indices, dtype=np.int64)
