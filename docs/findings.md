# Findings

The checked-in benchmark is a controlled synthetic experiment rather than a deployment claim.

## Forecasting

Across three deterministic seeds, the ordinary LSTM has lower clean one-step RMSE in both telemetry domains.

| Domain | LSTM RMSE | xLSTM-style RMSE | LSTM Gaussian NLL | xLSTM-style Gaussian NLL |
|---|---:|---:|---:|---:|
| Spacecraft | **0.232 ± 0.003** | 0.285 ± 0.007 | **0.494** | 0.495 |
| Robotics | **0.320 ± 0.007** | 0.361 ± 0.012 | 0.575 | **0.501** |

RMSE is expressed in standardised telemetry units. Gaussian NLL captures both mean prediction and predicted scale, so its ranking need not match RMSE exactly.

## Fault detection

Packet loss must be separated from value-only faults because the detector receives an explicit missingness signal.

| Domain | Model | Packet-loss F1 | Value-only F1 mean | Mixed F1 | Overall six-fault F1 |
|---|---|---:|---:|---:|---:|
| Spacecraft | LSTM | 0.913 | **0.046** | **0.318** | **0.236** |
| Spacecraft | xLSTM-style | **0.916** | 0.046 | 0.298 | 0.233 |
| Robotics | LSTM | 0.916 | 0.025 | 0.281 | 0.216 |
| Robotics | xLSTM-style | **0.920** | **0.025** | **0.281** | **0.217** |

The value-only mean covers spike, stuck, drift and regime shift. The mixed scenario contains packet loss, spike and drift and is therefore reported separately. Per-fault means are available in `results/benchmark/summary.json`.

The main result is that the strong packet-loss number is driven by explicit missingness evidence, while the simple residual detector remains weak on value-only faults. Forecast quality and value-fault detection quality are therefore separate objectives in this benchmark.

## Guarded adaptation

The guarded head-adaptation experiment generated 12 candidate updates. Nine were accepted and three were rolled back after guard loss exceeded the allowed tolerance.

This demonstrates rollback behaviour only. It is not evidence that the adaptation policy is deployment-safe because the synthetic experiment has access to a trusted clean counterfactual target.

## Host timing

The reported `window_inference_latency_ms` is one complete model forward pass over one 24-sample input window on the recorded host CPU. It is hardware-dependent and is not recurrent-timestep latency, spacecraft or robot real-time timing, or WCET.

## Interpretation

The benchmark supports three narrow conclusions:

1. The compact xLSTM-style recurrence can be evaluated reproducibly, but this implementation does not outperform the ordinary LSTM on clean RMSE.
2. Packet-loss detection with explicit missingness should not be conflated with learned residual detection of value faults.
3. Candidate online adaptation can be isolated and rolled back when an independent guard objective deteriorates.

It does not establish architecture superiority, flight readiness, robotic functional safety, general anomaly-detection performance or deployment timing guarantees.
