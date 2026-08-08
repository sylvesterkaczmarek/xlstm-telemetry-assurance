# Findings

The checked-in benchmark is a controlled synthetic experiment rather than a deployment claim.

## Forecasting result

Across three deterministic seeds, the ordinary LSTM produces lower clean one-step RMSE than the compact xLSTM-style recurrence in both telemetry domains.

| Domain | LSTM RMSE | xLSTM-style RMSE |
|---|---:|---:|
| Spacecraft | **0.232 ± 0.003** | 0.285 ± 0.007 |
| Robotics | **0.320 ± 0.007** | 0.361 ± 0.012 |

RMSE is expressed in standardized telemetry units. The xLSTM-style model uses slightly fewer trainable parameters in this implementation, but its explicit Python recurrence is slower than PyTorch's optimized LSTM kernel.

## Assurance result

After correcting the runtime scoring semantics so that anomaly residuals use the **observed measurement available to the system**, neither recurrent model has a meaningful overall fault-detection advantage.

| Domain | LSTM mean fault F1 | xLSTM-style mean fault F1 |
|---|---:|---:|
| Spacecraft | **0.236** | 0.233 |
| Robotics | 0.216 | **0.217** |

The near-equality is more important than the third decimal place. Packet loss is detected reliably because missingness is represented explicitly. The simple residual detector remains weak on value-only faults such as stuck sensors, gradual drift and sustained regime shifts.

This result supports a narrower conclusion than the original benchmark output: forecasting quality and residual-based assurance are separate objectives, and a detector must be evaluated using only information that would actually exist at runtime.

## Guarded adaptation

The guarded head-adaptation experiment generated 12 candidate updates across models, domains and seeds. Nine were accepted; three exceeded the allowed guard-loss degradation and were rolled back.

The mechanism is deliberately conservative, but this result should not be interpreted as evidence that the adaptation policy is deployment-safe. The guard target is available because the benchmark retains the clean counterfactual synthetic trajectory.

## Interpretation

The benchmark supports three narrow conclusions:

1. A compact xLSTM-style recurrence can be evaluated reproducibly on structured spacecraft and robotics telemetry, but this implementation does not outperform the ordinary LSTM on clean forecasting.
2. Forecast quality alone is not a proxy for anomaly-detection quality, and runtime fault scoring must use observed telemetry rather than inaccessible clean counterfactual values.
3. Candidate online adaptation can be isolated and automatically rolled back when an independent guard objective deteriorates.

It does not establish architecture superiority, flight readiness, robotic functional safety, or general anomaly-detection performance.
