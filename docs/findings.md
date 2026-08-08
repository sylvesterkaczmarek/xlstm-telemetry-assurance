# Findings

The checked-in benchmark is a controlled synthetic experiment rather than a deployment claim.

## Main result

Across three deterministic seeds, the xLSTM-style recurrent model reduces clean one-step forecasting error relative to the ordinary LSTM in both telemetry domains.

| Domain | LSTM RMSE | xLSTM-style RMSE |
|---|---:|---:|
| Spacecraft | 0.560 ± 0.023 | **0.231 ± 0.006** |
| Robotics | 0.422 ± 0.030 | **0.167 ± 0.002** |

The result is in standardized telemetry units.

## Assurance result

The lower forecasting error does not translate into better anomaly detection. Mean fault-detection F1 is higher for the simpler LSTM in both tracks:

| Domain | LSTM mean fault F1 | xLSTM-style mean fault F1 |
|---|---:|---:|
| Spacecraft | **0.355** | 0.239 |
| Robotics | **0.573** | 0.227 |

This is the central negative result. A recurrent predictor that tracks persistent drift or a regime change more effectively can reduce the residual used by a residual-based detector. Prediction quality and assurance value are therefore not interchangeable objectives.

## Adaptation result

The guarded head-adaptation experiment generated 12 candidate updates across models, domains and seeds. Three were accepted; nine exceeded the allowed guard-loss degradation and were rolled back.

The mechanism is deliberately conservative, but this result should not be interpreted as evidence that the adaptation policy is deployment-safe. The guard target is available because the benchmark retains the clean counterfactual synthetic trajectory.

## Interpretation

The benchmark supports three narrow conclusions:

1. A stabilized xLSTM-style recurrence can be useful for compact physical telemetry forecasting.
2. Forecast quality alone is a poor proxy for anomaly-detection quality.
3. Candidate online adaptation can be isolated and automatically rolled back when an independent guard objective deteriorates.

It does not establish architecture superiority, flight readiness, robotic functional safety, or general anomaly-detection performance.
