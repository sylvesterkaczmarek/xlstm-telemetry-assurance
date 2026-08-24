# Method

## Telemetry generators

The benchmark uses two deterministic synthetic physical-system generators.

### Spacecraft

The spacecraft track contains six channels: battery state of charge, bus voltage, temperature, reaction-wheel speed, pointing-error proxy and payload current. A synthetic schedule changes solar input, payload demand and slew activity. The state variables have memory, so the series is not an independent collection of sinusoids.

### Robotics

The robotics track contains joint position, joint velocity, motor current, motor temperature, vibration and tool load. The generator switches between idle, nominal manipulation, precision work and higher-load operation.

These are structured test fixtures, not digital twins.

## Fault injection

All faults are injected into a clean counterfactual trajectory while retaining the clean target for controlled evaluation.

- `packet_loss`: observations become unavailable and a missingness mask is raised.
- `spike`: a short additive sensor excursion.
- `stuck`: a channel is held at a constant value.
- `drift`: an increasing sensor bias.
- `regime_shift`: a persistent multi-channel offset representing an unseen operating regime.
- `mixed`: packet loss, a spike and a drift occur in one stream.

The model receives imputed normalised values plus the missingness mask. Missing values are forward-filled before standardisation, but missing target channels are excluded from the residual and missingness contributes an explicit anomaly penalty. Packet-loss F1 is therefore not purely learned residual detection. The mixed scenario also contains explicit missingness and is kept separate from the value-only mean.

## Models

Both models use the same input representation and probabilistic forecast head.

### LSTM

A one-layer PyTorch LSTM followed by a residual Gaussian head.

### xLSTM-style recurrence

A compact stabilised recurrent cell motivated by the sLSTM recurrence in xLSTM. Input and forget gates are represented in log space and a state-dependent stabiliser rescales exponential gates before the cell and normaliser are updated.

The implementation is deliberately small enough to inspect. It is not a wrapper around the official xLSTM package and does not claim exact architectural equivalence to every component of xLSTMTime.

## Uncertainty and runtime anomaly score

The forecast head predicts a mean and positive standard deviation per channel. Training minimises diagonal Gaussian negative log-likelihood.

A separate clean calibration sequence sets the anomaly threshold from the 99th percentile of uncertainty-normalised residual scores. At runtime, residuals are computed against observed telemetry, not the clean counterfactual retained by the benchmark. The clean counterfactual is used only for forecast metrics and fault ground truth.

## Guarded adaptation

Only the output head is adapted. The benchmark saves the current parameters, applies a candidate update on an adaptation window, evaluates a disjoint guard window, accepts the candidate if guard loss remains within tolerance, and otherwise restores the saved parameters.

A real system would require a separate trusted guard signal or independent validation mechanism.

## Timing measurement

`window_inference_latency_ms` measures wall-clock time for one complete model forward pass over one input window on the host CPU after warm-up. The full benchmark uses a 24-sample window and averages 120 repeated forward passes per model instance.

This is a host benchmark only. It is hardware- and load-dependent, is not a per-recurrent-step measurement, and does not represent spacecraft or robot real-time performance or worst-case execution time.
