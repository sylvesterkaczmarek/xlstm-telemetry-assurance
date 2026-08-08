# Method

## Telemetry generators

The benchmark uses two deterministic synthetic physical-system generators.

### Spacecraft

The spacecraft track contains six normalized channels derived from coupled low-order dynamics:

- battery state of charge;
- bus voltage;
- temperature;
- reaction-wheel speed;
- pointing-error proxy;
- payload current.

A synthetic operating schedule changes solar input, payload demand and slew activity. Thermal state, battery state and wheel activity have memory, so the series is not an independent collection of sinusoids.

### Robotics

The robotics track contains six normalized channels:

- joint position;
- joint velocity;
- motor current;
- motor temperature;
- vibration;
- tool load.

The generator switches between idle, nominal manipulation, precision work and higher-load operation.

These are structured test fixtures, not digital twins.

## Fault injection

All faults are injected into a clean counterfactual trajectory while retaining the clean target for controlled evaluation.

- `packet_loss`: observations become unavailable and a missingness mask is raised.
- `spike`: a short additive sensor excursion.
- `stuck`: a channel is held at a constant value.
- `drift`: an increasing sensor bias.
- `regime_shift`: a persistent multi-channel offset representing an unseen operating regime.
- `mixed`: packet loss, a spike and a drift occur in one stream.

The model receives both imputed normalized values and the missingness mask. Missing values are forward-filled before standardization; missingness is therefore not silently hidden.

## Models

Both models use the same input representation and probabilistic forecast head.

### LSTM

A one-layer PyTorch LSTM followed by a residual Gaussian head.

### xLSTM-style recurrence

A compact stabilized recurrent cell motivated by the sLSTM recurrence in xLSTM. Input and forget gates are represented in log space. A state-dependent stabilizer rescales exponential gates before the cell and normalizer are updated.

The implementation is deliberately small enough to inspect. It is not a wrapper around the official xLSTM package and does not claim exact architectural equivalence to every component of xLSTMTime.

## Uncertainty and runtime anomaly score

The forecast head predicts a mean and positive standard deviation per channel. Training minimizes diagonal Gaussian negative log-likelihood.

A separate clean calibration sequence sets the anomaly threshold from the 99th percentile of uncertainty-normalized residual scores. At runtime, residuals are computed against the **observed telemetry available to the system**, not against the clean counterfactual retained by the synthetic benchmark. If a target channel is missing, its forward-filled value is excluded from the residual and missingness contributes an explicit anomaly penalty.

The clean counterfactual is used only to measure forecast error and to provide ground-truth fault labels in the controlled experiment. It is not available to the runtime detector.

## Guarded adaptation

Only the output head is adapted. The benchmark:

1. saves the current head parameters;
2. applies a candidate gradient update on an adaptation window;
3. measures loss on a disjoint guard window;
4. accepts the candidate if guard loss has not worsened beyond tolerance;
5. otherwise restores the saved parameters.

The synthetic benchmark can evaluate the candidate against the clean counterfactual target. A real system would need a separate trusted guard signal or independent validation mechanism.
