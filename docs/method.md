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

The model receives imputed normalised values plus a missingness mask. A channel is unavailable when its explicit mask is true **or** its observed value is non-finite. Finite placeholders in masked channels are discarded before forward filling, so an arbitrary placeholder cannot become a measurement or the value carried into later gaps. Forward filling uses the last available observation in that channel, with zero as the initial fallback, followed by standardisation using training statistics.

Missing target channels are excluded from the residual. A timestep with any unavailable channel receives an explicit anomaly penalty of 4.0; if every channel is unavailable, the residual contribution is zero. Packet-loss F1 therefore includes direct missingness detection. The mixed scenario also contains explicit missingness and is kept separate from the value-only mean.

## Models

Both models use the same input representation and probabilistic forecast head.

### LSTM

A one-layer PyTorch LSTM followed by a residual Gaussian head.

### xLSTM-style recurrence

A compact stabilised recurrent cell motivated by the sLSTM recurrence in xLSTM. Input and forget gates are represented in log space and a state-dependent stabiliser rescales exponential gates before the cell and normaliser are updated.

Writing the gate logits as $a_t$ and $b_t$, the implemented updates are

$$
m_t = \max(a_t, b_t + m_{t-1}), \qquad
i_t = \exp(a_t - m_t), \qquad
f_t = \exp(b_t + m_{t-1} - m_t),
$$

$$
c_t = f_t c_{t-1} + i_t z_t, \qquad
n_t = f_t n_{t-1} + i_t, \qquad
h_t = o_t c_t / n_t.
$$

Here $z_t$ is a tanh candidate and $o_t$ is a sigmoid output gate. Each input window starts with $h_0=c_0=n_0=0$ and $m_0=-\infty$, representing an empty memory. For finite gate logits this gives $i_1=1$, $f_1=0$ and $n_1=1$, preserving the first candidate even when its input-gate logit is very negative. Subsequent states have $n_t\geq1$, so artificial gate and denominator floors are unnecessary. Tests compare both outputs and gradients with the corresponding unscaled recurrence over a bounded sequence where direct exponentiation is safe.

This implements the exponential-forget-gate option in the [xLSTM paper](https://arxiv.org/abs/2405.04517), equations 8–17. The repository provides a compact single-cell research variant; architectural differences from the full xLSTM and xLSTMTime implementations remain.

## Uncertainty and runtime anomaly score

The forecast head predicts a mean and positive standard deviation per channel. Training minimises diagonal Gaussian negative log-likelihood (NLL), including the Gaussian normalising constant, in standardised channel units. A Gaussian density can exceed one, so valid NLL values can be negative. Predictions, targets and scales must have matching nonempty shapes, finite values and positive standard deviations; invalid numerical results raise an error.

A separate clean calibration sequence sets the anomaly threshold from the 99th percentile of uncertainty-normalised residual scores. Runtime residuals use available observed telemetry. The clean counterfactual supplies forecast metrics, fault ground truth and the supervised targets used in the synthetic adaptation experiment. A fitted Gaussian head and a clean calibration quantile do not establish uncertainty calibration under every fault or operating condition.

## Guarded adaptation

Only the output head is adapted. The benchmark splits the raw adaptation timeline into an earlier 70% segment and a later 30% guard segment, then constructs forecasting windows within each segment. This prevents the same raw timestep from appearing in both the adaptation and guard windows. Both segments still belong to one synthetic trajectory and can be temporally correlated; the split does not establish statistical independence.

Before the candidate update, the benchmark saves the model state and measures guard NLL. The candidate is accepted when

$$
L_{\mathrm{after}} \leq L_{\mathrm{before}} + \tau\lvert L_{\mathrm{before}}\rvert,
$$

where $\tau=0.01$ is the default tolerance. This permits relative degradation for either sign of NLL; a zero baseline permits no degradation. For example, a baseline NLL of $-1$ gives an acceptance limit of $-0.99$.

During candidate updates the frozen feature extractor stays in evaluation mode and the head enters training mode. Rejection or an exception restores the saved parameters and buffers. Every outcome restores the caller's module modes, parameter gradient flags and existing gradients; numerical failures raise an error. The separate `predict` helper also preserves the caller's module modes.

Each adaptation row in `metrics.csv` records `guard_loss_before`, `guard_loss_after` and `adaptation_accepted`. The after value describes the evaluated candidate even when that candidate was rejected and rolled back, allowing the decision to be checked against the tolerance rule.

Both adaptation and guard targets come from the known clean synthetic trajectory. Deployment would require a trusted source of targets or an independent validation mechanism. Passing this guard is evidence for this particular supervised check and does not establish operational safety or improvement on unseen streams.

## Timing measurement

`window_inference_latency_ms` measures wall-clock time for one complete model forward pass over one input window on the host CPU after warm-up. The full benchmark uses a 24-sample window and averages 120 repeated forward passes per model instance.

This is a host benchmark only. It is hardware- and load-dependent, is not a per-recurrent-step measurement, and does not represent spacecraft or robot real-time performance or worst-case execution time.
