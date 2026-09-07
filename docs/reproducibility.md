# Reproducibility

## Environment

`pyproject.toml` is the authoritative definition of supported runtime and development dependencies. The [CI workflow](../.github/workflows/ci.yml) lists the tested Python versions and checks the pinned reference environment separately.

For normal development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

For the pinned Python 3.12 reference environment used by CI:

```bash
python -m pip install -r requirements-reference.txt
python -m pip install --no-build-isolation --no-deps -e .
```

`requirements-reference.txt` is a reproducibility snapshot, not the package dependency specification.

## Full benchmark

```bash
python -m xlstm_telemetry_assurance.benchmark --output results/benchmark
```

The default benchmark uses seeds `11`, `29` and `47` and writes:

- `metrics.csv`: per-seed, per-domain, per-model and per-scenario measurements, including the guard losses before and after each candidate adaptation and its acceptance decision;
- `summary.json`: aggregate forecasting, Gaussian NLL, packet-loss, value-only, mixed and per-fault statistics plus timing semantics;
- `fault_detection_f1.png`: packet-loss, value-only and mixed F1 comparison;
- `run_environment.json`: Git state when attributable to the executed package, SHA-256 hashes of every package Python source file, Python/platform/CPU information, package versions, benchmark configuration, deterministic settings and an overall SHA-256 fingerprint.

Git discovery starts from the executed package location and requires its provenance module to be tracked by that repository. An unrelated working directory or a wheel installed in an unrelated repository's virtual environment cannot supply the package's commit identity. When no attributable checkout exists, `git.available` is false and `git.commit_sha` is null. The `GITHUB_SHA` environment variable is not treated as verified package provenance.

The `source_sha256` mapping identifies all package `.py` files using package-relative paths. These hashes cover the benchmark implementation as well as the model and remain available for installed wheels and uncommitted source changes. A Git commit with `dirty: true` identifies the checkout base; use the source hashes to distinguish the code used by that run. The overall fingerprint includes the source hashes, effective configuration and recorded environment. Metadata with non-finite values is rejected before writing, preserving any existing provenance file.

## Smoke benchmark

```bash
python -m xlstm_telemetry_assurance.benchmark --smoke --output results/smoke
```

Smoke mode uses a shorter stream and one seed. It validates the pipeline rather than reproducing the checked-in headline numbers. CI verifies that smoke mode also produces `run_environment.json`.

## Determinism

Each benchmark run fixes Python, NumPy and PyTorch seeds, uses explicit seeded NumPy generators for synthetic telemetry, uses deterministic fault transformations, enables PyTorch deterministic algorithms, fixes intra-op CPU execution to one thread and requests one inter-op thread before benchmark work begins.

These controls improve repeatability but do not promise bit-for-bit equality across operating systems, CPU libraries, PyTorch builds or hardware.

Source changes can alter the scientific results even when the seeds are unchanged. In particular, the corrected empty-memory sLSTM initialisation and the separation of adaptation and guard timelines require a fresh benchmark run. Compare source hashes and effective settings before treating two result files as repetitions of the same experiment.

## Checking adaptation decisions

For each `scenario=adaptation` row in `metrics.csv`, check

```text
guard_loss_after <= guard_loss_before + tolerance * abs(guard_loss_before)
```

The tolerance is recorded in `run_environment.json` under `benchmark.adaptation_tolerance`. A rejected row retains the candidate's guard loss for inspection, while the model parameters and buffers have been restored. The raw-timeline split is recorded as `benchmark.adaptation_split`; windowing happens separately in each segment. This avoids shared timesteps while retaining the temporal dependence and clean-target assumptions described in [the method](method.md).

## Timing

The recorded latency metric is `window_inference_latency_ms`. It is the mean host-CPU wall-clock time for one complete forward pass over one input window. The full benchmark uses 24 samples per window. It is hardware-dependent, not per-timestep latency, not spacecraft or robot real-time timing, and not WCET.

## Updating checked-in results

The summary reports sample standard deviations across distinct seeds and records the seed count. A one-seed smoke run uses zero as an output convention for standard deviation; it does not estimate uncertainty. Incomplete or duplicate scenario rows are rejected.

The benchmark prepares its outputs in a temporary directory. Computation or serialisation failure leaves prior evidence files intact; final replacement occurs one file at a time.

Do not edit result numbers manually. Re-run the benchmark and regenerate `metrics.csv`, `summary.json`, `fault_detection_f1.png` and `run_environment.json` together.
