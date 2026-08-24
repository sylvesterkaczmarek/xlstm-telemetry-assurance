# Reproducibility

## Environment

`pyproject.toml` is the authoritative definition of supported runtime and development dependencies. CI exercises Python 3.10, 3.11 and 3.12.

For normal development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
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

- `metrics.csv`: per-seed, per-domain, per-model and per-scenario measurements;
- `summary.json`: aggregate forecasting, Gaussian NLL, packet-loss, value-only, mixed and per-fault statistics plus timing semantics;
- `fault_detection_f1.png`: packet-loss, value-only and mixed F1 comparison;
- `run_environment.json`: Git state when available, Python/platform/CPU information, package versions, benchmark configuration, deterministic settings and a SHA-256 fingerprint.

Git provenance is collected when the run is inside a Git checkout. Outside a checkout, the file is still written and may use `GITHUB_SHA` when supplied.

## Smoke benchmark

```bash
python -m xlstm_telemetry_assurance.benchmark --smoke --output results/smoke
```

Smoke mode uses a shorter stream and one seed. It validates the pipeline rather than reproducing the checked-in headline numbers. CI verifies that smoke mode also produces `run_environment.json`.

## Determinism

Each benchmark run fixes Python, NumPy and PyTorch seeds, uses explicit seeded NumPy generators for synthetic telemetry, uses deterministic fault transformations, enables PyTorch deterministic algorithms, fixes intra-op CPU execution to one thread and requests one inter-op thread before benchmark work begins.

These controls improve repeatability but do not promise bit-for-bit equality across operating systems, CPU libraries, PyTorch builds or hardware.

## Timing

The recorded latency metric is `window_inference_latency_ms`. It is the mean host-CPU wall-clock time for one complete forward pass over one input window. The full benchmark uses 24 samples per window. It is hardware-dependent, not per-timestep latency, not spacecraft or robot real-time timing, and not WCET.

## Updating checked-in results

Do not edit result numbers manually. Re-run the benchmark and regenerate `metrics.csv`, `summary.json`, `fault_detection_f1.png` and `run_environment.json` together.
