# Reproducibility

## Environment

The package metadata in `pyproject.toml` is the authoritative definition of supported runtime and development dependencies. CI exercises Python 3.10, 3.11 and 3.12.

For normal development, install from the supported dependency ranges:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
python -m pytest
```

For the exact Python 3.12 reference environment used by CI, install the direct pins and then install the package without resolving a second dependency set:

```bash
python -m pip install -r requirements-reference.txt
python -m pip install --no-build-isolation --no-deps -e .
```

`requirements-reference.txt` is a reproducibility snapshot, not the package dependency specification. Broad compatible ranges remain in `pyproject.toml`.

## Full benchmark

```bash
python -m xlstm_telemetry_assurance.benchmark --output results/benchmark
```

The default benchmark uses seeds `11`, `29` and `47` and writes:

- `metrics.csv`: per-seed, per-domain, per-model and per-fault measurements;
- `summary.json`: compact aggregate statistics;
- `fault_detection_f1.png`: plot generated from the measured CSV results;
- `run_environment.json`: Git state, Python/platform/CPU information, package versions, benchmark configuration, deterministic-execution settings and a SHA-256 environment/configuration fingerprint.

Git provenance is collected when the run is inside a Git checkout. Outside a checkout, the file is still written and the Git fields are marked unavailable rather than failing the benchmark.

## Smoke benchmark

```bash
python -m xlstm_telemetry_assurance.benchmark --smoke --output results/smoke
```

The smoke mode uses a shorter stream and one seed. It is intended to validate the pipeline, not reproduce the checked-in headline numbers. CI verifies that the smoke run also produces a valid `run_environment.json`.

## Determinism

Each benchmark run:

- fixes the existing Python, NumPy and PyTorch seeds;
- uses explicit seeded NumPy generators for synthetic telemetry and fault injection;
- enables PyTorch deterministic algorithms;
- fixes PyTorch intra-op execution to one thread and requests one inter-op thread before benchmark work begins;
- records the effective deterministic settings in `run_environment.json`;
- runs the checked-in benchmark on CPU rather than depending on GPU kernels.

These controls improve repeatability but are not a promise of bit-for-bit equality across different operating systems, CPU libraries, PyTorch builds or hardware. Latency is hardware- and load-dependent and should not be expected to reproduce exactly across machines.

## Updating checked-in results

Do not edit result numbers manually. Re-run the benchmark and regenerate `summary.json`, `metrics.csv`, the figure and `run_environment.json` together.
