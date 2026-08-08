# Reproducibility

## Environment

The benchmark supports Python 3.10+ and CPU execution. The primary dependencies are PyTorch, NumPy and Matplotlib.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
python -m pytest
```

## Full benchmark

```bash
python -m xlstm_telemetry_assurance.benchmark --output results/benchmark
```

The default benchmark uses seeds `11`, `29` and `47` and writes:

- `metrics.csv`: per-seed, per-domain, per-model and per-fault measurements;
- `summary.json`: compact aggregate statistics;
- `fault_detection_f1.png`: plot generated from the measured CSV results.

## Smoke benchmark

```bash
python -m xlstm_telemetry_assurance.benchmark --smoke --output results/smoke
```

The smoke mode uses a shorter stream and one seed. It is intended to validate the pipeline, not reproduce the checked-in headline numbers.

## Determinism

Each run sets Python, NumPy and PyTorch seeds. Synthetic telemetry and fault injection use explicit seeded NumPy generators. CPU execution is used for the checked-in benchmark to avoid GPU-kernel nondeterminism.

Latency is inherently hardware- and load-dependent and should not be expected to reproduce exactly across machines.

## Updating checked-in results

Do not edit result numbers manually. Re-run the benchmark and regenerate `summary.json`, `metrics.csv` and the figure together.
