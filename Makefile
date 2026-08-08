.PHONY: install test smoke benchmark clean

install:
	python -m pip install -e .[dev]

test:
	python -m pytest

smoke:
	python -m xlstm_telemetry_assurance.benchmark --smoke --output results/smoke

benchmark:
	python -m xlstm_telemetry_assurance.benchmark --output results/benchmark

clean:
	rm -rf .pytest_cache results/tmp
