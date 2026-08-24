import json
import re

import torch

from xlstm_telemetry_assurance.benchmark import _benchmark_config
from xlstm_telemetry_assurance.provenance import (
    collect_run_environment,
    configure_deterministic_execution,
    write_run_environment,
)


def test_benchmark_config_records_effective_scientific_parameters():
    full = _benchmark_config(smoke=False)
    smoke = _benchmark_config(smoke=True)
    assert full["seeds"] == [11, 29, 47]
    assert full["sequence_length"] == 24
    assert full["epochs"] == 10
    assert full["train_length"] == 1050
    assert full["eval_length"] == 700
    assert full["hidden_size"] == 32
    assert full["timing"]["metric"] == "window_inference_latency_ms"
    assert smoke["seeds"] == [11]
    assert smoke["sequence_length"] == 16
    assert smoke["epochs"] == 2


def test_deterministic_cpu_configuration_is_enabled():
    settings = configure_deterministic_execution()
    assert settings["torch_deterministic_algorithms"] is True
    assert torch.are_deterministic_algorithms_enabled()
    assert settings["torch_num_threads"] == 1
    assert torch.get_num_threads() == 1
    assert settings["cuda_used"] is False


def test_run_environment_is_structured_and_fingerprint_is_stable(tmp_path):
    settings = configure_deterministic_execution()
    config = _benchmark_config(smoke=True)
    first = collect_run_environment(config, settings, repo_root=tmp_path)
    second = collect_run_environment(config, settings, repo_root=tmp_path)

    assert first["schema_version"] == 1
    assert first["benchmark"] == config
    assert first["packages"]["torch"]
    assert first["packages"]["numpy"]
    assert first["packages"]["matplotlib"]
    assert "commit_sha" in first["git"]
    assert "dirty" in first["git"]
    assert first["cpu"]["logical_cores"] is None or first["cpu"]["logical_cores"] >= 1
    assert re.fullmatch(r"[0-9a-f]{64}", first["environment_fingerprint_sha256"])
    assert first["environment_fingerprint_sha256"] == second["environment_fingerprint_sha256"]

    path = tmp_path / "run_environment.json"
    written = write_run_environment(path, config, settings, repo_root=tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == written
