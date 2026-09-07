import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest
import torch

from xlstm_telemetry_assurance import provenance
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


def _init_repository(root: Path, filename: str) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    tracked = root / filename
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("# Initial source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", filename], check=True)
    subprocess.run(
        [
            "git", "-C", str(root), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "-qm", "Initial source",
        ],
        check=True,
    )
    return tracked


def test_git_discovery_uses_executed_package_instead_of_working_directory(tmp_path, monkeypatch):
    package_root = tmp_path / "package"
    module = _init_repository(package_root, "src/example/provenance.py")
    unrelated_root = tmp_path / "unrelated"
    _init_repository(unrelated_root, "README.md")
    monkeypatch.setattr(provenance, "__file__", str(module))
    monkeypatch.chdir(unrelated_root)

    assert provenance._discover_git_root() == package_root
    expected = subprocess.check_output(
        ["git", "-C", str(package_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert provenance._git_state() == {"available": True, "commit_sha": expected, "dirty": False}


def test_installed_package_does_not_claim_enclosing_repository_commit(tmp_path, monkeypatch):
    unrelated_root = tmp_path / "unrelated"
    _init_repository(unrelated_root, "README.md")
    module = unrelated_root / ".venv/lib/site-packages/example/provenance.py"
    module.parent.mkdir(parents=True)
    module.write_text("# Installed package\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "__file__", str(module))
    monkeypatch.chdir(unrelated_root)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)

    assert provenance._discover_git_root() is None
    assert provenance._git_state() == {"available": False, "commit_sha": None, "dirty": None}


def test_environment_hashes_every_package_source_file(tmp_path):
    environment = collect_run_environment({}, {}, repo_root=tmp_path)
    package = Path(provenance.__file__).resolve().parent
    expected = {
        path.relative_to(package.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package.rglob("*.py")
    }
    assert environment["source_sha256"] == expected
    assert "xlstm_telemetry_assurance/provenance.py" in expected
    assert "xlstm_telemetry_assurance/benchmark.py" in expected


def test_source_changes_are_detected_even_without_git_metadata(tmp_path, monkeypatch):
    package = tmp_path / "example"
    package.mkdir()
    module = package / "provenance.py"
    module.write_text("# Version one\n", encoding="utf-8")
    nested_module = package / "nested/model.py"
    nested_module.parent.mkdir()
    nested_module.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "__file__", str(module))

    first = collect_run_environment({}, {}, repo_root=tmp_path)
    nested_module.write_text("VALUE = 2\n", encoding="utf-8")
    second = collect_run_environment({}, {}, repo_root=tmp_path)

    assert first["git"] == second["git"]
    assert first["source_sha256"]["example/provenance.py"] == second["source_sha256"]["example/provenance.py"]
    assert first["source_sha256"]["example/nested/model.py"] != second["source_sha256"]["example/nested/model.py"]
    assert first["environment_fingerprint_sha256"] != second["environment_fingerprint_sha256"]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["benchmark", "determinism"])
def test_invalid_metadata_cannot_overwrite_existing_evidence(tmp_path, invalid, field):
    path = tmp_path / "run_environment.json"
    path.write_text('{"existing": true}\n', encoding="utf-8")
    config = {"value": invalid} if field == "benchmark" else {}
    settings = {"value": invalid} if field == "determinism" else {}

    with pytest.raises(ValueError, match="JSON compliant"):
        write_run_environment(path, config, settings, repo_root=tmp_path)
    assert path.read_text(encoding="utf-8") == '{"existing": true}\n'
