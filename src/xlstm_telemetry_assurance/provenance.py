from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

RUN_ENVIRONMENT_SCHEMA_VERSION = 1


def configure_deterministic_execution() -> dict[str, Any]:
    """Configure deterministic CPU execution where PyTorch exposes controls."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    # Inter-op threads can only be changed before parallel work starts. A fresh
    # benchmark process should accept this; imported-library contexts may not.
    interop_requested = 1
    interop_configured = torch.get_num_interop_threads() == interop_requested
    if not interop_configured:
        try:
            torch.set_num_interop_threads(interop_requested)
            interop_configured = True
        except RuntimeError:
            interop_configured = False

    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "requested_num_interop_threads": interop_requested,
        "interop_threads_configured": interop_configured,
        "cuda_used": False,
    }


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _discover_git_root() -> Path | None:
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for candidate in candidates:
        root = _run_git(candidate, "rev-parse", "--show-toplevel")
        if root:
            return Path(root)
    return None


def _git_state(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root if repo_root is not None else _discover_git_root()
    if root is not None:
        sha = _run_git(root, "rev-parse", "HEAD")
        status = _run_git(root, "status", "--porcelain")
        if sha is not None:
            return {
                "available": True,
                "commit_sha": sha,
                "dirty": None if status is None else bool(status),
            }

    return {
        "available": False,
        "commit_sha": os.environ.get("GITHUB_SHA"),
        "dirty": None,
    }


def _cpu_model() -> str:
    processor = platform.processor().strip()
    if processor:
        return processor

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass

    return platform.machine() or "unknown"


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def collect_run_environment(
    benchmark_config: dict[str, Any],
    deterministic_settings: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Collect stable, machine-readable provenance for one benchmark run."""
    payload: dict[str, Any] = {
        "schema_version": RUN_ENVIRONMENT_SCHEMA_VERSION,
        "git": _git_state(repo_root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": _cpu_model(),
            "logical_cores": os.cpu_count(),
        },
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "benchmark": benchmark_config,
        "determinism": deterministic_settings,
    }
    payload["environment_fingerprint_sha256"] = _fingerprint(payload)
    return payload


def write_run_environment(
    path: Path,
    benchmark_config: dict[str, Any],
    deterministic_settings: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    payload = collect_run_environment(
        benchmark_config,
        deterministic_settings,
        repo_root=repo_root,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
