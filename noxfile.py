"""Nox automation for ragrag tests and coverage."""
from __future__ import annotations

import importlib
import os
import subprocess
from typing import Any, cast

nox = cast(Any, importlib.import_module("nox"))

nox.options.sessions = ["unit"]


def _torch_index_url() -> str | None:
    """Pick a torch wheel index suited to the host.

    Honors TORCH_INDEX_URL first. Otherwise shells out to nvidia-smi and, if
    the driver is too old for default PyPI cu12x wheels, falls back to the
    cu118 index. Returns None when the default PyPI wheels will work.
    """
    override = os.environ.get("TORCH_INDEX_URL")
    if override:
        return override or None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    line = (result.stdout or "").strip().splitlines()
    if not line:
        return None
    try:
        major = int(line[0].split(".")[0])
    except (ValueError, IndexError):
        return None

    # CUDA 12.x wheels on PyPI need driver >= 525. Older drivers work with cu118.
    return "https://download.pytorch.org/whl/cu118" if major < 525 else None


def _preinstall_torch(session: Any) -> None:
    """If the host needs non-default torch wheels, install them before the project."""
    idx = _torch_index_url()
    if idx:
        session.log(f"Preinstalling torch/torchvision from {idx}")
        session.install("torch", "torchvision", "--index-url", idx)


@nox.session
def unit(session: Any) -> None:
    """Run unit tests with coverage (no model required)."""
    _preinstall_torch(session)
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "tests/",
        "--ignore=tests/test_e2e.py",
        "-x",
        "-v",
        "--cov=src",
        "--cov-branch",
        "--cov-report=term-missing",
        env={"COVERAGE_FILE": ".coverage.unit"},
    )


@nox.session
def e2e(session: Any) -> None:
    """Run end-to-end tests with full pipeline (requires HuggingFace model)."""
    _preinstall_torch(session)
    if os.environ.get("CI") == "true":
        session.install("--no-deps", ".")
        session.install("pydantic>=2.0.0")
    else:
        session.install(".")
    session.install("pytest", "pytest-timeout", "pytest-cov")
    session.run(
        "pytest",
        "tests/test_e2e.py",
        "-x",
        "-v",
        "--timeout=600",
        "--cov=src",
        "--cov-branch",
        "--cov-fail-under=0",
        env={"COVERAGE_FILE": ".coverage.e2e", "CI": os.environ.get("CI", "false")},
    )


@nox.session
def coverage(session: Any) -> None:
    """Combine coverage data and report (run after unit and e2e)."""
    session.install("coverage[toml]")
    session.run("coverage", "combine", ".coverage.unit", ".coverage.e2e", success_codes=[0, 1])
    session.run("coverage", "report", "--fail-under=80")
    session.run("coverage", "html")
