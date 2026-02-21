"""Nox automation for ragrag tests and coverage."""
from __future__ import annotations

import os

import nox

nox.options.sessions = ["unit"]


@nox.session(python="3.10")
def unit(session: nox.Session) -> None:
    """Run unit tests with coverage (no model required)."""
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


@nox.session(python="3.10")
def e2e(session: nox.Session) -> None:
    """Run end-to-end tests with full pipeline (requires HuggingFace model)."""
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


@nox.session(python="3.10")
def coverage(session: nox.Session) -> None:
    """Combine coverage data and report (run after unit and e2e)."""
    session.install("coverage[toml]")
    session.run("coverage", "combine", ".coverage.unit", ".coverage.e2e", success_codes=[0, 1])
    session.run("coverage", "report", "--fail-under=80")
    session.run("coverage", "html")
