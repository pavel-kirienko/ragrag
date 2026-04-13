"""End-to-end tests for ragrag CLI.

These tests exercise the installed ragrag command via subprocess.
The search tests (test_e2e_text_search, test_e2e_markdown_output) require
the HuggingFace model to be available (cached or downloaded).
They are slow (~60-300s on CPU) and are run in the 'e2e' nox session.

The help/version tests are fast and do not require the model.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


SKIP_MODEL_E2E_ON_CI = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skipped on CI due to model and runner resource limits",
)


def _run_ragrag(*args: str, timeout: int = 30, cwd: str | None = None) -> "subprocess.CompletedProcess[str]":
    """Run ragrag CLI as subprocess and return CompletedProcess."""
    # Also try the installed 'ragrag' command if available
    try:
        result = subprocess.run(
            ["ragrag"] + list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result
    except FileNotFoundError:
        # Fall back to python -m ragrag.cli
        return subprocess.run(
            [sys.executable, "-m", "ragrag.cli"] + list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )


def test_e2e_help() -> None:
    """ragrag --help exits 0 and shows usage."""
    result = _run_ragrag("--help")
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    output = result.stdout + result.stderr
    assert "Search query string" in output or "query" in output.lower(), (
        f"Expected usage info in output, got: {output[:500]}"
    )


def test_e2e_version() -> None:
    """ragrag --version exits 0 and shows version number."""
    result = _run_ragrag("--version")
    assert result.returncode == 0, f"--version failed: {result.stderr}"
    output = result.stdout + result.stderr
    assert "0.1.0" in output, f"Expected version 0.1.0 in output, got: {output}"


@pytest.mark.timeout(600)
@SKIP_MODEL_E2E_ON_CI
def test_e2e_text_search() -> None:
    """Full pipeline: index a text file and search it. Requires HuggingFace model."""
    content = (
        "The STM32 microcontroller GPIO configuration requires setting the MODER register "
        "to configure each pin as input, output, alternate function, or analog mode. "
        "The BSRR register provides atomic set/reset operations for GPIO pins. "
    ) * 8  # ~500+ chars to ensure chunking

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a text file with known content
        (Path(tmpdir) / "gpio_notes.txt").write_text(content, encoding="utf-8")

        # Run ragrag search with --new to create index in tmpdir
        result = _run_ragrag(
            "GPIO configuration",
            str(tmpdir),
            "--new",
            "--json",
            timeout=600,
            cwd=tmpdir,
        )

        assert result.returncode == 0, (
            f"ragrag search failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout[:1000]}\n"
            f"stderr: {result.stderr[:1000]}"
        )

        # Parse JSON output
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            pytest.fail(f"stdout is not valid JSON: {e}\nstdout: {result.stdout[:500]}")

        assert response.get("status") == "complete", (
            f"Expected status=complete, got: {response.get('status')}\n"
            f"errors: {response.get('errors', [])}"
        )
        results = response.get("results", [])
        assert len(results) > 0, (
            f"Expected at least 1 result, got 0.\n"
            f"indexed_now: {response.get('indexed_now')}"
        )
        # At least one result should be from our file
        paths = [r.get("path", "") for r in results]
        assert any("gpio_notes.txt" in p for p in paths), (
            f"Expected gpio_notes.txt in results, got paths: {paths}"
        )


@pytest.mark.timeout(600)
@SKIP_MODEL_E2E_ON_CI
def test_e2e_markdown_output() -> None:
    """Full pipeline with --markdown output. Requires HuggingFace model."""
    content = (
        "The SPI peripheral on STM32 supports full-duplex synchronous serial communication. "
        "Configure SPI1 with master mode, software slave management, and appropriate baud rate. "
    ) * 8

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "spi_notes.txt").write_text(content, encoding="utf-8")

        result = _run_ragrag(
            "SPI configuration",
            str(tmpdir),
            "--new",
            "--markdown",
            timeout=600,
            cwd=tmpdir,
        )

        assert result.returncode == 0, (
            f"ragrag --markdown failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr[:500]}"
        )
        assert "# Search Results:" in result.stdout, (
            f"Expected markdown header in output, got: {result.stdout[:500]}"
        )
