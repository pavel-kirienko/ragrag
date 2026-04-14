"""Smoke tests for ``scripts/diff_bench.py``.

Exercises the pairwise and matrix modes against synthetic reports
so PR authors can trust the headline numbers produced in commit
messages.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diff_bench.py"


def _make_report(tmp_path: Path, name: str, **metrics) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"summary": metrics}))
    return path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_pairwise_diff_prints_metric_deltas(tmp_path: Path) -> None:
    a = _make_report(tmp_path, "baseline", p_at_1=0.25, p_at_5=0.67, mrr=0.4, semantic_at_5=1.0)
    b = _make_report(tmp_path, "phaseB", p_at_1=0.75, p_at_5=0.92, mrr=0.8, semantic_at_5=1.0)
    result = _run(str(a), str(b))
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "P@1" in out
    assert "0.250" in out
    assert "0.750" in out
    assert "0.500" in out  # delta


def test_pairwise_diff_flags_sem5_regression(tmp_path: Path) -> None:
    a = _make_report(tmp_path, "baseline", semantic_at_5=1.0)
    b = _make_report(tmp_path, "phaseX", semantic_at_5=0.5)
    result = _run(str(a), str(b))
    # Script returns 1 when semantic_at_5 regresses.
    assert result.returncode == 1
    assert "semantic_at_5 regressed" in result.stderr


def test_matrix_mode_prints_metric_by_phase(tmp_path: Path) -> None:
    a = _make_report(tmp_path, "baseline", p_at_1=0.25, mrr=0.40)
    b = _make_report(tmp_path, "phaseA", p_at_1=0.25, mrr=0.42)
    c = _make_report(tmp_path, "phaseB", p_at_1=0.75, mrr=0.80)
    result = _run("--matrix", str(a), str(b), str(c))
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "baseline" in out
    assert "phaseA" in out
    assert "phaseB" in out
    # Every metric row shows up
    for label in ("P@1", "P@5", "P@10", "MRR"):
        assert label in out


def test_matrix_markdown_mode(tmp_path: Path) -> None:
    a = _make_report(tmp_path, "baseline", p_at_1=0.25)
    b = _make_report(tmp_path, "phaseB", p_at_1=0.75)
    result = _run("--matrix", str(a), str(b), "--markdown")
    assert result.returncode == 0
    assert result.stdout.strip().startswith("| Metric |")
    assert "| P@1 |" in result.stdout
