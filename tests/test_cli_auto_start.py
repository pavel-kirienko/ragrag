"""Test that the CLI auto-spawns the daemon when none is running.

We don't actually exec the real ``ragrag.daemon`` — that would spin up
ColQwen3 and load weights. We patch the ``DaemonClient`` methods that touch
subprocess and the socket to verify the spawn-then-call sequence happens.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ragrag.cli import main as cli_main
from ragrag.config import Settings


class _StubResponse:
    def __init__(self) -> None:
        self.query = "stub"
        self.status = "complete"
        self.results = []
        from ragrag.models import IndexingStats, TimingInfo

        self.indexed_now = IndexingStats()
        self.timing_ms = TimingInfo()


def test_cli_calls_ensure_daemon_then_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Happy path: env var unset, daemon path succeeds, no in-process fallback."""
    monkeypatch.delenv("RAGRAG_NO_DAEMON", raising=False)
    monkeypatch.setattr(sys, "argv", ["ragrag", "test query"])

    settings = Settings(
        index_path=str(tmp_path / ".ragrag"),
        top_k=5,
        daemon_autostart=True,
    )
    (tmp_path / ".ragrag").mkdir()

    fake_response = {
        "query": "test query",
        "status": "complete",
        "indexed_now": {"files_added": 0, "files_updated": 0, "files_skipped_unchanged": 0},
        "results": [],
        "timing_ms": {"total_ms": 12.5},
    }

    with (
        patch("ragrag.daemon.client.DaemonClient.ensure_daemon", return_value=None) as mock_ensure,
        patch("ragrag.daemon.client.DaemonClient.search", return_value=fake_response) as mock_search,
        patch("ragrag.cli.find_index_root", return_value=(str(tmp_path), settings)),
        patch("ragrag.cli.get_settings", return_value=settings),
        # In-process imports MUST NOT execute on the happy path
        patch("ragrag.embedding.colqwen_embedder.ColQwenEmbedder") as mock_embedder_cls,
    ):
        rc = cli_main()

    assert rc == 0
    mock_ensure.assert_called_once()
    mock_search.assert_called_once()
    mock_embedder_cls.assert_not_called()
    out = capsys.readouterr().out
    assert '"status": "complete"' in out


def test_cli_uses_daemon_status_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(sys, "argv", ["ragrag", "status"])
    settings = Settings(index_path=str(tmp_path / ".ragrag"))
    (tmp_path / ".ragrag").mkdir()
    # Touch the socket file so the CLI thinks a daemon exists
    (tmp_path / ".ragrag" / "daemon.sock").write_bytes(b"")

    fake_status = {"version": "0.1.0", "uptime_s": 1.0, "device_mode": "cpu"}
    with (
        patch("ragrag.cli.find_index_root", return_value=(str(tmp_path), settings)),
        patch("ragrag.daemon.client.DaemonClient.status", return_value=fake_status),
    ):
        rc = cli_main()

    assert rc == 0
    out = capsys.readouterr().out
    assert '"version"' in out
    assert '"device_mode"' in out


def test_cli_status_when_no_daemon_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(sys, "argv", ["ragrag", "status"])
    settings = Settings(index_path=str(tmp_path / ".ragrag"))
    (tmp_path / ".ragrag").mkdir()
    # No daemon.sock file present

    with patch("ragrag.cli.find_index_root", return_value=(str(tmp_path), settings)):
        rc = cli_main()

    assert rc == 1
    err = capsys.readouterr().err
    assert "no daemon running" in err.lower()
