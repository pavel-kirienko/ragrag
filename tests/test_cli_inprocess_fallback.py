"""CLI falls back to in-process when the daemon path fails."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ragrag.cli import main as cli_main
from ragrag.config import Settings
from ragrag.daemon.client import DaemonStartupError


class _StubResponse:
    """Minimal SearchResponse stand-in returned by the mock engine."""

    def __init__(self) -> None:
        self.query = "stub"
        self.status = "complete"
        self.results = []
        from ragrag.models import IndexingStats, TimingInfo

        self.indexed_now = IndexingStats()
        self.timing_ms = TimingInfo()

    def model_dump_json(self, indent: int = 2) -> str:
        return '{"status": "complete"}'


def test_cli_falls_back_to_inprocess_when_daemon_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """If the daemon path raises DaemonStartupError, the CLI runs in-process and still prints results."""
    # Allow the daemon path to be tried (conftest sets RAGRAG_NO_DAEMON globally).
    monkeypatch.delenv("RAGRAG_NO_DAEMON", raising=False)

    # The CLI sets up logging; isolate it from prior state.
    monkeypatch.setattr(sys, "argv", ["ragrag", "test query"])

    settings = Settings(
        index_path=str(tmp_path / ".ragrag"),
        top_k=5,
        daemon_autostart=True,
    )
    (tmp_path / ".ragrag").mkdir()

    response = _StubResponse()
    with (
        patch("ragrag.daemon.client.DaemonClient.ensure_daemon", side_effect=DaemonStartupError("nope")),
        patch("ragrag.cli.find_index_root", return_value=(str(tmp_path), settings)),
        patch("ragrag.cli.get_settings", return_value=settings),
        patch("ragrag.embedding.colqwen_embedder.ColQwenEmbedder") as mock_embedder_cls,
        patch("ragrag.index.qdrant_store.QdrantStore") as mock_store_cls,
        patch("ragrag.index.ingest_manager.IngestManager") as mock_ingest_cls,
        patch("ragrag.retrieval.search_engine.SearchEngine") as mock_engine_cls,
    ):
        mock_engine_cls.return_value.search.return_value = response
        rc = cli_main()

    assert rc == 0
    captured = capsys.readouterr()
    assert "complete" in captured.out
    # In-process path was reached → mocks were called
    assert mock_embedder_cls.called
    assert mock_engine_cls.called


def test_cli_skips_daemon_when_no_daemon_flag_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.delenv("RAGRAG_NO_DAEMON", raising=False)
    monkeypatch.setattr(sys, "argv", ["ragrag", "test query", "--no-daemon"])

    settings = Settings(
        index_path=str(tmp_path / ".ragrag"),
        top_k=5,
        daemon_autostart=True,
    )
    (tmp_path / ".ragrag").mkdir()

    response = _StubResponse()
    with (
        patch("ragrag.daemon.client.DaemonClient.ensure_daemon") as mock_ensure,
        patch("ragrag.cli.find_index_root", return_value=(str(tmp_path), settings)),
        patch("ragrag.cli.get_settings", return_value=settings),
        patch("ragrag.embedding.colqwen_embedder.ColQwenEmbedder") as mock_embedder_cls,
        patch("ragrag.index.qdrant_store.QdrantStore"),
        patch("ragrag.index.ingest_manager.IngestManager"),
        patch("ragrag.retrieval.search_engine.SearchEngine") as mock_engine_cls,
    ):
        mock_engine_cls.return_value.search.return_value = response
        rc = cli_main()

    assert rc == 0
    mock_ensure.assert_not_called()
    assert mock_embedder_cls.called


def test_cli_skips_daemon_when_env_var_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("RAGRAG_NO_DAEMON", "1")
    monkeypatch.setattr(sys, "argv", ["ragrag", "test query"])

    settings = Settings(
        index_path=str(tmp_path / ".ragrag"),
        top_k=5,
        daemon_autostart=True,
    )
    (tmp_path / ".ragrag").mkdir()

    response = _StubResponse()
    with (
        patch("ragrag.daemon.client.DaemonClient.ensure_daemon") as mock_ensure,
        patch("ragrag.cli.find_index_root", return_value=(str(tmp_path), settings)),
        patch("ragrag.cli.get_settings", return_value=settings),
        patch("ragrag.embedding.colqwen_embedder.ColQwenEmbedder"),
        patch("ragrag.index.qdrant_store.QdrantStore"),
        patch("ragrag.index.ingest_manager.IngestManager"),
        patch("ragrag.retrieval.search_engine.SearchEngine") as mock_engine_cls,
    ):
        mock_engine_cls.return_value.search.return_value = response
        rc = cli_main()

    assert rc == 0
    mock_ensure.assert_not_called()
