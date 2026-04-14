"""End-to-end lifecycle tests for the daemon.

These tests start a real daemon process in the same Python interpreter (no
detach, no subprocess), with an injected stub engine cache so they don't load
ColQwen3. They exercise: bind → status RPC → search RPC → shutdown → cleanup.
"""
from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import pytest

from ragrag.daemon.client import DaemonClient, DaemonError
from ragrag.daemon.server import DaemonServer
from ragrag.models import IndexingStats, SearchResponse, TimingInfo


class _FakeEngine:
    """Minimal SearchEngine stand-in returning a fixed response."""

    def search(self, request):
        return SearchResponse(
            query=request.query,
            status="complete",
            indexed_now=IndexingStats(),
            results=[],
            timing_ms=TimingInfo(total_ms=12.5),
        )


def _start_daemon_in_thread(tmp_path: Path) -> tuple[DaemonServer, threading.Thread]:
    ragrag_dir = tmp_path / ".ragrag"
    ragrag_dir.mkdir()
    sock_path = ragrag_dir / "daemon.sock"
    pid_path = ragrag_dir / "daemon.pid"
    server = DaemonServer(
        index_path=str(tmp_path),
        socket_path=sock_path,
        pid_path=pid_path,
        idle_timeout_s=86400,
    )
    # Pre-populate the engine cache so we never load real models.
    server.engine_cache._engines[str(tmp_path)] = _FakeEngine()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait until a real connect() succeeds — file existence alone is not enough,
    # the server has to be in the accept loop with listen() called.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if sock_path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.5)
                    probe.connect(str(sock_path))
                break
            except (OSError, socket.timeout):
                pass
        time.sleep(0.05)
    else:
        raise AssertionError("daemon did not become accept-ready in 5 s")
    return server, thread


def test_status_roundtrip(tmp_path: Path) -> None:
    server, _thread = _start_daemon_in_thread(tmp_path)
    try:
        client = DaemonClient(str(tmp_path), ensure_timeout_s=2.0)
        snapshot = client.status()
        assert "version" in snapshot
        assert snapshot["protocol_version"] == 1
        assert snapshot["device_mode"] in {"cuda", "cpu", "mps"}
        assert "uptime_s" in snapshot
    finally:
        server.shutdown_event.set()


def test_search_roundtrip(tmp_path: Path) -> None:
    server, _thread = _start_daemon_in_thread(tmp_path)
    try:
        client = DaemonClient(str(tmp_path))
        result = client.search("hello", paths=[str(tmp_path)], top_k=3)
        assert result["query"] == "hello"
        assert result["status"] == "complete"
        assert result["timing_ms"]["total_ms"] == 12.5
    finally:
        server.shutdown_event.set()


def test_shutdown_cleans_socket(tmp_path: Path) -> None:
    server, thread = _start_daemon_in_thread(tmp_path)
    sock_path = tmp_path / ".ragrag" / "daemon.sock"
    pid_path = tmp_path / ".ragrag" / "daemon.pid"
    client = DaemonClient(str(tmp_path))
    client.shutdown(wait_s=2.0)
    thread.join(timeout=5.0)
    assert not sock_path.exists()
    assert not pid_path.exists()


def test_unknown_method_returns_error(tmp_path: Path) -> None:
    server, _thread = _start_daemon_in_thread(tmp_path)
    try:
        client = DaemonClient(str(tmp_path))
        with pytest.raises(DaemonError) as exc_info:
            client.call("nonexistent", {})
        assert "unknown method" in str(exc_info.value).lower()
    finally:
        server.shutdown_event.set()


def test_garbage_request_returns_parse_error(tmp_path: Path) -> None:
    server, _thread = _start_daemon_in_thread(tmp_path)
    sock_path = tmp_path / ".ragrag" / "daemon.sock"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(sock_path))
            s.sendall(b"this is not json\n")
            buf = b""
            while True:
                chunk = s.recv(8192)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in chunk:
                    break
        assert b"\"error\"" in buf
    finally:
        server.shutdown_event.set()
