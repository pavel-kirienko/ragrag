"""Integration tests for the Phase E HTTP dashboard.

The dashboard lives in ``ragrag/daemon/http_status.py`` and is served
on a second thread alongside the JSON-RPC Unix socket. These tests
boot the daemon the same way ``test_daemon_lifecycle`` does — one
process, stubbed engine cache — then hit the HTTP endpoints with the
stdlib ``urllib`` so we do not add a test-time dependency.
"""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from ragrag.daemon.server import DaemonServer
from ragrag.index.page_cache import PageImageCache
from ragrag.models import IndexingStats, SearchResponse, TimingInfo


class _FakeEngine:
    """Minimal SearchEngine stand-in with an ingest_manager + page_cache."""

    def __init__(self, page_cache_root: Path) -> None:
        self.ingest_manager = _FakeIngest(page_cache_root)

    def search(self, request):
        return SearchResponse(
            query=request.query,
            status="complete",
            indexed_now=IndexingStats(),
            results=[],
            timing_ms=TimingInfo(total_ms=0.1),
        )


class _FakeIngest:
    def __init__(self, page_cache_root: Path) -> None:
        self.page_cache = PageImageCache(str(page_cache_root))


def _start_daemon(tmp_path: Path) -> DaemonServer:
    ragrag_dir = tmp_path / ".ragrag"
    ragrag_dir.mkdir()
    sock_path = ragrag_dir / "daemon.sock"
    pid_path = ragrag_dir / "daemon.pid"
    server = DaemonServer(
        index_path=str(tmp_path),
        socket_path=sock_path,
        pid_path=pid_path,
        idle_timeout_s=86400,
        http_port=0,  # bind to a random port
    )
    server.engine_cache._engines[str(tmp_path)] = _FakeEngine(  # type: ignore[attr-defined]
        ragrag_dir / "page_cache"
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait until the daemon socket is accept-ready, matching
    # test_daemon_lifecycle's probe loop.
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
    return server


def _dashboard_base(server: DaemonServer) -> str:
    assert server.http_port, "status server did not bind a port"
    return f"http://127.0.0.1:{server.http_port}"


def test_dashboard_index_returns_html(tmp_path: Path) -> None:
    server = _start_daemon(tmp_path)
    try:
        with urllib.request.urlopen(_dashboard_base(server) + "/", timeout=2.0) as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
        assert "ragrag dashboard" in body
        assert "/status" in body  # the poller URL is inlined in JS
    finally:
        server.shutdown_event.set()


def test_dashboard_status_payload_has_expected_keys(tmp_path: Path) -> None:
    server = _start_daemon(tmp_path)
    try:
        with urllib.request.urlopen(_dashboard_base(server) + "/status", timeout=2.0) as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
        assert "version" in payload
        assert "uptime_s" in payload
        assert "device_mode" in payload
        assert "resources" in payload
        # recent_queries should exist (may be empty)
        assert "recent_queries" in payload
    finally:
        server.shutdown_event.set()


def test_dashboard_unknown_path_returns_404(tmp_path: Path) -> None:
    server = _start_daemon(tmp_path)
    try:
        req = urllib.request.Request(_dashboard_base(server) + "/does-not-exist")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=2.0)
        assert exc_info.value.code == 404
    finally:
        server.shutdown_event.set()


def test_dashboard_shutdown_requires_confirmation_header(tmp_path: Path) -> None:
    server = _start_daemon(tmp_path)
    try:
        req = urllib.request.Request(
            _dashboard_base(server) + "/shutdown",
            method="POST",
            data=b"",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=2.0)
        assert exc_info.value.code == 400
    finally:
        server.shutdown_event.set()


def test_dashboard_pages_endpoint_serves_cached_webp(tmp_path: Path) -> None:
    server = _start_daemon(tmp_path)
    try:
        # Seed the fake engine's page cache with one WebP.
        engine = server.engine_cache._engines[str(tmp_path)]  # type: ignore[attr-defined]
        cache: PageImageCache = engine.ingest_manager.page_cache
        from PIL import Image as _Image

        img = _Image.new("RGB", (32, 32), (200, 100, 50))
        cache.put("deadbeef" * 8, 3, img)

        url = _dashboard_base(server) + "/pages/" + ("deadbeef" * 8) + "/3.webp"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            assert resp.status == 200
            body = resp.read()
        assert body.startswith(b"RIFF") and b"WEBP" in body[:16]
    finally:
        server.shutdown_event.set()
