"""Unit tests for the daemon's RPC dispatcher and engine cache.

These don't spin up a real daemon process — they exercise the in-memory
classes directly. Coverage focus: error paths, parameter validation,
status snapshot construction, engine cache behaviour, and idle/timeout
plumbing that the lifecycle integration tests can't hit cheaply.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragrag.daemon.rpc import (
    ERROR_INVALID_PARAMS,
    ERROR_METHOD_NOT_FOUND,
    JsonRpcError,
    Request,
)
from ragrag.daemon.server import (
    DaemonState,
    Dispatcher,
    EngineCache,
    RecentQuery,
    _detect_device_mode,
    _read_one_line,
)


def _make_dispatcher(engines: dict | None = None) -> Dispatcher:
    state = DaemonState(started_at=time.time(), device_mode="cpu", last_request_at=time.time())
    cache = EngineCache(settings_factory=lambda p: MagicMock())
    if engines:
        cache._engines.update(engines)  # type: ignore[attr-defined]
    return Dispatcher(cache, state, threading.Event())


def test_dispatch_unknown_method_returns_error_response() -> None:
    disp = _make_dispatcher()
    req = Request(method="totally_made_up", params={}, id="x")
    resp = disp.dispatch(req)
    assert resp.error is not None
    assert resp.error["code"] == ERROR_METHOD_NOT_FOUND


def test_dispatch_handler_exception_wraps_into_internal_error() -> None:
    disp = _make_dispatcher()

    def boom(_params):
        raise RuntimeError("kaboom")

    disp.methods["explode"] = boom
    resp = disp.dispatch(Request(method="explode", params={}, id="x"))
    assert resp.error is not None
    assert resp.error["code"] != ERROR_METHOD_NOT_FOUND
    assert "kaboom" in resp.error["message"]


def test_status_snapshot_keys() -> None:
    disp = _make_dispatcher()
    snap = disp._status({})
    assert set(snap) >= {
        "version",
        "protocol_version",
        "device_mode",
        "uptime_s",
        "idle_s",
        "models_loaded",
        "indexing",
        "recent_queries",
    }
    assert snap["device_mode"] == "cpu"
    assert snap["models_loaded"] == []


def test_search_requires_index_path_param() -> None:
    disp = _make_dispatcher()
    with pytest.raises(JsonRpcError) as exc_info:
        disp._search({"query": "hi"})
    assert exc_info.value.code == ERROR_INVALID_PARAMS


def test_search_runs_against_cached_engine() -> None:
    fake_engine = MagicMock()
    fake_engine.search.return_value = MagicMock(
        results=[MagicMock(path="/foo")],
        status="complete",
    )
    disp = _make_dispatcher(engines={"/idx": fake_engine})
    result = disp._search({"index_path": "/idx", "query": "abc"})
    assert result.status == "complete"
    fake_engine.search.assert_called_once()
    # The dispatcher must record the query in recent_queries
    assert len(disp.state.recent_queries) == 1
    assert disp.state.recent_queries[0].query == "abc"


def test_index_dispatch_calls_ingest_manager() -> None:
    fake_engine = MagicMock()
    fake_engine.ingest_manager.ingest_paths.return_value = (
        MagicMock(model_dump=lambda: {"files_added": 1}),
        [],
        ["/idx/file.txt"],
    )
    disp = _make_dispatcher(engines={"/idx": fake_engine})
    result = disp._index({"index_path": "/idx", "paths": ["/idx"]})
    assert "stats" in result
    assert "skipped" in result


def test_shutdown_sets_event() -> None:
    disp = _make_dispatcher()
    assert not disp.shutdown_event.is_set()
    result = disp._shutdown({})
    assert result == {"ack": True}
    assert disp.shutdown_event.is_set()


def test_reload_config_empties_cache() -> None:
    disp = _make_dispatcher(engines={"/a": MagicMock(), "/b": MagicMock()})
    assert len(disp.engine_cache._engines) == 2
    result = disp._reload_config({})
    assert result == {"reloaded": True}
    assert disp.engine_cache._engines == {}


def test_engine_cache_returns_cached_engine() -> None:
    cache = EngineCache(settings_factory=lambda p: MagicMock())
    sentinel = object()
    cache._engines["/idx"] = sentinel  # type: ignore[attr-defined]
    assert cache.get("/idx") is sentinel


def test_recent_query_dataclass() -> None:
    rq = RecentQuery(query="x", wall_ms=42, status="complete", top1_path=None, timestamp=time.time())
    assert rq.query == "x"
    assert rq.wall_ms == 42


def test_read_one_line_returns_first_line(tmp_path: Path) -> None:
    """_read_one_line is socket-driven; we exercise it via socketpair to keep
    the test in-process."""
    import socket

    a, b = socket.socketpair()
    try:
        b.sendall(b"hello world\nignored")
        line = _read_one_line(a)
        assert line == b"hello world"
    finally:
        a.close()
        b.close()


def test_read_one_line_caps_at_max() -> None:
    import socket

    a, b = socket.socketpair()
    try:
        b.sendall(b"x" * 50 + b"\n")
        line = _read_one_line(a, max_bytes=50)
        # Capped before newline → returns whatever it had at the cap
        assert len(line) <= 50
    finally:
        a.close()
        b.close()


def test_read_one_line_empty_on_close() -> None:
    import socket

    a, b = socket.socketpair()
    b.close()
    try:
        line = _read_one_line(a)
        assert line == b""
    finally:
        a.close()


def test_detect_device_mode_known_value() -> None:
    assert _detect_device_mode() in {"cuda", "cpu", "mps"}
