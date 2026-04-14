"""Tests for DaemonClient.call against an in-process fake daemon thread.

These exercise the encode → connect → write → read → decode round-trip so
we don't have to spin up a real daemon process.
"""
from __future__ import annotations

import socket
import threading
from pathlib import Path
from typing import Any

import pytest

from ragrag.daemon.client import DaemonClient, DaemonError
from ragrag.daemon.rpc import (
    Request,
    Response,
    decode_request,
    encode_response,
    JsonRpcError,
    error_response,
    ERROR_INVALID_PARAMS,
)


class _FakeDaemon:
    """Tiny accept-loop that handles one request per connection."""

    def __init__(self, sock_path: Path, handler):
        self.sock_path = sock_path
        self.handler = handler
        self.thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.sock_path.exists():
            self.sock_path.unlink()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(self.sock_path))
        self._sock.listen(8)
        self._sock.settimeout(0.5)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self.sock_path.exists():
            self.sock_path.unlink()

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            try:
                buf = b""
                while b"\n" not in buf:
                    try:
                        chunk = conn.recv(8192)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                # Skip probes that close without sending anything.
                if not buf:
                    continue
                line = buf.split(b"\n", 1)[0]
                try:
                    req = decode_request(line)
                    result = self.handler(req)
                    resp = Response(id=req.id, result=result)
                except JsonRpcError as exc:
                    resp = error_response(None, exc)
                try:
                    conn.sendall(encode_response(resp))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass


def _wait_for_socket(sock_path: Path, timeout: float = 2.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sock_path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.5)
                    probe.connect(str(sock_path))
                return
            except OSError:
                pass
        time.sleep(0.02)
    raise AssertionError("fake daemon socket did not appear")


def test_call_search_round_trip(tmp_path: Path) -> None:
    sock = tmp_path / ".ragrag" / "daemon.sock"

    def handler(req: Request) -> Any:
        assert req.method == "search"
        assert req.params["query"] == "hello"
        return {"query": "hello", "status": "complete", "results": []}

    fake = _FakeDaemon(sock, handler)
    fake.start()
    try:
        _wait_for_socket(sock)
        client = DaemonClient(str(tmp_path), request_timeout_s=5.0)
        result = client.search("hello", paths=[str(tmp_path)], top_k=3)
        assert result["status"] == "complete"
    finally:
        fake.stop()


def test_call_index_round_trip(tmp_path: Path) -> None:
    sock = tmp_path / ".ragrag" / "daemon.sock"

    def handler(req: Request) -> Any:
        assert req.method == "index"
        assert req.params["index_path"] == str(tmp_path)
        return {"stats": {"files_added": 7}, "skipped": []}

    fake = _FakeDaemon(sock, handler)
    fake.start()
    try:
        _wait_for_socket(sock)
        client = DaemonClient(str(tmp_path), request_timeout_s=5.0)
        result = client.index([str(tmp_path)])
        assert result["stats"]["files_added"] == 7
    finally:
        fake.stop()


def test_call_status_round_trip(tmp_path: Path) -> None:
    sock = tmp_path / ".ragrag" / "daemon.sock"

    def handler(req: Request) -> Any:
        return {"version": "0.1.0", "device_mode": "cpu"}

    fake = _FakeDaemon(sock, handler)
    fake.start()
    try:
        _wait_for_socket(sock)
        client = DaemonClient(str(tmp_path), request_timeout_s=5.0)
        snap = client.status()
        assert snap["device_mode"] == "cpu"
    finally:
        fake.stop()


def test_call_error_response_raises_daemon_error(tmp_path: Path) -> None:
    sock = tmp_path / ".ragrag" / "daemon.sock"

    def handler(req: Request) -> Any:
        raise JsonRpcError(ERROR_INVALID_PARAMS, "no good")

    fake = _FakeDaemon(sock, handler)
    fake.start()
    try:
        _wait_for_socket(sock)
        client = DaemonClient(str(tmp_path), request_timeout_s=5.0)
        with pytest.raises(DaemonError) as exc_info:
            client.call("search", {"x": 1})
        assert exc_info.value.code == ERROR_INVALID_PARAMS
        assert "no good" in str(exc_info.value)
    finally:
        fake.stop()
