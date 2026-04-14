"""Thin JSON-RPC client for the ragrag daemon.

The CLI is the only user. Behaviour:

  1. Try to connect to ``<index>/.ragrag/daemon.sock``.
  2. If the socket is missing, dead, or stale, optionally spawn
     ``python -m ragrag.daemon --detach --index-path <index>`` and poll for
     readiness up to ``ensure_timeout_s``.
  3. If the daemon still isn't reachable, raise ``DaemonStartupError``.
  4. ``CLI`` callers catch that exception and fall back to the in-process
     SearchEngine path. We never crash the user query because the daemon
     can't start.

The client opens one socket per request — there is no pooling. Requests are
single-shot, response ≤ a few hundred KiB, this is fine.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from ragrag.daemon import rpc
from ragrag.daemon.rpc import (
    JsonRpcError,
    Request,
    Response,
    decode_response,
    encode_request,
)


class DaemonStartupError(RuntimeError):
    """Raised when the client cannot connect to or spawn a daemon."""


class DaemonError(RuntimeError):
    """Wraps a JSON-RPC error returned by the daemon."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"daemon error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class DaemonClient:
    """Per-index client. One instance per ``DaemonClient(index_path)``.

    Most users construct it once at the top of ``cli.main`` and call
    ``ensure_daemon()`` then ``search(...)``.
    """

    def __init__(
        self,
        index_path: str,
        *,
        ensure_timeout_s: float = 60.0,
        request_timeout_s: float = 7200.0,
    ) -> None:
        # Accept either the knowledge-base root or the ``.ragrag`` directory
        # itself. Settings.index_path conventionally points at ``.ragrag``;
        # the daemon and CLI subcommands pass that in directly.
        abs_path = Path(os.path.abspath(index_path))
        if abs_path.name == ".ragrag":
            self.ragrag_dir = abs_path
            self.index_path = str(abs_path.parent)
        else:
            self.ragrag_dir = abs_path / ".ragrag"
            self.index_path = str(abs_path)
        self.socket_path = self.ragrag_dir / "daemon.sock"
        self.pid_path = self.ragrag_dir / "daemon.pid"
        self.ensure_timeout_s = float(ensure_timeout_s)
        self.request_timeout_s = float(request_timeout_s)

    # ------------------------------------------------------------------ #
    # Connection / spawn
    # ------------------------------------------------------------------ #

    def _socket_alive(self) -> bool:
        if not self.socket_path.exists():
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(str(self.socket_path))
            return True
        except (OSError, socket.timeout):
            return False

    def _read_pid(self) -> int | None:
        try:
            text = self.pid_path.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError):
            return None
        line = text.splitlines()[0] if text else ""
        try:
            return int(line)
        except ValueError:
            return None

    def _process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but not ours

    def _cleanup_stale(self) -> None:
        pid = self._read_pid()
        if pid is not None and not self._process_alive(pid):
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
            try:
                self.pid_path.unlink()
            except FileNotFoundError:
                pass

    def _spawn(self) -> None:
        self.ragrag_dir.mkdir(parents=True, exist_ok=True)
        # Use the same Python interpreter so we inherit the venv.
        cmd = [sys.executable, "-m", "ragrag.daemon", "--detach", "--index-path", self.index_path]
        try:
            subprocess.run(cmd, check=True, timeout=10)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            raise DaemonStartupError(f"failed to spawn daemon: {exc}") from exc

    def ensure_daemon(self) -> None:
        """Block until the daemon is reachable on the socket.

        Spawns it if missing; cleans up stale socket+pid files; gives up
        after ``ensure_timeout_s`` seconds with a ``DaemonStartupError``.
        """
        deadline = time.monotonic() + self.ensure_timeout_s
        if self._socket_alive():
            return
        self._cleanup_stale()
        self._spawn()
        while time.monotonic() < deadline:
            if self._socket_alive():
                return
            time.sleep(0.2)
        raise DaemonStartupError(
            f"daemon socket {self.socket_path} did not appear within {self.ensure_timeout_s:.0f}s"
        )

    # ------------------------------------------------------------------ #
    # RPC
    # ------------------------------------------------------------------ #

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = uuid.uuid4().hex
        line = encode_request(method, params, request_id)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(self.request_timeout_s)
            s.connect(str(self.socket_path))
            s.sendall(line)
            buf = bytearray()
            while True:
                try:
                    chunk = s.recv(8192)
                except socket.timeout as exc:
                    raise DaemonError(-32000, f"daemon timed out after {self.request_timeout_s}s") from exc
                if not chunk:
                    break
                buf.extend(chunk)
                if b"\n" in chunk:
                    break
        nl = buf.find(b"\n")
        line = bytes(buf[:nl]) if nl >= 0 else bytes(buf)
        if not line:
            raise DaemonError(-32000, "daemon closed connection before responding")
        try:
            response = decode_response(line)
        except JsonRpcError as exc:
            raise DaemonError(exc.code, exc.message, exc.data) from exc
        if response.error is not None:
            err = response.error
            raise DaemonError(int(err.get("code", -32000)), str(err.get("message", "")), err.get("data"))
        return response.result

    # ------------------------------------------------------------------ #
    # Convenience wrappers
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        paths: list[str] | None = None,
        *,
        top_k: int | None = None,
        include_markdown: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "index_path": self.index_path,
            "query": query,
            "paths": paths or [self.index_path],
            "include_markdown": include_markdown,
        }
        if top_k is not None:
            params["top_k"] = int(top_k)
        return self.call("search", params)

    def index(self, paths: list[str] | None = None, force: bool = False) -> dict[str, Any]:
        return self.call(
            "index",
            {"index_path": self.index_path, "paths": paths or [self.index_path], "force": force},
        )

    def status(self) -> dict[str, Any]:
        return self.call("status", {})

    def shutdown(self, wait_s: float = 10.0) -> dict[str, Any]:
        result = self.call("shutdown", {})
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if not self.socket_path.exists():
                return result
            time.sleep(0.1)
        return result
