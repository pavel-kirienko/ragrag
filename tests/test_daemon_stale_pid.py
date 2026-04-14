"""Stale-pid detection in DaemonClient."""
from __future__ import annotations

from pathlib import Path

from ragrag.daemon.client import DaemonClient


def test_cleanup_stale_removes_dead_pid_and_socket(tmp_path: Path) -> None:
    ragrag_dir = tmp_path / ".ragrag"
    ragrag_dir.mkdir()
    sock = ragrag_dir / "daemon.sock"
    pid_file = ragrag_dir / "daemon.pid"
    sock.write_bytes(b"")  # empty file standing in for a stale socket
    pid_file.write_text("999999\n0\n1\n0\n", encoding="utf-8")  # almost-certainly-dead PID

    client = DaemonClient(str(tmp_path))
    assert not client._socket_alive()
    client._cleanup_stale()
    assert not sock.exists()
    assert not pid_file.exists()


def test_cleanup_stale_keeps_live_pid(tmp_path: Path) -> None:
    import os

    ragrag_dir = tmp_path / ".ragrag"
    ragrag_dir.mkdir()
    sock = ragrag_dir / "daemon.sock"
    pid_file = ragrag_dir / "daemon.pid"
    sock.write_bytes(b"")
    pid_file.write_text(f"{os.getpid()}\n0\n1\n0\n", encoding="utf-8")  # always alive

    client = DaemonClient(str(tmp_path))
    client._cleanup_stale()
    # PID is alive → cleanup should NOT touch the files
    assert sock.exists()
    assert pid_file.exists()


def test_read_pid_handles_missing_file(tmp_path: Path) -> None:
    client = DaemonClient(str(tmp_path))
    assert client._read_pid() is None


def test_read_pid_handles_garbage(tmp_path: Path) -> None:
    ragrag_dir = tmp_path / ".ragrag"
    ragrag_dir.mkdir()
    (ragrag_dir / "daemon.pid").write_text("not a number\n", encoding="utf-8")
    client = DaemonClient(str(tmp_path))
    assert client._read_pid() is None
