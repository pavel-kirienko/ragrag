"""CPU detection in the daemon's device-mode logic."""
from __future__ import annotations

import pytest

from ragrag.daemon.server import _detect_device_mode


def test_detect_device_mode_returns_known_value() -> None:
    mode = _detect_device_mode()
    assert mode in {"cuda", "cpu", "mps"}


def test_detect_device_mode_falls_back_to_cpu_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """If torch import fails, the daemon must report CPU mode."""
    import sys

    saved = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        # Re-import the function so it re-runs the import inside.
        from ragrag.daemon.server import _detect_device_mode as fresh_detect

        mode = fresh_detect()
        assert mode == "cpu"
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
        else:
            sys.modules.pop("torch", None)


def test_daemon_state_starts_in_known_device_mode(tmp_path) -> None:
    """A fresh DaemonServer captures device_mode at construction time."""
    from ragrag.daemon.server import DaemonServer

    server = DaemonServer(
        index_path=str(tmp_path),
        socket_path=tmp_path / "daemon.sock",
        pid_path=tmp_path / "daemon.pid",
        idle_timeout_s=86400,
    )
    assert server.state.device_mode in {"cuda", "cpu", "mps"}
