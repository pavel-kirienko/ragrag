"""Verify the parent-side rerank spawner passes the moondream model id.

The reranker module builds the subprocess argv from settings and then
hands it to ``subprocess.Popen``. We monkeypatch Popen with a recorder
so no child is actually launched, then assert that:

  * ``settings.reranker_model_id`` ends up in the spawn argv,
  * ``--require-gpu`` is present by default,
  * the activation-headroom flag is wired through.
"""
from __future__ import annotations

import json
import threading

from ragrag.config import Settings
from ragrag.retrieval.reranker import VLMReranker


class _Pipe:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._cv = threading.Condition()

    def write(self, line: str) -> None:
        with self._cv:
            self._lines.append(line)
            self._cv.notify_all()

    def flush(self) -> None:
        return None

    def readline(self) -> str:
        with self._cv:
            while not self._lines:
                self._cv.wait(timeout=1.0)
                if not self._lines:
                    return ""
            return self._lines.pop(0)


class _RecordingWorker:
    """Popen stand-in that captures argv and replies with a ready frame."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.stdin = _Pipe()
        self.stdout = _Pipe()
        self.stdout.write(
            json.dumps({"status": "ready", "device": "cuda", "model_id": "stub"}) + "\n"
        )

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        return None


def test_reranker_spawn_uses_moondream_model_id(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_popen(argv, **_kwargs):  # type: ignore[no-untyped-def]
        captured["argv"] = list(argv)
        return _RecordingWorker(list(argv))

    monkeypatch.setattr("ragrag.retrieval.reranker.subprocess.Popen", _fake_popen)

    settings = Settings(
        index_path="/tmp/ragrag-moondream-test",
        reranker_model="vlm",
        reranker_model_id="vikhyatk/moondream2",
        reranker_require_gpu=True,
        moondream_activation_headroom_mib=384,
    )
    rr = VLMReranker(settings)
    try:
        rr._spawn()  # type: ignore[attr-defined]
    finally:
        rr.close()

    argv = captured["argv"]
    assert "--model-id" in argv
    model_idx = argv.index("--model-id")
    assert argv[model_idx + 1] == "vikhyatk/moondream2"
    assert "--require-gpu" in argv
    assert "--activation-headroom-mib" in argv
    hdx = argv.index("--activation-headroom-mib")
    assert argv[hdx + 1] == "384"
    # And the chunker model id must NOT have been used — the two
    # knobs have to stay independent so the chunker can keep using
    # Qwen2.5-VL-3B while the reranker uses Moondream2.
    assert argv[model_idx + 1] != settings.vlm_model_id


def test_reranker_spawn_honors_require_gpu_off(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_popen(argv, **_kwargs):  # type: ignore[no-untyped-def]
        captured["argv"] = list(argv)
        return _RecordingWorker(list(argv))

    monkeypatch.setattr("ragrag.retrieval.reranker.subprocess.Popen", _fake_popen)

    settings = Settings(
        index_path="/tmp/ragrag-moondream-test",
        reranker_model="vlm",
        reranker_require_gpu=False,
    )
    rr = VLMReranker(settings)
    try:
        rr._spawn()  # type: ignore[attr-defined]
    finally:
        rr.close()

    assert "--require-gpu" not in captured["argv"]
