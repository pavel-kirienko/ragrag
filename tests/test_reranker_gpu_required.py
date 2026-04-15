"""The worker must refuse to run on CPU when --require-gpu is set.

We monkeypatch ``load_vlm`` so no real weights are touched; the stub
returns a handle pretending to have landed on ``cpu``. The worker
should emit a ``{"status": "fatal", ...}`` line on stdout and exit
with a non-zero code instead of accepting any rerank commands.
"""
from __future__ import annotations

import io
import json
import sys

import pytest

from ragrag.embedding.vlm_loader import VLMHandle
from ragrag.retrieval import vlm_rerank_worker


class _SentinelModel:
    """Smallest possible object with ``.parameters()`` for VLMHandle.generate."""


class _SentinelProcessor:
    pass


def _fake_handle(device: str) -> VLMHandle:
    return VLMHandle(
        model_id="stub-model",
        device=device,
        quantization="none",
        model=_SentinelModel(),
        processor=_SentinelProcessor(),
    )


def test_worker_refuses_cpu_under_require_gpu(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        vlm_rerank_worker,
        "__name__",
        "ragrag.retrieval.vlm_rerank_worker",
        raising=False,
    )

    monkeypatch.setattr(
        "ragrag.embedding.vlm_loader.load_vlm",
        lambda *a, **k: _fake_handle("cpu"),
    )

    # Empty stdin so if the worker *did* accept commands the main loop
    # would exit cleanly on EOF. We still fail the test in that case
    # because the fatal frame must appear.
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(sys, "argv", [
        "rerank_worker",
        "--model-id", "vikhyatk/moondream2",
        "--require-gpu",
    ])

    rc = vlm_rerank_worker.main()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert lines, "worker produced no stdout"
    first = json.loads(lines[0])
    assert first.get("status") == "fatal", first
    assert "cpu" in (first.get("error") or "").lower()
    assert rc != 0


def test_worker_accepts_cuda_handle(monkeypatch) -> None:
    """When the handle lands on cuda, the worker emits 'ready' and
    then exits cleanly on EOF instead of fatally."""
    monkeypatch.setattr(
        "ragrag.embedding.vlm_loader.load_vlm",
        lambda *a, **k: _fake_handle("cuda"),
    )

    # Avoid hitting the torch.cuda probe in the activation-headroom
    # check — we just want to confirm the fatal branch is NOT taken.
    import ragrag.retrieval.vlm_rerank_worker as worker_mod

    class _StubTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class backends:
            class cudnn:
                enabled = True

    # Swap the whole 'torch' import the worker makes in its probe by
    # replacing the probe block's import target via sys.modules.
    monkeypatch.setitem(sys.modules, "torch", _StubTorch)  # type: ignore[arg-type]

    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # EOF → clean loop exit
    monkeypatch.setattr(sys, "argv", [
        "rerank_worker",
        "--model-id", "vikhyatk/moondream2",
        "--require-gpu",
    ])

    # Drain stdout so capsys in a later test doesn't bleed in.
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    rc = worker_mod.main()

    sys.stdout = sys.__stdout__  # restore for pytest
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    # Expect: ready (no fatal). shutdown path reached via EOF → rc 0.
    assert lines, f"no output, rc={rc}"
    statuses = [json.loads(line).get("status") for line in lines]
    assert "ready" in statuses
    assert "fatal" not in statuses
    assert rc == 0
