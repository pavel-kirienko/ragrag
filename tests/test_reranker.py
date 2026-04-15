"""Unit tests for the Phase D VLM reranker.

The reranker runs its VLM inside a subprocess, but for unit tests we
monkeypatch ``subprocess.Popen`` with a fake line-based protocol
driver so we don't need to load any model weights.
"""
from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from ragrag.config import Settings
from ragrag.models import Chunk, ChunkKind, Location, SearchResult
from ragrag.retrieval.reranker import (
    VLMReranker,
    _format_pages,
    _pick_hero_image,
    _reassign_ranks,
)
from ragrag.retrieval.vlm_rerank_worker import _build_prompt, _parse_ranks


# --------------------------------------------------------------------------- #
# Helpers: fake worker Popen
# --------------------------------------------------------------------------- #


class _FakePipe:
    """Two-ended line-buffered pipe shared between parent and fake worker."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._lines: list[str] = []
        self._closed = False

    def write(self, line: str) -> None:
        with self._cv:
            self._lines.append(line)
            self._cv.notify_all()

    def flush(self) -> None:
        return None

    def readline(self) -> str:
        with self._cv:
            while not self._lines and not self._closed:
                self._cv.wait(timeout=2.0)
                if not self._lines and self._closed:
                    return ""
            if self._lines:
                return self._lines.pop(0)
            return ""

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class FakeWorker:
    """Fake subprocess that speaks the reranker's line-based protocol."""

    def __init__(self, ranks_fn) -> None:
        self.ranks_fn = ranks_fn
        self.stdin = _FakePipe()
        self.stdout = _FakePipe()
        self._alive = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # Emit the ready sentinel that the client blocks on after spawn.
        self.stdout.write(
            json.dumps({"status": "ready", "device": "cpu", "model_id": "stub"}) + "\n"
        )

    def _loop(self) -> None:
        while self._alive:
            line = self.stdin.readline()
            if not line:
                break
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self.stdout.write(json.dumps({"status": "error", "error": "bad json"}) + "\n")
                continue
            cmd = request.get("cmd")
            if cmd == "shutdown":
                self.stdout.write(json.dumps({"status": "bye"}) + "\n")
                self._alive = False
                break
            if cmd == "ping":
                self.stdout.write(json.dumps({"status": "pong"}) + "\n")
                continue
            if cmd == "rerank":
                candidates = request.get("candidates") or []
                ranks = self.ranks_fn(request.get("query"), candidates)
                self.stdout.write(json.dumps({"status": "ok", "ranks": ranks}) + "\n")
                continue
            self.stdout.write(json.dumps({"status": "error", "error": "unknown"}) + "\n")

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self._alive = False
        self.stdin.close()
        self.stdout.close()
        return 0

    def kill(self) -> None:
        self.wait()


def _fake_popen_factory(ranks_fn):
    def _spawn(*_args, **_kwargs):
        return FakeWorker(ranks_fn)
    return _spawn


def _make_result(rank: int, chunk_id: str, title: str, pages: list[int], summary: str = "") -> SearchResult:
    return SearchResult(
        rank=rank,
        score=1.0 / rank,
        path="/fake/doc.pdf",
        file_type="pdf",
        modality="text",
        excerpt=f"{title}\n{summary}",
        chunk_id=chunk_id,
        title=title,
        summary=summary,
        page_refs=pages,
        context_pages=[],
        location=Location(path="/fake/doc.pdf", directory="/fake"),
    )


# --------------------------------------------------------------------------- #
# Parser / helper tests (no subprocess)
# --------------------------------------------------------------------------- #


def test_format_pages_contiguous_and_gap() -> None:
    assert _format_pages([1, 2, 3, 5, 6]) == "1-3, 5-6"
    assert _format_pages([7, 2, 8, 3, 1]) == "1-3, 7-8"
    assert _format_pages([]) == ""
    assert _format_pages([42]) == "42"


def test_pick_hero_image_prefers_context() -> None:
    from ragrag.models import PageContext

    r = SearchResult(
        rank=1, score=1.0, path="/x/y.pdf", file_type="pdf", modality="image",
        excerpt="",
        context_pages=[
            PageContext(page=3, page_image_path=None),
            PageContext(page=4, page_image_path="/x/cache/4.webp"),
        ],
    )
    assert _pick_hero_image(r) == "/x/cache/4.webp"


def test_reassign_ranks_preserves_order_and_sets_rank() -> None:
    rs = [
        _make_result(5, "a", "A", [1]),
        _make_result(2, "b", "B", [2]),
        _make_result(9, "c", "C", [3]),
    ]
    out = _reassign_ranks(rs)
    assert [r.rank for r in out] == [1, 2, 3]
    assert [r.chunk_id for r in out] == ["a", "b", "c"]


def test_parse_ranks_accepts_json_array() -> None:
    raw = '[{"id":0,"rank":1,"score":9,"reason":"perfect match"}]'
    out = _parse_ranks(raw, [{"id": 0}])
    assert out == [{"id": 0, "rank": 1, "score": 9.0, "reason": "perfect match"}]


def test_parse_ranks_strips_code_fence() -> None:
    raw = "```json\n[{\"id\": 0, \"rank\": 1}]\n```"
    out = _parse_ranks(raw, [{"id": 0}])
    assert out and out[0]["id"] == 0


def test_parse_ranks_drops_unknown_ids() -> None:
    raw = '[{"id":99,"rank":1}]'
    assert _parse_ranks(raw, [{"id": 0}]) == []


def test_build_prompt_contains_all_candidate_ids() -> None:
    candidates = [
        {"id": 0, "title": "ADC", "summary": "ADC stuff", "pages": "1-5", "excerpt": "…"},
        {"id": 1, "title": "DAC", "summary": "DAC stuff", "pages": "6-9", "excerpt": "…"},
    ]
    prompt = _build_prompt("ADC sample rate", candidates)
    assert "ADC sample rate" in prompt
    assert "[0] ADC" in prompt
    assert "[1] DAC" in prompt
    assert "JSON" in prompt
    assert '"id"' in prompt and '"rank"' in prompt


# --------------------------------------------------------------------------- #
# Integration: fake subprocess end-to-end
# --------------------------------------------------------------------------- #


def _settings() -> Settings:
    return Settings(
        index_path="/tmp/ragrag-rerank-test",
        reranker_model="vlm",
        rerank_max_candidates=5,
    )


def test_reranker_happy_path(monkeypatch) -> None:
    def ranks_fn(_query: str, candidates: list[dict]) -> list[dict]:
        # Invert: highest id ends up first.
        return [
            {"id": c["id"], "rank": len(candidates) - c["id"], "score": 9 - c["id"], "reason": f"r{c['id']}"}
            for c in candidates
        ]

    monkeypatch.setattr(
        "ragrag.retrieval.reranker.subprocess.Popen",
        _fake_popen_factory(ranks_fn),
    )
    rr = VLMReranker(_settings())
    try:
        results = [
            _make_result(1, "a", "Topic A", [1, 2, 3]),
            _make_result(2, "b", "Topic B", [4, 5]),
            _make_result(3, "c", "Topic C", [6]),
        ]
        out = rr.rerank("query", results)
        assert [r.chunk_id for r in out] == ["c", "b", "a"]
        assert out[0].rerank_reason == "r2"
        assert [r.rank for r in out] == [1, 2, 3]
    finally:
        rr.close()


def test_reranker_handles_worker_error_by_falling_back(monkeypatch) -> None:
    def ranks_fn(_query, _candidates):
        raise RuntimeError("boom")

    class _RaisingWorker(FakeWorker):
        def __init__(self) -> None:
            super().__init__(lambda q, c: [])

        def _loop(self) -> None:
            # Emit ready then immediately close stdout on the first request.
            line = self.stdin.readline()
            if not line:
                return
            self.stdout.close()

    def _spawn(*_a, **_k) -> FakeWorker:
        return _RaisingWorker()

    monkeypatch.setattr(
        "ragrag.retrieval.reranker.subprocess.Popen",
        _spawn,
    )
    rr = VLMReranker(_settings())
    try:
        results = [
            _make_result(1, "a", "A", [1]),
            _make_result(2, "b", "B", [2]),
        ]
        out = rr.rerank("query", results)
        # Fallback preserves original order when the worker errors.
        assert [r.chunk_id for r in out] == ["a", "b"]
    finally:
        rr.close()


def test_reranker_respects_max_candidates(monkeypatch) -> None:
    seen_batches: list[int] = []

    def ranks_fn(_query: str, candidates: list[dict]) -> list[dict]:
        seen_batches.append(len(candidates))
        return [{"id": c["id"], "rank": i + 1, "score": 1, "reason": ""} for i, c in enumerate(candidates)]

    monkeypatch.setattr(
        "ragrag.retrieval.reranker.subprocess.Popen",
        _fake_popen_factory(ranks_fn),
    )
    rr = VLMReranker(_settings())  # max_candidates=5
    try:
        results = [_make_result(i + 1, f"c{i}", f"T{i}", [i + 1]) for i in range(8)]
        out = rr.rerank("q", results)
        assert seen_batches == [5], seen_batches
        # Worker only saw the first 5 candidates; the remaining 3
        # results are preserved at their original relative positions.
        assert [r.chunk_id for r in out][:5] == ["c0", "c1", "c2", "c3", "c4"]
        assert [r.chunk_id for r in out][5:] == ["c5", "c6", "c7"]
        assert [r.rank for r in out] == list(range(1, 9))
    finally:
        rr.close()
