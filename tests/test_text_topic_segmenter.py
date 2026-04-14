"""Unit tests for the VLM-driven text topic segmenter."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragrag.config import Settings
from ragrag.extractors.text_topic_segmenter import (
    TextSegmenterError,
    TextTopicSegmenter,
    _dedupe_ranges,
    _line_ranges_to_byte_ranges,
)
from ragrag.extractors.vlm_topic_client import TextTopic, VLMTopicClient, VLMTopicClientError
from ragrag.models import ChunkKind


class _StubHandle:
    def generate(self, text, images=None, *, max_new_tokens=512, temperature=0.0):
        raise NotImplementedError  # subclass doesn't call the base handle


class _StubClient(VLMTopicClient):
    def __init__(self, topics_by_offset: dict[int, list[TextTopic]]) -> None:
        super().__init__(_StubHandle())
        self._topics_by_offset = topics_by_offset
        self.calls: list[tuple[int, int]] = []  # (content_len, offset)

    def identify_text_topics(self, content, *, language_hint="text", absolute_line_offset=0):
        self.calls.append((len(content), absolute_line_offset))
        if absolute_line_offset in self._topics_by_offset:
            return self._topics_by_offset[absolute_line_offset]
        return self._topics_by_offset.get(0, [])

    def identify_pdf_topics(self, *args, **kwargs):
        raise NotImplementedError


class _FailingClient(VLMTopicClient):
    def __init__(self) -> None:
        super().__init__(_StubHandle())

    def identify_text_topics(self, *a, **kw):
        raise VLMTopicClientError("stub failure")

    def identify_pdf_topics(self, *a, **kw):
        raise NotImplementedError


def _write_file(tmp_path: Path, name: str, content: str) -> Path:
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


def _settings(ctx_tokens: int = 8192) -> Settings:
    return Settings(index_path=".ragrag", chunker_vlm_ctx_tokens=max(512, ctx_tokens))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_segment_small_file_single_topic(tmp_path: Path) -> None:
    f = _write_file(tmp_path, "hello.txt", "first line\nsecond line\nthird line\n")
    client = _StubClient({
        0: [TextTopic(title="hello", summary="s", ranges=[(1, 3)])],
    })
    seg = TextTopicSegmenter(client, _settings())
    chunks = seg.segment(str(f))
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.kind == ChunkKind.TEXT_TOPIC
    assert chunk.title == "hello"
    assert chunk.line_ranges == [(1, 3)]
    # byte_ranges should cover the whole file
    assert chunk.byte_ranges[0][0] == 0
    assert chunk.byte_ranges[0][1] == len(f.read_text(encoding="utf-8"))
    assert "first line" in chunk.excerpt


def test_segment_overlapping_topics_on_same_file(tmp_path: Path) -> None:
    content = "\n".join(f"line {i}" for i in range(1, 11)) + "\n"
    f = _write_file(tmp_path, "a.py", content)
    client = _StubClient({
        0: [
            TextTopic(title="topic a", summary="s", ranges=[(1, 6)]),
            TextTopic(title="topic b", summary="s", ranges=[(4, 10)]),
        ],
    })
    seg = TextTopicSegmenter(client, _settings())
    chunks = seg.segment(str(f))
    assert [c.title for c in chunks] == ["topic a", "topic b"]
    # Overlapping line ranges survive
    assert chunks[0].line_ranges == [(1, 6)]
    assert chunks[1].line_ranges == [(4, 10)]


def test_segment_non_contiguous_ranges_survive(tmp_path: Path) -> None:
    content = "line\n" * 30
    f = _write_file(tmp_path, "x.md", content)
    client = _StubClient({
        0: [TextTopic(title="split", summary="s", ranges=[(1, 5), (20, 25)])],
    })
    seg = TextTopicSegmenter(client, _settings())
    chunks = seg.segment(str(f))
    assert chunks[0].line_ranges == [(1, 5), (20, 25)]
    # byte_ranges has two non-contiguous spans
    assert len(chunks[0].byte_ranges) == 2
    assert chunks[0].byte_ranges[0][1] < chunks[0].byte_ranges[1][0]


def test_segment_sliding_window_merges_titles(tmp_path: Path) -> None:
    # Make the file big so the ctx_tokens budget forces a slide.
    content = "\n".join(f"line {i}" for i in range(1, 1001)) + "\n"
    f = _write_file(tmp_path, "big.txt", content)
    # ctx_tokens=100 → approx_tokens > 100 → sliding path
    client = _StubClient({
        0: [TextTopic(title="section a", summary="first", ranges=[(1, 30)])],
    })
    seg = TextTopicSegmenter(client, _settings(ctx_tokens=100))
    # stub's identify_text_topics returns the same topics regardless of offset
    chunks = seg.segment(str(f))
    # Even though the segmenter slid multiple windows, titles merge to one.
    assert len(chunks) == 1
    assert chunks[0].title == "section a"


def test_segment_empty_file_returns_no_chunks(tmp_path: Path) -> None:
    f = _write_file(tmp_path, "empty.txt", "")
    client = _StubClient({0: [TextTopic(title="t", summary="s", ranges=[(1, 1)])]})
    seg = TextTopicSegmenter(client, _settings())
    chunks = seg.segment(str(f))
    assert chunks == []


def test_segment_raises_on_vlm_failure(tmp_path: Path) -> None:
    f = _write_file(tmp_path, "x.py", "def foo():\n    pass\n")
    seg = TextTopicSegmenter(_FailingClient(), _settings())
    with pytest.raises(TextSegmenterError):
        seg.segment(str(f))


def test_dedupe_ranges_merges_adjacent() -> None:
    merged = _dedupe_ranges([(5, 10), (11, 20), (30, 40)])
    assert merged == [(5, 20), (30, 40)]


def test_dedupe_ranges_merges_overlap() -> None:
    merged = _dedupe_ranges([(5, 15), (10, 20)])
    assert merged == [(5, 20)]


def test_line_ranges_to_byte_ranges_preserves_non_contiguous() -> None:
    content = "".join(f"L{i}\n" for i in range(1, 11))  # L1\n ... L10\n
    byte_ranges = _line_ranges_to_byte_ranges(content, [(1, 2), (6, 7)])
    assert len(byte_ranges) == 2
    assert content[byte_ranges[0][0]:byte_ranges[0][1]] == "L1\nL2\n"
    assert content[byte_ranges[1][0]:byte_ranges[1][1]] == "L6\nL7\n"
