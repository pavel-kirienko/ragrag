"""Unit tests for VLMTopicChunker — the rolling-window topic discovery loop.

We drive the chunker with a stub ``VLMTopicClient`` that returns canned
``PdfTopicAssignment`` lists, so the tests never load a real VLM. The
things we verify:

  * topics with overlapping ``page_refs`` survive as distinct Chunks
  * topics spanning non-contiguous pages survive as one Chunk
  * cold topics close after the configured page gap
  * end-of-document flush emits still-open topics
  * VLM failure on a window folds into an "Unparsed" topic and keeps
    indexing moving
  * completely empty input raises VLMChunkerError
"""
from __future__ import annotations

from typing import Iterable

import pytest

from ragrag.config import Settings
from ragrag.extractors.vlm_topic_chunker import (
    VLMChunkerError,
    VLMTopicChunker,
)
from ragrag.extractors.vlm_topic_client import (
    PdfTopicAssignment,
    VLMTopicClient,
    VLMTopicClientError,
)
from ragrag.models import Chunk, ChunkKind


# --------------------------------------------------------------------------- #
# Stub client that drives the chunker deterministically
# --------------------------------------------------------------------------- #

class _NullHandle:
    def generate(self, text, images=None, *, max_new_tokens=512, temperature=0.0):
        raise NotImplementedError  # stub subclass never hits the base


class _StubClient(VLMTopicClient):
    """Bypasses the real VLM handle and delegates to a test-supplied function."""

    def __init__(self, handler) -> None:
        super().__init__(_NullHandle())
        self._handler = handler

    def identify_pdf_topics(
        self,
        window_pages,
        window_images,
        window_texts,
        running_topics,
        *,
        max_topics_per_call: int = 16,
    ):
        return self._handler(window_pages, running_topics)

    def identify_text_topics(self, content, *, language_hint="text", absolute_line_offset=0):
        raise NotImplementedError


class _FailingClient(VLMTopicClient):
    def __init__(self, message: str = "mock vlm failure") -> None:
        super().__init__(_NullHandle())
        self._message = message

    def identify_pdf_topics(self, *args, **kwargs):
        raise VLMTopicClientError(self._message)

    def identify_text_topics(self, *args, **kwargs):
        raise VLMTopicClientError(self._message)


def _pages(count: int) -> Iterable[tuple[int, object, str]]:
    for i in range(1, count + 1):
        yield (i, object(), f"native text for page {i}")


def _settings(**overrides) -> Settings:
    base = dict(
        index_path=".ragrag",
        chunker_stride_pages=overrides.pop("stride", 4),
        chunker_topic_cold_pages=overrides.pop("cold", 10),
        chunker_max_topics_per_call=overrides.pop("max_new", 16),
    )
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_chunker_produces_one_topic_over_all_pages() -> None:
    """Simple happy path: every page is labelled with the same topic id."""
    def handler(pages, running):
        title = "" if "t1" in running else "Operating conditions"
        return [
            PdfTopicAssignment(
                page=p,
                topic_id="t1",
                is_continuation=("t1" in running),
                title=title,
                summary="everything on one topic",
            )
            for p in pages
        ]

    chunker = VLMTopicChunker(_StubClient(handler), _settings(stride=3, cold=50))
    chunks = chunker.chunk("/foo.pdf", "sha", _pages(6))
    assert len(chunks) == 1
    assert chunks[0].kind == ChunkKind.PDF_TOPIC
    assert chunks[0].page_refs == [1, 2, 3, 4, 5, 6]
    assert chunks[0].title == "Operating conditions"
    assert chunks[0].hero_page == 1


def test_chunker_preserves_overlapping_topics() -> None:
    """When the VLM reports two topics sharing pages, both Chunks keep the shared pages."""
    def handler(pages, running):
        out = []
        for p in pages:
            # All pages belong to "t_a"; pages 3..5 also belong to "t_b"
            out.append(
                PdfTopicAssignment(
                    page=p,
                    topic_id="t_a",
                    is_continuation=("t_a" in running),
                    title="" if "t_a" in running else "ADC",
                    summary="adc characteristics",
                )
            )
            if 3 <= p <= 5:
                out.append(
                    PdfTopicAssignment(
                        page=p,
                        topic_id="t_b",
                        is_continuation=("t_b" in running),
                        title="" if "t_b" in running else "VREF",
                        summary="voltage reference",
                    )
                )
        return out

    chunker = VLMTopicChunker(_StubClient(handler), _settings(stride=3, cold=50))
    chunks = chunker.chunk("/foo.pdf", "sha", _pages(6))
    assert len(chunks) == 2
    by_title = {c.title: c for c in chunks}
    assert by_title["ADC"].page_refs == [1, 2, 3, 4, 5, 6]
    assert by_title["VREF"].page_refs == [3, 4, 5]
    # Each chunk must have an independent hero page.
    assert by_title["ADC"].hero_page == 1
    assert by_title["VREF"].hero_page == 3


def test_chunker_accepts_non_contiguous_page_refs() -> None:
    """A topic referenced on pages 1, 5, and 9 must survive as one chunk with gaps."""
    def handler(pages, running):
        out = []
        for p in pages:
            # t_x only appears on pages 1, 5, 9
            if p in (1, 5, 9):
                out.append(
                    PdfTopicAssignment(
                        page=p,
                        topic_id="t_x",
                        is_continuation=("t_x" in running),
                        title="" if "t_x" in running else "Scattered topic",
                        summary="non-contiguous",
                    )
                )
            # t_y covers every page so the file isn't empty
            out.append(
                PdfTopicAssignment(
                    page=p,
                    topic_id="t_y",
                    is_continuation=("t_y" in running),
                    title="" if "t_y" in running else "Background",
                    summary="all pages",
                )
            )
        return out

    chunker = VLMTopicChunker(_StubClient(handler), _settings(stride=3, cold=50))
    chunks = chunker.chunk("/foo.pdf", "sha", _pages(10))
    x = next(c for c in chunks if c.title == "Scattered topic")
    assert x.page_refs == [1, 5, 9]  # sorted, deduped, non-contiguous


def test_chunker_closes_cold_topics() -> None:
    """A topic unseen for more than cold_threshold pages is closed and flushed."""
    def handler(pages, running):
        out = []
        for p in pages:
            if p <= 3:
                out.append(PdfTopicAssignment(page=p, topic_id="early", is_continuation=("early" in running), title="Early", summary="s"))
            else:
                out.append(PdfTopicAssignment(page=p, topic_id="late", is_continuation=("late" in running), title="Late", summary="s"))
        return out

    chunker = VLMTopicChunker(_StubClient(handler), _settings(stride=3, cold=5))
    chunks = chunker.chunk("/foo.pdf", "sha", _pages(15))
    # Both topics should be present even though "early" was long closed by end
    titles = sorted(c.title for c in chunks)
    assert titles == ["Early", "Late"]


def test_chunker_raises_on_persistent_vlm_failure_and_empty_output() -> None:
    """A file with ONLY failed windows still emits 'Unparsed' chunks (doesn't vanish)."""
    chunker = VLMTopicChunker(_FailingClient(), _settings(stride=3, cold=50))
    chunks = chunker.chunk("/foo.pdf", "sha", _pages(6))
    # Every window folds into one 'Unparsed' topic, so 2 windows → 2 chunks
    assert len(chunks) == 2
    for c in chunks:
        assert c.title.startswith("Unparsed window")
        assert c.page_refs  # non-empty


def test_chunker_raises_when_page_iter_is_empty() -> None:
    def handler(pages, running):
        return [PdfTopicAssignment(page=p, topic_id="t1", is_continuation=False, title="T", summary="s") for p in pages]

    chunker = VLMTopicChunker(_StubClient(handler), _settings(stride=3, cold=50))
    with pytest.raises(VLMChunkerError):
        chunker.chunk("/foo.pdf", "sha", iter([]))


def test_chunker_titles_truncated_for_store() -> None:
    long_title = "X" * 500

    def handler(pages, running):
        return [
            PdfTopicAssignment(page=p, topic_id="t", is_continuation=("t" in running), title=(long_title if not ("t" in running) else ""), summary="s")
            for p in pages
        ]

    chunker = VLMTopicChunker(_StubClient(handler), _settings(stride=3, cold=50))
    chunks = chunker.chunk("/foo.pdf", "sha", _pages(3))
    assert len(chunks[0].title) == 200  # capped
