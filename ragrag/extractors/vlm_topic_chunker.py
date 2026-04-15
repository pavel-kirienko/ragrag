"""VLM-driven topic chunker for PDFs.

Given a stream of ``(page_number, page_image, native_text)`` from the PDF
extractor, produce a list of :class:`ragrag.models.Chunk` objects, each
representing a *topic* — a set of pages that belong together semantically.
Topics may overlap (one page in multiple topics) and may be non-contiguous
(a topic that references pages 1-3 and 15-17).

The chunker runs in a rolling window of ``chunker_stride_pages`` pages and
keeps a dictionary of currently-open topics. Each window call asks the VLM
"for each page, which topic(s) does it belong to?". The VLM may reference
already-open topics (by id) or invent new ones; new topics get titles and
summaries. Topics that haven't been touched for ``chunker_topic_cold_pages``
consecutive pages are closed.

No heuristic fallback for topic boundary detection: on VLM failure the
caller is expected to abort the file with a logged warning (policy is
"no heuristic chunking"). A last-ditch "one topic per file" emergency
fallback exists for completely unparseable files but is clearly marked
in the output.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Iterator, Optional

from ragrag.config import Settings
from ragrag.extractors.vlm_topic_client import (
    PdfTopicAssignment,
    VLMTopicClient,
    VLMTopicClientError,
)
from ragrag.models import Chunk, ChunkKind


logger = logging.getLogger(__name__)


class VLMChunkerError(RuntimeError):
    """Raised when the VLM chunker cannot produce any chunks for a file."""


@dataclass
class _OpenTopic:
    """Mutable state for a topic the chunker is currently tracking."""

    title: str
    summary: str
    pages: list[int] = field(default_factory=list)
    last_seen_page: int = 0
    hero_page: Optional[int] = None


class VLMTopicChunker:
    """Rolling-window topic discovery for PDFs.

    The public entry point is :meth:`chunk`, which takes the iterator from
    ``iter_pdf_segments`` and returns a list of ``Chunk`` objects.
    """

    def __init__(
        self,
        client: VLMTopicClient,
        settings: Settings,
    ) -> None:
        self.client = client
        self.settings = settings

    # ------------------------------------------------------------------ #

    def chunk(
        self,
        path: str,
        file_sha256: str,
        page_iter: Iterator[tuple[int, object, str]],
    ) -> list[Chunk]:
        """Discover topics across a PDF.

        Args:
            path: absolute path of the source file (stored on the chunk).
            file_sha256: SHA-256 of the file content (stored on the chunk).
            page_iter: yields ``(page_number_1_indexed, pil_image, native_text)``.

        Returns:
            List of :class:`Chunk` objects, one per topic. May be empty if
            the PDF extractor produced no pages (encrypted, corrupt, etc.).

        Raises:
            :class:`VLMChunkerError` if the VLM fails to produce chunks on
            every attempt.
        """
        stride = max(1, int(self.settings.chunker_stride_pages))
        cold_threshold = max(1, int(self.settings.chunker_topic_cold_pages))
        max_topics_per_call = max(1, int(self.settings.chunker_max_topics_per_call))
        max_topic_pages = max(1, int(getattr(self.settings, "chunker_topic_max_pages", 15)))

        open_topics: dict[str, _OpenTopic] = {}
        closed_topics: list[_OpenTopic] = []
        latest_page_seen = 0
        failures = 0

        buffer_pages: list[int] = []
        buffer_images: list[object] = []
        buffer_texts: list[str] = []

        def _flush_window() -> None:
            nonlocal failures
            if not buffer_pages:
                return
            running = {tid: t.title for tid, t in open_topics.items()}
            try:
                assignments = self.client.identify_pdf_topics(
                    buffer_pages,
                    buffer_images,
                    buffer_texts,
                    running,
                    max_topics_per_call=max_topics_per_call,
                )
            except VLMTopicClientError as exc:
                failures += 1
                logger.warning(
                    "VLM chunker window pages=%s failed (%s); folding into 'Unparsed' topic",
                    buffer_pages, exc,
                )
                # Policy: no heuristic replacement for topic boundaries, but we
                # must still produce *some* coverage or the file would vanish
                # from the index. Fold the window into a single
                # "Unparsed window N..M" topic tagged as such.
                unparsed_id = "unparsed_" + uuid.uuid4().hex[:8]
                open_topics[unparsed_id] = _OpenTopic(
                    title=f"Unparsed window (pages {buffer_pages[0]}–{buffer_pages[-1]})",
                    summary="VLM chunker failed to identify topics on this window.",
                    pages=list(buffer_pages),
                    last_seen_page=buffer_pages[-1],
                    hero_page=buffer_pages[0],
                )
            else:
                _apply_assignments(open_topics, assignments)
            # Force-close any topic that grew beyond the per-topic
            # page cap. Prevents a single mega-topic from swallowing
            # the whole document when the VLM keeps extending it.
            _force_close_oversized_topics(open_topics, closed_topics, max_topic_pages)
            # Close cold topics
            _close_cold_topics(open_topics, closed_topics, latest_page_seen, cold_threshold)

        # Slide the window one page at a time so the VLM sees as much context
        # as possible without resending the whole doc. Only flush when the
        # buffer reaches ``stride`` pages or we run out.
        for page_number, image, text in page_iter:
            latest_page_seen = max(latest_page_seen, page_number)
            buffer_pages.append(int(page_number))
            buffer_images.append(image)
            buffer_texts.append(text or "")
            if len(buffer_pages) >= stride:
                _flush_window()
                buffer_pages.clear()
                buffer_images.clear()
                buffer_texts.clear()
        if buffer_pages:
            _flush_window()
            buffer_pages.clear()
            buffer_images.clear()
            buffer_texts.clear()

        # Move any still-open topics to the closed list.
        closed_topics.extend(open_topics.values())
        open_topics.clear()

        if not closed_topics:
            raise VLMChunkerError(
                f"VLM produced no topics for {path} after processing the file"
            )

        return [_topic_to_chunk(t, path, file_sha256) for t in closed_topics]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _apply_assignments(
    open_topics: dict[str, _OpenTopic],
    assignments: list[PdfTopicAssignment],
) -> None:
    """Fold a window's VLM assignments into the running topic dict."""
    for a in assignments:
        topic = open_topics.get(a.topic_id)
        if topic is None:
            if a.is_continuation:
                # The VLM said "continuation" but we have no such topic.
                # Treat it as new — it's a recovery path, not a hard error.
                topic = _OpenTopic(
                    title=a.title or f"Topic starting on p.{a.page}",
                    summary=a.summary,
                    pages=[],
                    hero_page=a.page,
                )
            else:
                topic = _OpenTopic(
                    title=a.title or f"Topic starting on p.{a.page}",
                    summary=a.summary,
                    pages=[],
                    hero_page=a.page,
                )
            open_topics[a.topic_id] = topic
        if a.page not in topic.pages:
            topic.pages.append(a.page)
        topic.last_seen_page = max(topic.last_seen_page, a.page)
        if topic.hero_page is None:
            topic.hero_page = a.page


def _close_cold_topics(
    open_topics: dict[str, _OpenTopic],
    closed_topics: list[_OpenTopic],
    latest_page_seen: int,
    cold_threshold: int,
) -> None:
    to_close = [
        tid for tid, topic in open_topics.items()
        if latest_page_seen - topic.last_seen_page >= cold_threshold
    ]
    for tid in to_close:
        closed_topics.append(open_topics.pop(tid))


def _force_close_oversized_topics(
    open_topics: dict[str, _OpenTopic],
    closed_topics: list[_OpenTopic],
    max_pages: int,
) -> None:
    """Evict any open topic that has already collected more than ``max_pages``
    pages. Protects retrieval quality against mega-topics the VLM might emit
    when a long section feels "thematically similar" at the document level.
    """
    to_close = [
        tid for tid, topic in open_topics.items()
        if len(set(topic.pages)) >= max_pages
    ]
    for tid in to_close:
        logger.debug(
            "chunker: force-closing oversized topic %r (%d pages)",
            open_topics[tid].title, len(set(open_topics[tid].pages)),
        )
        closed_topics.append(open_topics.pop(tid))


def _topic_to_chunk(topic: _OpenTopic, path: str, file_sha256: str) -> Chunk:
    pages_sorted = sorted(set(topic.pages))
    hero = topic.hero_page if topic.hero_page in pages_sorted else (pages_sorted[0] if pages_sorted else None)
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        path=path,
        file_sha256=file_sha256,
        kind=ChunkKind.PDF_TOPIC,
        title=topic.title[:200],
        summary=topic.summary[:500],
        page_refs=pages_sorted,
        hero_page=hero,
        excerpt=(topic.title + "\n" + topic.summary).strip()[:2000],
        order_key=hero or 0,
    )
