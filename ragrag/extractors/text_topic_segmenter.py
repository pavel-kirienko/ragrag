"""VLM-driven topic segmenter for text files.

Same principle as :mod:`ragrag.extractors.vlm_topic_chunker` but for
plain text (source code, Markdown, config files). The segmenter reads the
file content, asks the VLM to partition it into logical topics, and
returns a list of :class:`ragrag.models.Chunk` objects.

Small files fit in a single VLM call. Larger files are fed in sliding
windows of ``chunker_vlm_ctx_tokens`` tokens with 25 % overlap, and topics
with matching titles across windows are coalesced into a single chunk.

Policy: no heuristic fallback for the topic map. On unparseable VLM
output the segmenter raises ``TextSegmenterError`` and the caller decides
what to do (typically skip the file with a warning).
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from ragrag.config import Settings
from ragrag.extractors.vlm_topic_client import (
    TextTopic,
    VLMTopicClient,
    VLMTopicClientError,
)
from ragrag.models import Chunk, ChunkKind


logger = logging.getLogger(__name__)


class TextSegmenterError(RuntimeError):
    """Raised when the VLM segmenter cannot produce any topics for a file."""


# Language hints used in the VLM prompt. Not an exhaustive list; unknown
# extensions default to "text file".
_LANGUAGE_HINTS: dict[str, str] = {
    ".py": "Python",
    ".c": "C",
    ".h": "C header",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++ header",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".md": "Markdown",
    ".rst": "reStructuredText",
    ".json": "JSON",
    ".yaml": "YAML", ".yml": "YAML",
    ".toml": "TOML",
    ".ini": "INI",
    ".cfg": "config",
}


class TextTopicSegmenter:
    """VLM-driven segmenter for plain text files."""

    def __init__(
        self,
        client: VLMTopicClient,
        settings: Settings,
    ) -> None:
        self.client = client
        self.settings = settings

    def segment(self, path: str) -> list[Chunk]:
        """Return chunks for ``path``. Reads the file as UTF-8.

        Raises :class:`TextSegmenterError` if the VLM fails to produce any
        topics. Returns an empty list if the file is empty.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as exc:  # pragma: no cover — unusual IO failures
            raise TextSegmenterError(f"could not read {path}: {exc}") from exc
        if not content.strip():
            return []

        file_sha256 = _compute_sha256(path)
        resolved = str(Path(path).resolve())
        language_hint = self._language_hint(path)

        # Cheap token estimate: 1 token ≈ 4 characters is close enough for
        # our budget check. The VLM tokenizer uses BPE; a real count would
        # require loading the processor, which we avoid here.
        approx_tokens = max(1, len(content) // 4)
        ctx_tokens = max(1, int(self.settings.chunker_vlm_ctx_tokens))

        if approx_tokens <= ctx_tokens:
            topics = self._segment_one_shot(content, language_hint)
        else:
            topics = self._segment_sliding_window(content, language_hint, ctx_tokens)

        if not topics:
            raise TextSegmenterError(f"VLM produced no topics for {path}")
        return [
            self._topic_to_chunk(topic, i, content, resolved, file_sha256)
            for i, topic in enumerate(topics)
        ]

    # ------------------------------------------------------------------ #

    def _segment_one_shot(self, content: str, language_hint: str) -> list[TextTopic]:
        try:
            return self.client.identify_text_topics(content, language_hint=language_hint)
        except VLMTopicClientError as exc:
            raise TextSegmenterError(str(exc)) from exc

    def _segment_sliding_window(
        self,
        content: str,
        language_hint: str,
        ctx_tokens: int,
    ) -> list[TextTopic]:
        lines = content.split("\n")
        # Window size in lines, estimated from the token budget.
        avg_line_len = max(1, sum(len(ln) for ln in lines[:100]) // max(1, min(100, len(lines))))
        lines_per_window = max(40, (ctx_tokens * 4) // max(1, avg_line_len))
        step = max(1, int(lines_per_window * 0.75))

        merged_topics: dict[str, TextTopic] = {}
        start = 0
        while start < len(lines):
            end = min(len(lines), start + lines_per_window)
            window_content = "\n".join(lines[start:end])
            try:
                topics = self.client.identify_text_topics(
                    window_content,
                    language_hint=language_hint,
                    absolute_line_offset=start,
                )
            except VLMTopicClientError as exc:
                raise TextSegmenterError(
                    f"sliding-window segmenter failed on lines {start + 1}–{end}: {exc}"
                ) from exc
            for topic in topics:
                key = _normalize_title(topic.title)
                existing = merged_topics.get(key)
                if existing is None:
                    merged_topics[key] = TextTopic(
                        title=topic.title,
                        summary=topic.summary,
                        ranges=list(topic.ranges),
                    )
                else:
                    existing.ranges.extend(topic.ranges)
                    if len(topic.summary) > len(existing.summary):
                        existing.summary = topic.summary
            if end >= len(lines):
                break
            start += step
        return list(merged_topics.values())

    def _topic_to_chunk(
        self,
        topic: TextTopic,
        order: int,
        content: str,
        path: str,
        file_sha256: str,
    ) -> Chunk:
        line_ranges = _dedupe_ranges(topic.ranges)
        byte_ranges = _line_ranges_to_byte_ranges(content, line_ranges)
        excerpt = _build_excerpt(content, line_ranges)
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            path=path,
            file_sha256=file_sha256,
            kind=ChunkKind.TEXT_TOPIC,
            title=topic.title[:200],
            summary=topic.summary[:500],
            line_ranges=line_ranges,
            byte_ranges=byte_ranges,
            excerpt=excerpt[:2000],
            order_key=order,
        )

    @staticmethod
    def _language_hint(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return _LANGUAGE_HINTS.get(ext, "text file")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _normalize_title(title: str) -> str:
    return " ".join(title.strip().lower().split())


def _dedupe_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_ranges = sorted((int(a), int(b)) for a, b in ranges if a <= b)
    merged: list[tuple[int, int]] = []
    for start, end in sorted_ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _line_offsets(content: str) -> list[int]:
    """Cumulative byte offset at the start of each line (1-indexed line N → offsets[N-1])."""
    offsets: list[int] = [0]
    for idx, ch in enumerate(content):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def _line_ranges_to_byte_ranges(
    content: str,
    line_ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not line_ranges:
        return []
    offsets = _line_offsets(content)
    total_lines = len(offsets)
    byte_ranges: list[tuple[int, int]] = []
    for start_line, end_line in line_ranges:
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines - 1, end_line - 1)
        start_byte = offsets[start_idx]
        if end_idx + 1 < total_lines:
            end_byte = offsets[end_idx + 1]
        else:
            end_byte = len(content)
        byte_ranges.append((start_byte, end_byte))
    return byte_ranges


def _build_excerpt(content: str, line_ranges: list[tuple[int, int]], max_chars: int = 1800) -> str:
    if not line_ranges:
        return ""
    lines = content.split("\n")
    chunks: list[str] = []
    budget = max_chars
    for start, end in line_ranges:
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        block = "\n".join(lines[start_idx:end_idx])
        if budget <= 0:
            break
        if len(block) > budget:
            block = block[:budget] + "..."
        chunks.append(block)
        budget -= len(block) + 4
    return "\n---\n".join(chunks)


def _compute_sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()
