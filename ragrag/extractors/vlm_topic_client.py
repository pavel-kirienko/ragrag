"""High-level VLM wrapper for topic discovery.

This is the only place in the codebase that knows how to phrase a "split
this document into topics" prompt. The PDF chunker and the text segmenter
both funnel through here, so prompt tweaks happen in one spot.

The client is constructed with a ``VLMHandle`` (or a compatible stub for
tests). On errors it retries with progressively terser prompts, then
raises ``VLMTopicClientError`` — the caller decides whether to abort the
file or fall back. Phase B's policy is **no heuristic fallback for topic
discovery**: on failure we propagate up and let the outer loop skip the
file with a warning, rather than silently producing bad chunks.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol


logger = logging.getLogger(__name__)


class VLMTopicClientError(RuntimeError):
    """Raised when the VLM cannot produce a valid topic map for an input."""


# --------------------------------------------------------------------------- #
# Public data shapes
# --------------------------------------------------------------------------- #

@dataclass
class PdfTopicAssignment:
    """One VLM-returned assignment of a page to a topic.

    ``topic_id`` is an identifier produced by the VLM in the context of the
    current window. It is stable within a single ``identify_pdf_topics`` call
    but NOT across calls — the chunker is responsible for mapping new ids onto
    its running topic dictionary. ``title`` and ``summary`` are only set for
    freshly-introduced topics; continuations leave them empty.
    """

    page: int
    topic_id: str
    is_continuation: bool
    title: str = ""
    summary: str = ""


@dataclass
class TextTopic:
    """A topic identified inside a text file."""

    title: str
    summary: str
    ranges: list[tuple[int, int]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Handle protocol — lets tests use a stub without importing vlm_loader
# --------------------------------------------------------------------------- #

class _VLMHandleLike(Protocol):
    def generate(
        self,
        text: str,
        images: list[Any] | None = None,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str: ...


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class VLMTopicClient:
    """Narrow interface: produce topic assignments from a VLM."""

    def __init__(self, handle: _VLMHandleLike, *, max_retries: int = 3) -> None:
        self.handle = handle
        self.max_retries = int(max_retries)

    # -------- PDF path ------------------------------------------------- #

    def identify_pdf_topics(
        self,
        window_pages: list[int],
        window_images: list[Any],
        window_texts: list[str],
        running_topics: dict[str, str],
        *,
        max_topics_per_call: int = 16,
    ) -> list[PdfTopicAssignment]:
        """Ask the VLM which topics the pages in the window belong to.

        Args:
            window_pages: 1-indexed page numbers currently in the rolling window.
            window_images: PIL images, one per page in ``window_pages`` (same order).
            window_texts: native text extracted from each page (same order).
            running_topics: ``{topic_id: title}`` for topics currently open in
                the surrounding ingest state. Keeps the VLM aligned with the
                chunker's view of what's already been discovered.
            max_topics_per_call: safety cap on new topics emitted per call.
        """
        if not window_pages:
            return []
        if len(window_pages) != len(window_images) or len(window_pages) != len(window_texts):
            raise ValueError("window_pages / window_images / window_texts must be the same length")

        prompt = self._build_pdf_prompt(window_pages, window_texts, running_topics, max_topics_per_call)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.handle.generate(
                    prompt if attempt == 1 else _terser_pdf_retry(prompt, attempt),
                    images=window_images,
                    max_new_tokens=512,
                    temperature=0.0,
                )
                parsed = _parse_pdf_topic_json(raw, window_pages, max_topics_per_call)
                return parsed
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "VLM PDF topic parse failed on attempt %d/%d for pages %s-%s: %s",
                    attempt, self.max_retries, window_pages[0], window_pages[-1], exc,
                )
        raise VLMTopicClientError(
            f"VLM failed to produce valid PDF topic JSON after {self.max_retries} attempts "
            f"(pages {window_pages[0]}-{window_pages[-1]}): {last_error}"
        )

    @staticmethod
    def _build_pdf_prompt(
        pages: list[int],
        texts: list[str],
        running_topics: dict[str, str],
        max_topics_per_call: int,
    ) -> str:
        running = ", ".join(f'"{tid}": {title[:60]!r}' for tid, title in running_topics.items())
        if not running:
            running = "(none yet)"
        page_list = ", ".join(str(p) for p in pages)
        return f"""You are indexing a technical document by topic. You see {len(pages)} consecutive
pages (numbers: {page_list}) as images plus their native text excerpts.

Currently open topics from earlier in the document (id -> title):
  {running}

For EACH page, decide which topic(s) it belongs to. A page may belong to
multiple topics when its content spans two subjects. A topic may continue
from earlier (reuse its id above) or start fresh (invent a new id like
"t7" that is not already in the list above).

Constraints:
- At most {max_topics_per_call} NEW topics across the whole window.
- Each new topic needs a title (one line, under 80 chars) and a summary
  (one sentence, under 200 chars) describing what the topic covers.
- Continuations do NOT need new title/summary.

Return a single JSON object (no prose, no code fences) of the form:

{{
  "pages": [
    {{
      "page": <int>,
      "topics": [
        {{
          "id": "<string>",
          "is_continuation": <bool>,
          "title": "<string, only if new>",
          "summary": "<string, only if new>"
        }},
        ...
      ]
    }},
    ...
  ]
}}

Native text for each page:
""" + "\n".join(
            f"[page {p}] {_truncate(text, 600)}" for p, text in zip(pages, texts)
        )

    # -------- Text path ------------------------------------------------ #

    def identify_text_topics(
        self,
        content: str,
        *,
        language_hint: str = "text",
        absolute_line_offset: int = 0,
    ) -> list[TextTopic]:
        """Ask the VLM to partition a block of text into topic-level ranges.

        Returns topics with line ranges expressed in absolute line numbers
        (1-indexed), shifted by ``absolute_line_offset`` when the caller
        is feeding in a sliding window of a larger file.
        """
        if not content.strip():
            return []

        prompt = self._build_text_prompt(content, language_hint=language_hint)
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.handle.generate(
                    prompt if attempt == 1 else _terser_text_retry(prompt, attempt),
                    images=None,
                    max_new_tokens=768,
                    temperature=0.0,
                )
                topics = _parse_text_topic_json(raw, content, absolute_line_offset)
                return topics
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "VLM text topic parse failed on attempt %d/%d (%d bytes): %s",
                    attempt, self.max_retries, len(content), exc,
                )
        raise VLMTopicClientError(
            f"VLM failed to produce valid text topic JSON after {self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _build_text_prompt(content: str, *, language_hint: str) -> str:
        # Number the lines so the VLM has explicit targets to point at.
        lines = content.split("\n")
        numbered = "\n".join(f"{i + 1:6d}  {line}" for i, line in enumerate(lines))
        return f"""You are indexing a {language_hint} file by topic. Each line is numbered.

Identify the distinct topics in this file. A topic is a group of lines that
belong together semantically: a function, a class, a configuration section,
a paragraph block discussing one idea. Two distinct topics MAY reference
overlapping lines when the content genuinely belongs to both; usually they
do not overlap.

Return a single JSON array (no prose, no code fences) of the form:

[
  {{
    "title": "<string, under 80 chars>",
    "summary": "<one sentence, under 200 chars>",
    "ranges": [[line_start, line_end], ...]
  }},
  ...
]

Line numbers are 1-indexed and inclusive. Cover every significant line at
least once. A short, coherent file may return a single topic covering the
whole range.

File content (with line numbers):
{numbered}"""


# --------------------------------------------------------------------------- #
# Parsers + helpers
# --------------------------------------------------------------------------- #

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_block(text: str, opener: str) -> str:
    """Pull the outermost JSON blob out of a VLM response.

    VLMs sometimes prefix responses with ``Here's the JSON:`` or wrap them
    in ```json fences``. We locate the first balanced opener and return it.
    """
    stripped = text.strip()
    # Strip code fences if present.
    if stripped.startswith("```"):
        nl = stripped.find("\n")
        if nl >= 0:
            stripped = stripped[nl + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()

    if opener == "{":
        match = _JSON_OBJECT_RE.search(stripped)
    else:
        match = _JSON_ARRAY_RE.search(stripped)
    if not match:
        raise ValueError(f"no JSON {opener}...{'}' if opener == '{' else ']'} block found in VLM response")
    return match.group(0)


def _parse_pdf_topic_json(
    raw: str,
    window_pages: list[int],
    max_topics: int,
) -> list[PdfTopicAssignment]:
    blob = _extract_json_block(raw, "{")
    payload = json.loads(blob)
    if not isinstance(payload, dict):
        raise ValueError("pdf topic response must be a JSON object")
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("pdf topic response missing 'pages' list")

    window_set = set(window_pages)
    assignments: list[PdfTopicAssignment] = []
    new_topic_titles: set[str] = set()
    for entry in pages:
        if not isinstance(entry, dict):
            continue
        page_num = entry.get("page")
        topics = entry.get("topics")
        if not isinstance(page_num, int) or page_num not in window_set:
            continue
        if not isinstance(topics, list):
            continue
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            topic_id = topic.get("id")
            if not isinstance(topic_id, str) or not topic_id:
                continue
            is_continuation = bool(topic.get("is_continuation", False))
            title = str(topic.get("title") or "").strip()
            summary = str(topic.get("summary") or "").strip()
            if not is_continuation and title and title not in new_topic_titles:
                if len(new_topic_titles) >= max_topics:
                    logger.warning(
                        "VLM emitted more than %d new topics in one window; ignoring extras",
                        max_topics,
                    )
                    continue
                new_topic_titles.add(title)
            assignments.append(
                PdfTopicAssignment(
                    page=page_num,
                    topic_id=topic_id,
                    is_continuation=is_continuation,
                    title=title,
                    summary=summary,
                )
            )
    if not assignments:
        raise ValueError("pdf topic response contained no valid page/topic entries")
    return assignments


def _parse_text_topic_json(
    raw: str,
    content: str,
    absolute_line_offset: int,
) -> list[TextTopic]:
    blob = _extract_json_block(raw, "[")
    payload = json.loads(blob)
    if not isinstance(payload, list):
        raise ValueError("text topic response must be a JSON array")

    total_lines = content.count("\n") + 1
    result: list[TextTopic] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        ranges_raw = entry.get("ranges")
        if not title or not isinstance(ranges_raw, list):
            continue
        ranges: list[tuple[int, int]] = []
        for r in ranges_raw:
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                continue
            try:
                start = int(r[0])
                end = int(r[1])
            except (TypeError, ValueError):
                continue
            if start < 1 or end < start:
                continue
            # Clamp to the actual line count of the window content; shift by
            # the absolute offset when the VLM was fed a window.
            start = min(start, total_lines) + absolute_line_offset
            end = min(end, total_lines) + absolute_line_offset
            ranges.append((start, end))
        if ranges:
            result.append(TextTopic(title=title, summary=summary, ranges=ranges))
    if not result:
        raise ValueError("text topic response produced no valid topics")
    return result


def _terser_pdf_retry(original: str, attempt: int) -> str:
    return (
        "Your previous response could not be parsed. Reply with ONLY a JSON object, "
        "no prose, no markdown fences.\n\n" + original
    )


def _terser_text_retry(original: str, attempt: int) -> str:
    return (
        "Your previous response could not be parsed. Reply with ONLY a JSON array, "
        "no prose, no markdown fences.\n\n" + original
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
