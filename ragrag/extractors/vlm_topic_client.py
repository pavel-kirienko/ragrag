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

    def __init__(
        self,
        handle: _VLMHandleLike,
        *,
        max_retries: int = 3,
        image_max_side: int = 896,
        pdf_max_new_tokens: int = 384,
        text_max_new_tokens: int = 768,
    ) -> None:
        self.handle = handle
        self.max_retries = int(max_retries)
        self.image_max_side = int(image_max_side)
        self.pdf_max_new_tokens = int(pdf_max_new_tokens)
        self.text_max_new_tokens = int(text_max_new_tokens)

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

        if len(window_pages) == 1:
            prompt = self._build_pdf_prompt_single(
                window_pages[0], window_texts[0], running_topics, max_topics_per_call
            )
        else:
            prompt = self._build_pdf_prompt(window_pages, window_texts, running_topics, max_topics_per_call)

        # On CPU the vision encoder is prohibitively slow (a 3-image window
        # at 640 px is ~3000 image tokens × ~15 ms/token on bf16 CPU = ~45 s
        # of prefill *per window*), so we fall back to a text-only prompt
        # that trusts the native PDF text extracted by PyMuPDF. Datasheets
        # are text-dominated so this is accurate in practice; for scanned
        # or image-only PDFs the user should run on a CUDA host.
        handle_device = str(getattr(self.handle, "device", "") or "").lower()
        if handle_device == "cpu":
            images_for_prompt: list[Any] | None = None
        else:
            images_for_prompt = [
                _downscale_for_chunker(img, self.image_max_side) for img in window_images
            ]

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.handle.generate(
                    prompt if attempt == 1 else _terser_pdf_retry(prompt, attempt),
                    images=images_for_prompt,
                    max_new_tokens=self.pdf_max_new_tokens,
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
    def _build_pdf_prompt_single(
        page: int,
        text: str,
        running_topics: dict[str, str],
        max_topics_per_call: int,
    ) -> str:
        """Compact prompt used when the window contains exactly one page.

        The default stride is 1, so this is the hot path. We keep the
        schema minimal (single-level JSON, terse field names) so the VLM
        can answer in well under 256 tokens.
        """
        running = ", ".join(f"{tid}={title[:40]!r}" for tid, title in running_topics.items()) or "none"
        return (
            f"Page {page} of a technical document. Identify which topic(s) this page belongs to.\n"
            f"Open topics from earlier in the doc: {running}\n"
            f"Cap: {max_topics_per_call} new topics per call.\n\n"
            f'Respond with ONE line of JSON (no prose, no fences):\n'
            f'{{"topics":[{{"id":"tN","c":false,"t":"title","s":"summary"}}]}}\n'
            f'Fields: id unique; c=true means continuation of an existing id '
            f'(t/s optional then); t = title under 80 chars; s = summary under 150 chars.\n\n'
            f"Native text of page {page}:\n"
            f"{_truncate(text, 1200)}"
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
                    max_new_tokens=self.text_max_new_tokens,
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
    payload = _loads_with_salvage(blob)
    if not isinstance(payload, dict):
        raise ValueError("pdf topic response must be a JSON object")

    # Compact single-page schema: {"topics":[{"id":"tN","c":false,"t":"","s":""}]}
    if len(window_pages) == 1 and "topics" in payload and "pages" not in payload:
        return _parse_compact_pdf_response(payload, window_pages[0], max_topics)

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


def _parse_compact_pdf_response(
    payload: dict,
    page_num: int,
    max_topics: int,
) -> list[PdfTopicAssignment]:
    topics = payload.get("topics")
    if not isinstance(topics, list):
        raise ValueError("compact pdf topic response missing 'topics' list")
    new_titles: set[str] = set()
    assignments: list[PdfTopicAssignment] = []
    for t in topics:
        if not isinstance(t, dict):
            continue
        topic_id = t.get("id")
        if not isinstance(topic_id, str) or not topic_id:
            continue
        is_continuation = bool(t.get("c", False))
        title = str(t.get("t") or "").strip()
        summary = str(t.get("s") or "").strip()
        if not is_continuation and title and title not in new_titles:
            if len(new_titles) >= max_topics:
                continue
            new_titles.add(title)
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
        raise ValueError("compact pdf topic response had no usable topics")
    return assignments


def _loads_with_salvage(blob: str) -> Any:
    """Parse JSON, or salvage a truncated object/array by closing it.

    VLM generations sometimes hit ``max_new_tokens`` mid-object. Rather
    than retrying at higher cost we backtrack the blob to the last
    balanced point and append the closing brackets that the parser
    would have expected. If salvage still fails, we re-raise the
    original exception.
    """
    try:
        return json.loads(blob)
    except json.JSONDecodeError as exc:
        # Walk the blob tracking depth; cut at the deepest point where
        # we have a well-formed prefix, then close it.
        last_good: int | None = None
        depth_stack: list[str] = []
        in_str = False
        escape = False
        for i, ch in enumerate(blob):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in "[{":
                depth_stack.append(ch)
            elif ch in "]}":
                if depth_stack:
                    depth_stack.pop()
                if not depth_stack:
                    last_good = i + 1
            elif ch == "," and depth_stack:
                # Mark that we can safely truncate here and close the
                # remaining open scopes — a trailing comma is invalid
                # in JSON so we'll also need to strip it.
                last_good = i
        if last_good is None:
            raise exc
        fragment = blob[:last_good].rstrip().rstrip(",")
        # Close any still-open scopes.
        closing = []
        temp_stack = []
        in_str = False
        escape = False
        for ch in fragment:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in "[{":
                temp_stack.append(ch)
            elif ch in "]}":
                if temp_stack:
                    temp_stack.pop()
        while temp_stack:
            opener = temp_stack.pop()
            closing.append("]" if opener == "[" else "}")
        salvaged = fragment + "".join(closing)
        try:
            return json.loads(salvaged)
        except json.JSONDecodeError:
            raise exc


def _parse_text_topic_json(
    raw: str,
    content: str,
    absolute_line_offset: int,
) -> list[TextTopic]:
    blob = _extract_json_block(raw, "[")
    payload = _loads_with_salvage(blob)
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


def _downscale_for_chunker(image: Any, max_side: int) -> Any:
    """Return a resized copy of a PIL image with longest side <= ``max_side``.

    The topic chunker only needs to see layout and headings, not fine-grained
    datasheet figures, so we downscale aggressively to keep the VLM's vision
    encoder activation footprint small on tight GPUs. Non-PIL inputs (e.g.
    test stubs) are passed through untouched.
    """
    try:
        width, height = image.size  # PIL duck-typing
    except AttributeError:
        return image
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    try:
        from PIL import Image

        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = None
    try:
        if resample is not None:
            return image.resize(new_size, resample)
        return image.resize(new_size)
    except Exception:
        return image
