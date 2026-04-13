"""Text extraction and chunking for ragrag.

Implements boundary-aware text chunking with configurable overlap.
Prefers splits at: blank lines > headings > sentence boundaries > word boundaries.
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

from ragrag.config import Settings
from ragrag.models import FileType, Modality, Segment

logger = logging.getLogger(__name__)


def iter_text_segments(
    path: str, settings: Settings
) -> Iterator[tuple[Segment, Optional[Image.Image]]]:
    """Lazily yield text segments for a single file.

    Each yielded tuple is ``(segment, None)`` (text files have no images).
    The whole-file content is loaded once; chunks are yielded one at a time
    so the consumer can embed and upsert without buffering the whole file.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return

    if not content:
        return

    resolved = str(Path(path).resolve())
    for index, (chunk_text, start_line, end_line) in enumerate(
        _chunk_text(content, target_chars=settings.chunk_size, overlap_chars=settings.chunk_overlap)
    ):
        yield (
            Segment(
                segment_id=str(uuid.uuid4()),
                path=resolved,
                file_type=FileType.TEXT,
                modality=Modality.TEXT,
                start_line=start_line,
                end_line=end_line,
                excerpt=chunk_text,
            ),
            None,
        )


def extract_text_segments(path: str, settings: Settings) -> list[Segment]:
    """Read a text file and split into chunked Segments (eager wrapper).

    Backwards-compatible wrapper around :func:`iter_text_segments` for callers
    that want the full list at once. New code should prefer the iterator form.
    """
    return [seg for seg, _ in iter_text_segments(path, settings)]


def _chunk_text(
    content: str,
    target_chars: int,
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    """Split text into chunks with overlap, respecting boundaries.

    Returns (chunk_text, start_line, end_line) tuples with 1-indexed lines.
    """
    if not content:
        return []

    chunks: list[tuple[str, int, int]] = []
    pos = 0

    while pos < len(content):
        chunk_end = min(pos + target_chars, len(content))

        boundary_pos = _find_boundary(content, pos, chunk_end)
        if boundary_pos is None:
            boundary_pos = chunk_end

        chunk_text = content[pos:boundary_pos]
        start_line = content.count('\n', 0, pos) + 1
        end_line = start_line + chunk_text.count('\n')
        chunks.append((chunk_text, start_line, end_line))

        if boundary_pos >= len(content):
            break

        # Back up `overlap_chars` behind the boundary for the next window.
        # If that would land at or before the current start (which happens
        # when overlap >= chunk length, typical near end-of-file), drop the
        # overlap for this step and just advance past the emitted chunk —
        # that keeps iterations monotone and guarantees termination.
        next_pos = boundary_pos - overlap_chars
        if next_pos <= pos:
            next_pos = boundary_pos
        if next_pos >= len(content):
            break
        pos = next_pos

    return chunks


def _find_boundary(content: str, start: int, end: int) -> int | None:
    """Find a good split boundary between start and end positions.
    
    Prefers: blank lines > headings > sentence boundaries > word boundaries.
    
    Args:
        content: Full text content.
        start: Start position (inclusive).
        end: End position (exclusive).
    
    Returns:
        Position to split at, or None if no boundary found.
    """
    search_region = content[start:end]

    # In precedence order: blank line, markdown heading start, sentence end, word break.
    # For each, find the rightmost match and return the position right after it.
    for pattern in (r'\n\s*\n', r'\n#+\s', r'[.!?]\s+', r'\s+'):
        last_end = -1
        for m in re.finditer(pattern, search_region):
            last_end = m.end()
        if last_end > 0:
            return start + last_end

    return None
