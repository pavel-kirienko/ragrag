"""Text extraction and chunking for ragrag.

Implements boundary-aware text chunking with configurable overlap.
Prefers splits at: blank lines > headings > sentence boundaries > word boundaries.
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from src.config import Settings
from src.models import FileType, Modality, Segment

logger = logging.getLogger(__name__)


def extract_text_segments(path: str, settings: Settings) -> list[Segment]:
    """Read a text file and split into chunked Segments.
    
    Args:
        path: Absolute path to the text file.
        settings: Settings object with chunk_size and chunk_overlap.
    
    Returns:
        List of Segment objects, one per chunk. Empty list if file is empty or unreadable.
    """
    try:
        # Read file with UTF-8, fallback to replacement for invalid chars
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return []
    
    if not content:
        return []

    chunks = _chunk_text(
        content,
        target_chars=settings.chunk_size,
        overlap_chars=settings.chunk_overlap,
    )
    
    # Convert chunks to Segments
    segments = []
    for chunk_text, start_line, end_line in chunks:
        segment = Segment(
            segment_id=str(uuid.uuid4()),
            path=str(Path(path).resolve()),
            file_type=FileType.TEXT,
            modality=Modality.TEXT,
            start_line=start_line,
            end_line=end_line,
            excerpt=chunk_text,
        )
        segments.append(segment)
    
    return segments


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
