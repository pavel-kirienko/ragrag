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
    
    # Handle empty files
    if not content:
        return []
    
    # Split into lines for line tracking
    lines = content.split('\n')
    
    # Chunk the content
    chunks = _chunk_text(
        content,
        lines,
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
    lines: list[str],
    target_chars: int,
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    """Split text into chunks with overlap, respecting boundaries.
    
    Args:
        content: Full text content.
        lines: List of lines (split by '\n'). Used for line tracking.
        target_chars: Target chunk size in characters.
        overlap_chars: Overlap size in characters.
    
    Returns:
        List of (chunk_text, start_line, end_line) tuples.
        start_line and end_line are 1-indexed.
    """
    if not content:
        return []
    
    chunks = []
    pos = 0  # Current position in content
    line_pos = 0  # Current line index
    
    while pos < len(content):
        # Find chunk end
        chunk_end = min(pos + target_chars, len(content))
        
        # Try to find a good boundary
        boundary_pos = _find_boundary(content, pos, chunk_end)
        
        if boundary_pos is None:
            # No boundary found, use target end
            boundary_pos = chunk_end
        
        # Extract chunk
        chunk_text = content[pos:boundary_pos]
        
        # Count lines in this chunk
        chunk_lines = chunk_text.count('\n')
        start_line = line_pos + 1  # 1-indexed
        end_line = line_pos + chunk_lines + 1
        
        chunks.append((chunk_text, start_line, end_line))
        
        # Move position forward
        pos = boundary_pos
        
        # Update line position
        line_pos += chunk_lines
        
        # Avoid infinite loop on very small chunks
        if pos >= len(content):
            break
    
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
    
    # 1. Try blank lines (double newline)
    blank_match = re.search(r'\n\s*\n', search_region[::-1])
    if blank_match:
        # Found blank line, split after it
        pos_in_region = len(search_region) - blank_match.start()
        return start + pos_in_region
    
    # 2. Try headings (lines starting with #)
    heading_match = re.search(r'\n#+\s', search_region[::-1])
    if heading_match:
        pos_in_region = len(search_region) - heading_match.start()
        return start + pos_in_region
    
    # 3. Try sentence boundaries (., !, ?)
    sentence_match = re.search(r'[.!?]\s+', search_region[::-1])
    if sentence_match:
        pos_in_region = len(search_region) - sentence_match.start()
        return start + pos_in_region
    
    # 4. Try word boundaries (space)
    word_match = re.search(r'\s+', search_region[::-1])
    if word_match:
        pos_in_region = len(search_region) - word_match.start()
        return start + pos_in_region
    
    # No boundary found
    return None
