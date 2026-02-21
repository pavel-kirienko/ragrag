"""Data models for ragrag.

All models are pure Pydantic — no ORM, no DB dependencies.
No circular imports: this file depends only on stdlib and pydantic.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

import logging as _logging

_log = _logging.getLogger(__name__)
_magic: Any | None = None
try:
    import magic as _magic  # type: ignore[import-not-found]
except ImportError:
    _log.warning("python-magic (libmagic) not found, falling back to extension-based file detection")

_HAS_MAGIC = _magic is not None


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class FileType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"


# ---------------------------------------------------------------------------
# File-level state (staleness tracking — §4.3, §7.1)
# ---------------------------------------------------------------------------

class FileState(BaseModel):
    path: str
    size: int
    mtime_ns: int
    content_hash_sha256: str
    last_indexed_at: float
    point_ids: list[str]


# ---------------------------------------------------------------------------
# Segment (§7.2) — one indexable unit of content
# ---------------------------------------------------------------------------

class Segment(BaseModel):
    segment_id: str
    path: str
    file_type: FileType
    modality: Modality
    page: Optional[int] = None          # PDF page number (1-indexed), None for non-PDF
    start_line: Optional[int] = None    # For text segments
    end_line: Optional[int] = None      # For text segments
    excerpt: str                        # Human-readable snippet for result display


# ---------------------------------------------------------------------------
# Tool contract models (§5.2, §5.3)
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    paths: list[str] = Field(..., description="Files and/or directories to search")
    query: str = Field(..., description="Natural-language search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    include_markdown: bool = Field(default=False, description="Include Markdown summary")


class IndexingStats(BaseModel):
    files_added: int = 0
    files_updated: int = 0
    files_skipped_unchanged: int = 0


class SkippedFile(BaseModel):
    path: str
    reason: str


class SearchResult(BaseModel):
    rank: int
    score: float
    path: str
    file_type: str
    modality: str
    page: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    excerpt: str


class TimingInfo(BaseModel):
    discovery_ms: float = 0.0
    indexing_ms: float = 0.0
    query_embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    formatting_ms: float = 0.0
    total_ms: float = 0.0


class SearchResponse(BaseModel):
    query: str
    status: str                          # "complete" or "partial"
    indexed_now: IndexingStats
    skipped_files: list[SkippedFile] = []
    errors: list[str] = []
    results: list[SearchResult] = []
    markdown: Optional[str] = None
    timing_ms: TimingInfo


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A multivector is a list of token-level embedding vectors.
# Each inner list has length == embedding_dim (320 for ColQwen3).
MultiVector = list[list[float]]


# ---------------------------------------------------------------------------
# File extension mappings (§4.2)
# ---------------------------------------------------------------------------

TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".rst",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".py", ".js", ".ts",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
})

PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
})

SUPPORTED_EXTENSIONS: frozenset[str] = TEXT_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS


def get_file_type(path: str) -> Optional[FileType]:
    """Return the FileType for a given path, or None if unsupported.

    Uses MIME-type detection via libmagic when available; falls back to
    extension-based detection otherwise.
    """
    import os
    if _magic is not None:
        try:
            mime = _magic.from_file(path, mime=True)
        except Exception:
            mime = None
        if mime is None:
            return None
        if mime.startswith("text/"):
            return FileType.TEXT
        if mime == "application/pdf":
            return FileType.PDF
        if mime.startswith("image/"):
            return FileType.IMAGE
        if mime in ("application/json", "application/xml", "application/javascript", "application/x-yaml", "application/toml"):
            return FileType.TEXT
        # Skip: inode/x-empty (empty), inode/chardevice, application/octet-stream, executables
        return None
    else:
        # Fallback: extension-based
        ext = os.path.splitext(path)[1].lower()
        if ext in TEXT_EXTENSIONS:
            return FileType.TEXT
        if ext in PDF_EXTENSIONS:
            return FileType.PDF
        if ext in IMAGE_EXTENSIONS:
            return FileType.IMAGE
        return None
