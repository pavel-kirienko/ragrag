"""Data models for ragrag.

All models are pure Pydantic — no ORM, no DB dependencies.
No circular imports: this file depends only on stdlib and pydantic.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

import importlib

try:
    _magic: Any = importlib.import_module("magic")
except ImportError as ex:
    raise RuntimeError(
        "python-magic with libmagic is required. Install with `pip install python-magic` and install system libmagic (e.g., `brew install libmagic` on macOS)."
    ) from ex


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

# A multivector is a 2D array of token-level embedding vectors with shape
# ``(n_tokens, embedding_dim)`` (embedding_dim is 320 for ColQwen3).
# Use numpy.ndarray for memory efficiency (4 bytes/float vs ~32 bytes for
# nested Python lists), but accept any sequence-of-sequence at API
# boundaries — qdrant-client's PointStruct accepts both.
MultiVector = Any


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

    Uses MIME-type detection via libmagic only.
    """
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
