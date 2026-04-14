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
# Segment (legacy compat shim) and Chunk (the new indexable unit)
# ---------------------------------------------------------------------------

class ChunkKind(str, Enum):
    """Kind of indexable unit. Determines how the chunk's references are
    interpreted at retrieval time and which page-cache lookups apply."""

    PDF_TOPIC = "pdf_topic"
    TEXT_TOPIC = "text_topic"
    IMAGE = "image"


class Chunk(BaseModel):
    """One topic, the new indexable unit.

    A chunk is a *semantic view* over its source file: it points at the
    pages or line ranges that belong to one topic, and it may overlap with
    other chunks (a single page can belong to multiple topics, a single
    text region can belong to multiple chunks). Chunk references need not
    be contiguous — a topic can span pages 1–3 and 15–17 if the VLM
    decided that's how the material is organised.

    A chunk is stored as **two points** in the vector store: one for the
    text modality and one for the visual modality. Both points carry the
    same ``chunk_id`` in their payload; search-time rollup dedupes on it
    so a topic appears at most once in the final top-k.
    """

    chunk_id: str                                   # UUID4
    path: str                                       # absolute path to the source file
    file_sha256: str
    kind: ChunkKind
    title: str                                      # VLM-supplied title or filename
    summary: str = ""                               # VLM-supplied 1-sentence summary

    # PDF topics ------------------------------------------------------- #
    page_refs: list[int] = Field(default_factory=list)
    hero_page: Optional[int] = None                 # representative page for reranker prompts

    # Text topics ------------------------------------------------------ #
    line_ranges: list[tuple[int, int]] = Field(default_factory=list)
    byte_ranges: list[tuple[int, int]] = Field(default_factory=list)

    # Common ----------------------------------------------------------- #
    excerpt: str = ""                               # short representative snippet for display
    order_key: int = 0                              # monotone, used for prev/next navigation


class Segment(BaseModel):
    """Legacy segment type. Kept as a compat shim for tests that still
    construct it directly. New code uses :class:`Chunk`."""

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


class PageContext(BaseModel):
    """One rendered PDF page attached to a search result."""

    page: int
    page_image_path: Optional[str] = None
    page_image_b64: Optional[str] = None
    text: str = ""


class Location(BaseModel):
    """Filesystem locator for a search-result hit.

    Replaces the related-document graph from earlier plan revisions: just
    tells the LLM consumer where the file lives and what else is in the
    same directory, with no attempt to be clever about cross-document
    relations.
    """

    path: str
    directory: str
    directory_listing: list[str] = Field(default_factory=list)
    listing_truncated: bool = False
    listing_total: int = 0


class SearchResult(BaseModel):
    rank: int
    score: float
    path: str
    file_type: str
    modality: str
    page: Optional[int] = None              # legacy: first PDF page if any
    start_line: Optional[int] = None        # legacy: first text line range start
    end_line: Optional[int] = None          # legacy: first text line range end
    excerpt: str

    # New in Phase B (all optional so the legacy test fixtures still parse) #
    chunk_id: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    page_refs: Optional[list[int]] = None
    line_ranges: Optional[list[tuple[int, int]]] = None
    context_pages: list[PageContext] = Field(default_factory=list)
    location: Optional[Location] = None
    rerank_reason: Optional[str] = None


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
