"""Configuration management for ragrag.

All settings can be overridden via environment variables (case-insensitive).
Example: PDF_RENDER_DPI=150 python src/mcp_server.py
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Ragrag application settings.

    All values can be overridden via environment variables.
    List values use comma-separated format: ALLOWED_ROOTS=/home/a,/home/b
    """

    # Filesystem
    ALLOWED_ROOTS: Optional[list[str]] = Field(
        default=None,
        description="Allowed filesystem roots. None means no restriction.",
    )
    INCLUDE_HIDDEN_FILES: bool = Field(
        default=False,
        description="Whether to include hidden files/dirs (starting with '.').",
    )
    FOLLOW_SYMLINKS: bool = Field(
        default=False,
        description="Whether to follow symbolic links during directory walk.",
    )
    MAX_FILES_PER_REQUEST: int = Field(
        default=10000,
        description="Safety limit on files processed per request.",
    )

    # PDF extraction
    PDF_RENDER_DPI: int = Field(
        default=200,
        description="DPI for PDF page rendering. Lower = faster, higher = better quality.",
    )
    OCR_TEXT_THRESHOLD: int = Field(
        default=50,
        description="Min chars of native text before OCR fallback is skipped.",
    )

    # Text chunking
    CHUNK_TARGET_CHARS: int = Field(
        default=900,
        description="Target chunk size in characters (DESIGN.md: 600-1200).",
    )
    CHUNK_OVERLAP_CHARS: int = Field(
        default=100,
        description="Overlap between chunks in characters (DESIGN.md: 80-120).",
    )

    # Search
    TOP_K_DEFAULT: int = Field(
        default=10,
        description="Default number of results returned.",
    )
    TOP_K_MAX: int = Field(
        default=50,
        description="Maximum allowed top_k value.",
    )

    # Embedding model
    MODEL_ID: str = Field(
        default="TomoroAI/tomoro-colqwen3-embed-4b",
        description="HuggingFace model ID for embedding.",
    )
    MAX_VISUAL_TOKENS: int = Field(
        default=1280,
        description="Max visual tokens per image. Reduce for speed/memory.",
    )
    TEXT_EMBED_BATCH_SIZE: int = Field(
        default=4,
        description="Batch size for text embedding.",
    )
    IMAGE_EMBED_BATCH_SIZE: int = Field(
        default=1,
        description="Batch size for image embedding (keep at 1 on 16GB CPU).",
    )

    # Qdrant
    QDRANT_PATH: str = Field(
        default="./qdrant_data",
        description="Path for Qdrant on-disk local storage.",
    )
    QDRANT_COLLECTION: str = Field(
        default="ragrag_segments",
        description="Qdrant collection name.",
    )

    # Timeouts
    INDEXING_TIMEOUT_SECONDS: float = Field(
        default=600.0,
        description="Soft timeout for indexing phase in seconds.",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance.

    Call get_settings.cache_clear() to force reload (e.g. in tests).
    """
    return Settings()
