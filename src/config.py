"""Configuration management for ragrag.

Settings are loaded from ragrag.json or .ragrag.json in the current
working directory (first found wins). All fields are optional with defaults.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Ragrag settings. Loaded from ragrag.json or .ragrag.json in CWD."""

    # Index storage
    index_path: str = Field(default=".ragrag", description="Directory for index storage.")

    # Embedding model
    model_id: str = Field(default="TomoroAI/tomoro-colqwen3-embed-4b", description="HuggingFace model ID.")
    max_visual_tokens: int = Field(default=1280, description="Max visual tokens per image.")

    # Search
    top_k: int = Field(default=10, description="Default number of results.")
    max_top_k: int = Field(default=50, description="Maximum allowed top_k.")

    # PDF extraction
    pdf_dpi: int = Field(default=200, description="DPI for PDF page rendering.")
    ocr_threshold: int = Field(default=50, description="Min chars before OCR fallback skipped.")

    # Text chunking
    chunk_size: int = Field(default=900, description="Target chunk size in characters.")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks in characters.")

    # Filesystem
    max_files: int = Field(default=10000, description="Max files per search request.")
    include_hidden: bool = Field(default=False, description="Include hidden files/dirs.")
    follow_symlinks: bool = Field(default=False, description="Follow symbolic links.")

    # Timeouts
    indexing_timeout: float = Field(default=600.0, description="Soft timeout for indexing in seconds.")

    model_config = {"extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance. Reads ragrag.json or .ragrag.json from CWD."""
    for name in ("ragrag.json", ".ragrag.json"):
        path = os.path.join(os.getcwd(), name)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                return Settings(**data)
            except Exception as e:
                print(f"Warning: Failed to load {name}: {e}", file=sys.stderr)
    return Settings()
