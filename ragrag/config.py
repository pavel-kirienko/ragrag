"""Configuration management for ragrag.

Settings are loaded from ragrag.json or .ragrag.json in the current
working directory (first found wins). All fields are optional with defaults.
"""
from __future__ import annotations

import json
import os
import sys

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Ragrag settings. Loaded from ragrag.json or .ragrag.json in CWD."""

    # Index storage
    index_path: str = Field(default=".ragrag", description="Directory for index storage.")

    # Embedding model
    model_id: str = Field(default="TomoroAI/tomoro-colqwen3-embed-4b", description="HuggingFace model ID.")
    max_visual_tokens: int = Field(default=16384, description="Max visual tokens per image.")
    quantization: str = Field(
        default="auto",
        description="GPU weight quantization: 'auto' (8-bit on CUDA), 'none', '8bit', '4bit'.",
    )

    # Search
    top_k: int = Field(default=10, description="Default number of results.")
    max_top_k: int = Field(default=50, description="Maximum allowed top_k.")

    # PDF extraction
    pdf_dpi: int = Field(default=250, description="DPI for PDF page rendering.")
    ocr_threshold: int = Field(default=50, description="Min chars before OCR fallback skipped.")

    # Text chunking
    chunk_size: int = Field(default=900, description="Target chunk size in characters.")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in characters.")

    # Embedding batching
    text_batch_size: int = Field(
        default=8, ge=1, le=64,
        description="Number of text chunks embedded per forward pass.",
    )
    embed_text_max_chars: int = Field(
        default=3200, ge=256, le=32768,
        description="Hard cap on the number of characters fed to the "
                    "text embedder per chunk. A topic spanning many "
                    "pages can exceed ColQwen3's activation budget on "
                    "tight GPUs (8 GB with bnb 4-bit weights); "
                    "truncating to ~800 tokens keeps retrieval stable.",
    )

    # VLM topic chunker (Phase B)
    vlm_model_id: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="HuggingFace model ID for the topic chunker and reranker VLM.",
    )
    vlm_quantization: str = Field(
        default="auto",
        description="VLM weight quantization: 'auto' (4-bit on CUDA, bf16 on CPU), 'none', '4bit', '8bit'.",
    )
    chunker_vlm_ctx_tokens: int = Field(
        default=8192, ge=512,
        description="Max tokens per VLM prompt for the text topic segmenter.",
    )
    chunker_stride_pages: int = Field(
        default=1, ge=1, le=32,
        description="Pages per VLM call for the PDF topic chunker. Smaller "
                    "values keep activation memory lower on tight GPUs: "
                    "stride=1 fits under an 8 GB ceiling even with a busy "
                    "X11 desktop holding ~3 GB. Larger values give the VLM "
                    "more cross-page context on bigger GPUs.",
    )
    chunker_vlm_image_max_side: int = Field(
        default=448, ge=256, le=2048,
        description="Max image side (pixels) fed to the VLM topic chunker. "
                    "The embedder always sees full-resolution pages (DPI "
                    "setting); the chunker only needs thumbnails to spot "
                    "topic boundaries. 448 px ~= 512 visual tokens per "
                    "image, which keeps vision-encoder activations under "
                    "~300 MiB on 8 GB GPUs.",
    )
    chunker_max_topics_per_call: int = Field(
        default=16, ge=1, le=64,
        description="Safety cap on new topics emitted per VLM call.",
    )
    chunker_topic_cold_pages: int = Field(
        default=20, ge=1,
        description="Close a topic after this many pages without any reference.",
    )

    # Location builder (Phase B)
    location_directory_listing_max: int = Field(
        default=64, ge=1,
        description="Max entries shown in Location.directory_listing; excess is head/tail split.",
    )
    location_respect_gitignore: bool = Field(
        default=True,
        description="Filter directory listings through the nearest .gitignore.",
    )

    # Page image cache (Phase C)
    page_cache_max_mb: int = Field(
        default=1024, ge=16,
        description="Soft cap in MiB for the rendered page-image cache; LRU eviction.",
    )
    include_page_images_default: str = Field(
        default="path",
        description="Default image delivery mode: 'none' | 'path' | 'base64'.",
    )

    # Filesystem
    include_hidden: bool = Field(default=False, description="Include hidden files/dirs.")
    follow_symlinks: bool = Field(default=True, description="Follow symbolic links.")

    # Timeouts
    indexing_timeout: float = Field(default=100000.0, description="Soft timeout for indexing in seconds.")

    # Daemon
    daemon_autostart: bool = Field(
        default=True,
        description="Auto-spawn the ragrag daemon from CLI calls when no socket is found.",
    )
    daemon_idle_timeout_s: float = Field(
        default=12 * 3600,
        description="Daemon exits after this many seconds of no requests. Default: 12 hours.",
    )
    daemon_status_host: str = Field(
        default="127.0.0.1",
        description="Bind host for the daemon's HTTP status server (Phase E).",
    )
    daemon_status_port: int = Field(
        default=27272,
        description="Initial port for the daemon's HTTP status server (Phase E). 0 = pick free.",
    )

    model_config = {"extra": "ignore"}


def _load_settings_from_config(config_path: str) -> Settings:
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    settings = Settings(**data)
    config_dir = os.path.dirname(config_path)
    index_path = settings.index_path
    if not os.path.isabs(index_path):
        index_path = os.path.join(config_dir, index_path)

    return settings.model_copy(update={"index_path": os.path.abspath(index_path)})


def _load_settings_in_dir(directory: str) -> Settings | None:
    for name in ("ragrag.json", ".ragrag.json"):
        config_path = os.path.join(directory, name)
        if os.path.isfile(config_path):
            try:
                return _load_settings_from_config(config_path)
            except Exception as e:
                print(f"Warning: Failed to load {name}: {e}", file=sys.stderr)
    return None


def find_index_root(start_dir: str | None = None) -> tuple[str, Settings]:
    current_dir = os.path.abspath(start_dir or os.getcwd())

    while True:
        index_path = os.path.join(current_dir, ".ragrag")
        if os.path.isdir(index_path):
            return current_dir, Settings(index_path=os.path.abspath(index_path))

        settings = _load_settings_in_dir(current_dir)
        if settings is not None:
            return current_dir, settings

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise SystemExit(
                "No ragrag index found. Create a config file (ragrag.json) "
                "or run with --new to create a new index here."
            )
        current_dir = parent_dir


def get_settings(start_dir: str | None = None) -> Settings:
    settings = _load_settings_in_dir(os.path.abspath(start_dir or os.getcwd()))
    if settings is not None:
        return settings
    return Settings()
