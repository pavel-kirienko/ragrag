"""Topic-based ingest pipeline.

Phase B rewrite. Each file is processed in a two-phase pipeline:

  1. **Plan** — the VLM topic chunker (PDFs) or text topic segmenter
     (text files) walks the source and emits a list of ``Chunk`` objects.
     Image files bypass the VLM and emit one chunk per file.
  2. **Embed + upsert** — each chunk gets two vector-store points (one
     text-modality, one image-modality for PDFs) that share the same
     ``chunk_id``. Visual multivectors are built from concatenated
     per-page ColQwen3 embeddings; text multivectors from the
     concatenated native text of every referenced range.

Memory discipline: PDF page images are re-rendered in the embed phase
rather than kept in memory across the plan phase. Two passes over the
PDF cost a few tens of milliseconds per page — negligible next to the
VLM and embedder forward time.
"""
from __future__ import annotations

import importlib
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, cast

import numpy as np
from PIL import Image

from ragrag.config import Settings
from ragrag.embedding.colqwen_embedder import ColQwenEmbedder
from ragrag.extractors.text_topic_segmenter import TextSegmenterError, TextTopicSegmenter
from ragrag.extractors.vlm_topic_chunker import VLMChunkerError, VLMTopicChunker
from ragrag.extractors.vlm_topic_client import VLMTopicClient
from ragrag.file_state import FileStateTracker
from ragrag.index.page_cache import PageImageCache
from ragrag.index.qdrant_store import QdrantStore
from ragrag.models import (
    Chunk,
    ChunkKind,
    FileType,
    IndexingStats,
    SkippedFile,
    get_file_type,
)
from ragrag.path_discovery import discover_files


VLMFactory = Callable[[], VLMTopicClient]


logger = logging.getLogger(__name__)


# Modality strings stored on each vector-store point payload.
MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"


class IngestManager:
    """Walk a set of paths and update the vector store with topic chunks.

    The caller owns the ``embedder`` (ColQwen3) and the optional
    ``vlm_client`` (Qwen2.5-VL-3B). When ``vlm_client`` is None, any file
    that would require VLM chunking (PDF or text file) causes a skip
    with a clear error — this is the "no heuristic chunking without the
    VLM" policy. The daemon configures both in :class:`EngineCache` so
    production ingest always has a VLM available.
    """

    def __init__(
        self,
        embedder: ColQwenEmbedder,
        store: QdrantStore,
        settings: Settings,
        vlm_client: Optional[VLMTopicClient] = None,
        vlm_factory: Optional[VLMFactory] = None,
    ) -> None:
        """One of ``vlm_client`` (already loaded, kept across calls) or
        ``vlm_factory`` (called lazily each ``ingest_paths``) should be
        provided; tests typically pass ``vlm_client`` with a stub, while the
        daemon passes ``vlm_factory`` so the heavy model only sits in VRAM
        for the duration of an actual indexing pass.
        """
        self.embedder = embedder
        self.store = store
        self.settings = settings
        self.vlm_client = vlm_client
        self._vlm_factory = vlm_factory
        self.file_tracker = FileStateTracker(settings.index_path)
        # PDF pages rendered during the embed phase are mirrored to this cache
        # so the search path can attach them to ``SearchResult.context_pages``.
        import os as _os
        self.page_cache = PageImageCache(
            _os.path.join(settings.index_path, "page_cache"),
            max_mb=int(getattr(settings, "page_cache_max_mb", 1024)),
        )

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def ingest_paths(
        self, paths: list[str]
    ) -> tuple[IndexingStats, list[SkippedFile], list[str]]:
        stats = IndexingStats()
        per_file_skipped: list[SkippedFile] = []

        file_paths, discovery_skipped = discover_files(paths, self.settings)

        # Peek at the files: if any are stale, we need a VLM client. Load
        # one lazily from the factory (if one is registered) and unload it
        # at the end of ``ingest_paths``. This keeps VRAM free for search
        # when indexing is not in progress.
        need_vlm = any(self.file_tracker.check_staleness(p)[0] for p in file_paths)
        loaded_vlm_client: VLMTopicClient | None = None
        if (
            need_vlm
            and self.vlm_client is None
            and self._vlm_factory is not None
        ):
            try:
                logger.info("Loading VLM topic client for indexing pass ...")
                loaded_vlm_client = self._vlm_factory()
                self.vlm_client = loaded_vlm_client
            except Exception as exc:
                logger.warning("VLM topic client load failed: %s", exc)

        t_start = time.time()
        try:
            result = self._ingest_loop(
                file_paths, stats, per_file_skipped, t_start,
            )
        finally:
            if loaded_vlm_client is not None:
                self.vlm_client = None
                try:
                    handle = getattr(loaded_vlm_client, "handle", None)
                    if handle is not None and hasattr(handle, "unload"):
                        handle.unload()
                except Exception as exc:
                    logger.warning("VLM unload failed: %s", exc)

        return (
            result[0], result[1] + discovery_skipped + result[2], result[3],
        )

    def _ingest_loop(
        self,
        file_paths: list[str],
        stats: IndexingStats,
        per_file_skipped: list[SkippedFile],
        t_start: float,
    ) -> tuple[IndexingStats, list[SkippedFile], list[SkippedFile], list[str]]:
        inner_skipped: list[SkippedFile] = []
        for idx, file_path in enumerate(file_paths):
            if time.time() - t_start > self.settings.indexing_timeout:
                inner_skipped.extend(
                    SkippedFile(path=fp, reason="indexing timeout")
                    for fp in file_paths[idx:]
                )
                break
            try:
                existing_point_ids = self.file_tracker.get_point_ids(file_path)
                needs_reindex, current_state = self.file_tracker.check_staleness(file_path)

                if not needs_reindex:
                    stats.files_skipped_unchanged += 1
                    continue

                logger.info("Indexing %s (%d/%d)", file_path, idx + 1, len(file_paths))
                was_previously_indexed = len(existing_point_ids) > 0
                if was_previously_indexed:
                    self.store.delete_by_ids(existing_point_ids)

                file_type = get_file_type(file_path)
                if file_type is None:
                    raise ValueError("unsupported file type")

                file_sha256 = current_state.content_hash_sha256
                chunks = self._plan_chunks(file_path, file_type, file_sha256)
                point_ids = self._embed_and_store(file_path, file_type, chunks)

                self.file_tracker.mark_indexed(
                    file_path, point_ids, file_state=current_state,
                )

                if was_previously_indexed:
                    stats.files_updated += 1
                else:
                    stats.files_added += 1

            except Exception as exc:
                logger.warning("Ingest error on %s: %s", file_path, exc)
                inner_skipped.append(
                    SkippedFile(path=file_path, reason=f"ingest error: {exc}")
                )

        if stats.files_added == 0 and stats.files_updated == 0:
            logger.debug(
                "Index up to date: %d files unchanged, %d added, %d updated",
                stats.files_skipped_unchanged, stats.files_added, stats.files_updated,
            )
        else:
            logger.info(
                "Index up to date: %d files unchanged, %d added, %d updated",
                stats.files_skipped_unchanged, stats.files_added, stats.files_updated,
            )
        return stats, per_file_skipped, inner_skipped, file_paths

    # ------------------------------------------------------------------ #
    # Plan phase
    # ------------------------------------------------------------------ #

    def _plan_chunks(
        self, file_path: str, file_type: FileType, file_sha256: str
    ) -> list[Chunk]:
        if file_type == FileType.PDF:
            return self._plan_pdf(file_path, file_sha256)
        if file_type == FileType.TEXT:
            return self._plan_text(file_path, file_sha256)
        if file_type == FileType.IMAGE:
            return self._plan_image(file_path, file_sha256)
        raise ValueError(f"unsupported file type: {file_type}")

    def _plan_pdf(self, file_path: str, file_sha256: str) -> list[Chunk]:
        if self.vlm_client is None:
            raise RuntimeError(
                "PDF indexing requires a VLM client (no heuristic chunking policy)"
            )
        pdf_module = importlib.import_module("ragrag.extractors.pdf_extractor")
        iter_pdf = cast(
            Callable[[str, Settings], Iterable],
            getattr(pdf_module, "iter_pdf_segments"),
        )

        def _page_stream() -> Iterator[tuple[int, object, str]]:
            # iter_pdf_segments yields (Segment, PIL.Image). We pull the
            # image segment for each page + its native text excerpt.
            current_page: int | None = None
            current_text: str = ""
            current_image: object | None = None
            for segment, image in iter_pdf(file_path, self.settings):
                if segment.modality.value == "image":
                    if current_page is not None and current_image is not None:
                        yield (current_page, current_image, current_text)
                    current_page = segment.page or 0
                    current_image = image
                    current_text = segment.excerpt or ""
                # text-modality segments contribute to the current page's text
                elif current_page == segment.page:
                    current_text = (current_text + "\n" + (segment.excerpt or "")).strip()
            if current_page is not None and current_image is not None:
                yield (current_page, current_image, current_text)

        chunker = VLMTopicChunker(self.vlm_client, self.settings)
        return chunker.chunk(str(Path(file_path).resolve()), file_sha256, _page_stream())

    def _plan_text(self, file_path: str, file_sha256: str) -> list[Chunk]:
        if self.vlm_client is None:
            raise RuntimeError(
                "Text file indexing requires a VLM client (no heuristic chunking policy)"
            )
        segmenter = TextTopicSegmenter(self.vlm_client, self.settings)
        try:
            return segmenter.segment(file_path)
        except TextSegmenterError as exc:
            raise RuntimeError(f"text segmenter failed: {exc}") from exc

    def _plan_image(self, file_path: str, file_sha256: str) -> list[Chunk]:
        """Standalone image files map 1:1 to a single chunk."""
        import os

        resolved = str(Path(file_path).resolve())
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                path=resolved,
                file_sha256=file_sha256,
                kind=ChunkKind.IMAGE,
                title=os.path.basename(file_path),
                summary="standalone image",
                excerpt=os.path.basename(file_path),
                order_key=0,
            )
        ]

    # ------------------------------------------------------------------ #
    # Embed + upsert phase
    # ------------------------------------------------------------------ #

    def _embed_and_store(
        self, file_path: str, file_type: FileType, chunks: list[Chunk]
    ) -> list[str]:
        if not chunks:
            return []

        # For PDFs we need to rebuild per-page images the chunker already
        # discarded. Build a page→image map once and feed each chunk's
        # page_refs out of it. For text / image files we go through the
        # relevant helpers directly.
        page_images: dict[int, Image.Image] = {}
        page_texts: dict[int, str] = {}

        if file_type == FileType.PDF:
            pdf_module = importlib.import_module("ragrag.extractors.pdf_extractor")
            iter_pdf = cast(
                Callable[[str, Settings], Iterable],
                getattr(pdf_module, "iter_pdf_segments"),
            )
            needed_pages: set[int] = set()
            for ch in chunks:
                needed_pages.update(ch.page_refs)
            file_sha = chunks[0].file_sha256 if chunks else ""
            for segment, image in iter_pdf(file_path, self.settings):
                page_num = segment.page or 0
                if page_num in needed_pages and page_num not in page_images and image is not None:
                    page_images[page_num] = image
                    if file_sha:
                        try:
                            self.page_cache.put(file_sha, page_num, image)
                        except Exception as exc:
                            logger.warning(
                                "Page cache put failed for %s p%d: %s",
                                file_path, page_num, exc,
                            )
                # Accumulate text across text-segment fragments per page
                if page_num in needed_pages:
                    existing = page_texts.get(page_num, "")
                    addition = segment.excerpt or ""
                    if addition:
                        page_texts[page_num] = (existing + "\n" + addition).strip() if existing else addition

        point_ids: list[str] = []
        for chunk in chunks:
            point_ids.extend(self._embed_one_chunk(chunk, file_type, page_images, page_texts, file_path))
        return point_ids

    def _embed_one_chunk(
        self,
        chunk: Chunk,
        file_type: FileType,
        page_images: dict[int, Image.Image],
        page_texts: dict[int, str],
        file_path: str,
    ) -> list[str]:
        """Produce 1 or 2 vector-store points for one chunk (modality-split)."""
        point_ids: list[str] = []

        # ---- Text point --------------------------------------------------
        text_payload_id = str(uuid.uuid4())
        text_content = self._chunk_text_content(chunk, file_type, page_texts, file_path)
        if text_content.strip():
            try:
                vec = self.embedder.embed_text_chunks([text_content])[0]
            except Exception as exc:
                logger.warning("Text embed failed for chunk %s: %s", chunk.chunk_id, exc)
                vec = None
            if vec is not None:
                self.store.upsert_many(
                    [(self._materialize_point(chunk, MODALITY_TEXT, text_payload_id), vec)]
                )
                point_ids.append(text_payload_id)

        # ---- Image point -------------------------------------------------
        image_payload_id = str(uuid.uuid4())
        visual_multivector = self._build_visual_multivector(chunk, file_type, page_images, file_path)
        if visual_multivector is not None and visual_multivector.shape[0] > 0:
            self.store.upsert_many(
                [(self._materialize_point(chunk, MODALITY_IMAGE, image_payload_id), visual_multivector)]
            )
            point_ids.append(image_payload_id)

        return point_ids

    def _chunk_text_content(
        self,
        chunk: Chunk,
        file_type: FileType,
        page_texts: dict[int, str],
        file_path: str,
    ) -> str:
        """Assemble the text that represents the chunk's textual body."""
        if file_type == FileType.PDF:
            parts = []
            for page in chunk.page_refs:
                text = page_texts.get(page, "")
                if text:
                    parts.append(f"[page {page}]\n{text}")
            header = f"{chunk.title}\n{chunk.summary}".strip()
            if header:
                parts.insert(0, header)
            return "\n\n".join(parts)
        if file_type == FileType.TEXT:
            # Read the byte_ranges out of the source file.
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
            except OSError:
                return f"{chunk.title}\n{chunk.summary}"
            parts = []
            if chunk.byte_ranges:
                for start, end in chunk.byte_ranges:
                    parts.append(data[start:end].decode("utf-8", errors="replace"))
            else:
                parts.append(data.decode("utf-8", errors="replace"))
            header = f"{chunk.title}\n{chunk.summary}".strip()
            if header:
                parts.insert(0, header)
            return "\n\n".join(parts)
        if file_type == FileType.IMAGE:
            return f"{chunk.title}\n{chunk.summary}".strip() or chunk.title
        return chunk.excerpt

    def _build_visual_multivector(
        self,
        chunk: Chunk,
        file_type: FileType,
        page_images: dict[int, Image.Image],
        file_path: str,
    ) -> Optional[np.ndarray]:
        """Concat the per-reference visual embeddings for the chunk."""
        images: list[Image.Image] = []
        if file_type == FileType.PDF:
            for page in chunk.page_refs:
                img = page_images.get(page)
                if img is not None:
                    images.append(img)
        elif file_type == FileType.IMAGE:
            try:
                images.append(Image.open(file_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Could not open image file %s: %s", file_path, exc)
                return None
        elif file_type == FileType.TEXT:
            return None  # text files have no visual modality

        if not images:
            return None

        parts: list[np.ndarray] = []
        for img in images:
            try:
                vec = self.embedder.embed_image(img)
            except Exception as exc:
                logger.warning("Image embed failed for chunk %s: %s", chunk.chunk_id, exc)
                continue
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                parts.append(arr)
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    def _materialize_point(self, chunk: Chunk, modality: str, point_id: str):
        """Turn a Chunk + modality + point_id into the store's Segment-shaped row.

        The store calls ``segment.model_dump()`` for the payload. We can
        use either ``Segment`` or a pydantic model that carries the chunk
        fields. For backwards compatibility with existing store rows, we
        piggyback on ``Segment`` with ``start_line``/``end_line`` left
        blank and cram the chunk payload into the ``excerpt`` field …

        Actually, cleaner: build a minimal ``_PointRow`` model below that
        is duck-compatible with the store's expectations (has
        ``segment_id`` and ``model_dump()``).
        """
        return _PointRow(
            segment_id=point_id,
            path=chunk.path,
            file_type=_file_type_from_kind(chunk.kind),
            modality=modality,
            page=chunk.hero_page if chunk.page_refs else None,
            start_line=chunk.line_ranges[0][0] if chunk.line_ranges else None,
            end_line=chunk.line_ranges[0][1] if chunk.line_ranges else None,
            excerpt=chunk.excerpt,
            chunk_id=chunk.chunk_id,
            file_sha256=chunk.file_sha256,
            kind=chunk.kind.value,
            title=chunk.title,
            summary=chunk.summary,
            page_refs=list(chunk.page_refs),
            line_ranges=[list(r) for r in chunk.line_ranges],
            byte_ranges=[list(r) for r in chunk.byte_ranges],
            order_key=chunk.order_key,
            hero_page=chunk.hero_page,
        )


# --------------------------------------------------------------------------- #
# Point row — duck-type for the store, replaces the legacy Segment row.
# --------------------------------------------------------------------------- #

from pydantic import BaseModel
from typing import Any as _Any


class _PointRow(BaseModel):
    """Serialized form of a Chunk+modality point in the vector store.

    The mmap store only calls ``segment.segment_id`` and
    ``segment.model_dump()``, so any pydantic model with those works.
    Keeping the legacy top-level fields means older payload-reading code
    (``search_engine.py``) still works unchanged while the new fields
    (``chunk_id``, ``title``, ``page_refs``, ...) land for Phase C.
    """

    segment_id: str
    path: str
    file_type: str
    modality: str
    page: _Any = None
    start_line: _Any = None
    end_line: _Any = None
    excerpt: str

    # New-in-Phase-B fields.
    chunk_id: str
    file_sha256: str
    kind: str
    title: str
    summary: str = ""
    page_refs: list[int] = []
    line_ranges: list[list[int]] = []
    byte_ranges: list[list[int]] = []
    order_key: int = 0
    hero_page: _Any = None


def _file_type_from_kind(kind: ChunkKind) -> str:
    if kind == ChunkKind.PDF_TOPIC:
        return "pdf"
    if kind == ChunkKind.TEXT_TOPIC:
        return "text"
    return "image"
