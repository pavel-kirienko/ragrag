"""Search engine for ragrag.

Orchestrates the full search pipeline:
  1. Ingest (index new/changed files)
  2. Embed query text → MultiVector
  3. Retrieve top-k results from Qdrant
  4. Format ScoredPoints → SearchResult list
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from ragrag.config import Settings
from ragrag.embedding.colqwen_embedder import ColQwenEmbedder
from ragrag.index.page_cache import PageImageCache
from ragrag.index.qdrant_store import QdrantStore
from ragrag.models import (
    IndexingStats,
    PageContext,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TimingInfo,
)
from ragrag.retrieval.location_builder import build_location

if TYPE_CHECKING:
    from ragrag.index.ingest_manager import IngestManager


def _build_search_result(
    rank: int,
    point,
    *,
    respect_gitignore: bool,
    max_listing: int,
    page_cache: PageImageCache | None = None,
    include_page_images: str = "path",
) -> SearchResult:
    """Project a Qdrant-like point onto a ``SearchResult``.

    Works for both the new Chunk-shaped payloads (with ``chunk_id``, ``title``,
    ``page_refs``, etc.) and the legacy Segment payloads from pre-Phase-B
    indexes. Attaches a ``Location`` block computed on the fly via
    ``build_location`` and — when ``page_cache`` is provided and the payload
    has ``page_refs`` — ``context_pages`` referencing the cached WebP files.
    """
    payload = point.payload
    path = payload["path"]
    page_refs = payload.get("page_refs") or None
    raw_line_ranges = payload.get("line_ranges") or None
    line_ranges: list[tuple[int, int]] | None = None
    if raw_line_ranges:
        line_ranges = [tuple(pair) for pair in raw_line_ranges if len(pair) == 2]
    # Prefer the hero_page on the payload; fall back to the legacy ``page``.
    hero_page = payload.get("hero_page") or payload.get("page")
    try:
        location = build_location(
            path, max_entries=max_listing, respect_gitignore=respect_gitignore,
        )
    except Exception:
        location = None

    context_pages: list[PageContext] = []
    if page_cache is not None and include_page_images != "none" and page_refs:
        file_sha = payload.get("file_sha256") or ""
        for page_num in page_refs:
            cached = page_cache.get(file_sha, page_num) if file_sha else None
            ctx = PageContext(
                page=int(page_num),
                page_image_path=str(cached) if cached is not None and include_page_images == "path" else None,
                page_image_b64=_encode_webp_b64(cached) if cached is not None and include_page_images == "base64" else None,
                text="",  # filled by the consumer if it needs per-page text
            )
            context_pages.append(ctx)

    return SearchResult(
        rank=rank,
        score=point.score,
        path=path,
        file_type=payload["file_type"],
        modality=payload["modality"],
        page=hero_page,
        start_line=line_ranges[0][0] if line_ranges else payload.get("start_line"),
        end_line=line_ranges[0][1] if line_ranges else payload.get("end_line"),
        excerpt=payload.get("excerpt") or payload.get("summary") or "",
        chunk_id=payload.get("chunk_id"),
        title=payload.get("title"),
        summary=payload.get("summary"),
        page_refs=page_refs,
        line_ranges=line_ranges,
        context_pages=context_pages,
        location=location,
    )


def _encode_webp_b64(path) -> str | None:
    """Return a base64-encoded WebP payload for a cached page image."""
    import base64

    if path is None:
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None


def _resolve_filter_paths(request_paths: list[str], indexed_paths: list[str]) -> list[str]:
    """Turn request paths (which may be directories) into the flat list of file
    paths used by Qdrant's MatchAny filter.

    Qdrant's MatchAny is exact-string on the payload, so a directory entry would
    never match the per-file `path` payload. We expand directories by intersecting
    with the concrete file paths returned by the ingest phase and fall back to
    exact matches for request paths that point at a single file.
    """
    resolved_requests = [os.path.realpath(os.path.abspath(p)) for p in request_paths]
    resolved_indexed = [os.path.realpath(os.path.abspath(p)) for p in indexed_paths]
    out: set[str] = set()
    for req in resolved_requests:
        if os.path.isfile(req):
            out.add(req)
            continue
        prefix = req.rstrip(os.sep) + os.sep
        for f in resolved_indexed:
            if f == req or f.startswith(prefix):
                out.add(f)
    return sorted(out)


class SearchEngine:
    """Synchronous search engine that coordinates indexing and retrieval."""

    def __init__(
        self,
        embedder: ColQwenEmbedder,
        store: QdrantStore,
        ingest_manager: IngestManager,
        settings: Settings,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.ingest_manager = ingest_manager
        self.settings = settings

    def search(self, request: SearchRequest) -> SearchResponse:
        """Execute full search pipeline: ingest → embed query → retrieve → format."""
        t_start = time.time()
        errors: list[str] = []

        # ------------------------------------------------------------------
        # Phase 1: Indexing
        # ------------------------------------------------------------------
        t0 = time.time()
        indexed_paths: list[str] = []
        try:
            stats, skipped, indexed_paths = self.ingest_manager.ingest_paths(request.paths)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Indexing error: {exc}")
            stats = IndexingStats()
            skipped = []
        indexing_ms = (time.time() - t0) * 1000

        # ------------------------------------------------------------------
        # Phase 2: Query embedding
        # ------------------------------------------------------------------
        t0 = time.time()
        try:
            query_vec = self.embedder.embed_query_text(request.query)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Query embedding error: {exc}")
            total_ms = (time.time() - t_start) * 1000
            return SearchResponse(
                query=request.query,
                status="partial",
                indexed_now=stats,
                skipped_files=skipped,
                errors=errors,
                results=[],
                timing_ms=TimingInfo(
                    indexing_ms=indexing_ms,
                    total_ms=total_ms,
                ),
            )
        query_embedding_ms = (time.time() - t0) * 1000
        logger.debug("Query embedding: %.1fms", query_embedding_ms)

        # ------------------------------------------------------------------
        # Phase 3: Retrieval
        # ------------------------------------------------------------------
        t0 = time.time()
        try:
            filter_paths = _resolve_filter_paths(request.paths, indexed_paths)
            scored_points = self.store.search(query_vec, top_k=request.top_k, path_filter=filter_paths)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Retrieval error: {exc}")
            scored_points = []
        retrieval_ms = (time.time() - t0) * 1000
        logger.debug("Retrieval: %d results in %.1fms", len(scored_points), retrieval_ms)

        # ------------------------------------------------------------------
        # Phase 4: Chunk-level rollup + format results
        # ------------------------------------------------------------------
        t0 = time.time()
        results: list[SearchResult] = []
        seen_chunk_ids: set[str] = set()
        for point in scored_points:
            chunk_id = point.payload.get("chunk_id")
            # Legacy segment payloads (pre-Phase-B) have no chunk_id; treat
            # each such point as its own chunk.
            if chunk_id and chunk_id in seen_chunk_ids:
                continue
            if chunk_id:
                seen_chunk_ids.add(chunk_id)
            try:
                results.append(
                    _build_search_result(
                        len(results) + 1,
                        point,
                        respect_gitignore=self.settings.location_respect_gitignore,
                        max_listing=self.settings.location_directory_listing_max,
                        page_cache=getattr(self.ingest_manager, "page_cache", None),
                        include_page_images=getattr(
                            request, "include_page_images", None,
                        ) or self.settings.include_page_images_default,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Result formatting error (rank {len(results) + 1}): {exc}")
            if len(results) >= request.top_k:
                break
        formatting_ms = (time.time() - t0) * 1000

        total_ms = (time.time() - t_start) * 1000

        return SearchResponse(
            query=request.query,
            status="complete" if not errors else "partial",
            indexed_now=stats,
            skipped_files=skipped,
            errors=errors,
            results=results,
            timing_ms=TimingInfo(
                indexing_ms=indexing_ms,
                query_embedding_ms=query_embedding_ms,
                retrieval_ms=retrieval_ms,
                formatting_ms=formatting_ms,
                total_ms=total_ms,
            ),
        )
