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
from ragrag.index.qdrant_store import QdrantStore
from ragrag.models import (
    IndexingStats,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TimingInfo,
)

if TYPE_CHECKING:
    from ragrag.index.ingest_manager import IngestManager


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
        # Phase 4: Format results
        # ------------------------------------------------------------------
        t0 = time.time()
        results: list[SearchResult] = []
        for i, point in enumerate(scored_points):
            try:
                results.append(
                    SearchResult(
                        rank=i + 1,
                        score=point.score,
                        path=point.payload["path"],
                        file_type=point.payload["file_type"],
                        modality=point.payload["modality"],
                        page=point.payload.get("page"),
                        start_line=point.payload.get("start_line"),
                        end_line=point.payload.get("end_line"),
                        excerpt=point.payload["excerpt"],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Result formatting error (rank {i + 1}): {exc}")
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
