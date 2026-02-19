"""MCP server exposing the semantic_search tool for ragrag.

Provides a FastMCP server with a single tool `semantic_search` that:
  1. Discovers and indexes files in the given paths
  2. Embeds the query using ColQwen3
  3. Retrieves top-k results from Qdrant
  4. Returns JSON (and optionally Markdown) search results
"""
from __future__ import annotations

from typing import cast
import anyio
import anyio.to_thread
from anyio import CapacityLimiter
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
from src.config import Settings, get_settings
from src.models import SearchRequest
from src.embedding.colqwen_embedder import ColQwenEmbedder
from src.index.qdrant_store import QdrantStore
from src.index.ingest_manager import IngestManager
from src.retrieval.search_engine import SearchEngine
from src.retrieval.result_formatter import format_as_json, format_as_markdown

# Global limiter — only 1 concurrent search (model is not thread-safe)
_limiter = CapacityLimiter(1)


@asynccontextmanager
async def lifespan(server):
    settings = get_settings()
    embedder = ColQwenEmbedder(settings.MODEL_ID, settings.MAX_VISUAL_TOKENS)
    store = QdrantStore(settings.QDRANT_PATH, settings.QDRANT_COLLECTION, embedder.embedding_dim)
    ingest_mgr = IngestManager(embedder, store, settings)
    engine = SearchEngine(embedder, store, ingest_mgr, settings)
    yield {"engine": engine, "settings": settings}


mcp = FastMCP("ragrag", lifespan=lifespan)


@mcp.tool()
async def semantic_search(
    paths: list[str],
    query: str,
    top_k: int = 10,
    include_markdown: bool = False,
) -> str:
    """Search indexed documents using semantic similarity.

    Args:
        paths: List of file paths or directories to search
        query: Natural language search query
        top_k: Number of results to return (1-50)
        include_markdown: If True, include formatted Markdown in response

    Returns:
        JSON string with search results, timing, and indexing stats
    """
    # Validate inputs
    if not paths:
        raise ValueError("paths must be a non-empty list")
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    # Get lifespan context
    ctx = mcp.get_context()
    lifespan_ctx = ctx.request_context.lifespan_context
    engine = cast(SearchEngine, lifespan_ctx["engine"])
    settings = cast(Settings, lifespan_ctx["settings"])

    # Clamp top_k to [1, settings.TOP_K_MAX]
    top_k = max(1, min(top_k, settings.TOP_K_MAX))

    # Build request
    request = SearchRequest(
        paths=paths,
        query=query,
        top_k=top_k,
        include_markdown=include_markdown,
    )

    # Run sync search in thread (model is not thread-safe → serialize with limiter)
    try:
        response = await anyio.to_thread.run_sync(
            lambda: engine.search(request),
            limiter=_limiter,
        )
    except Exception as exc:
        raise RuntimeError(f"Search failed: {exc}") from exc

    # Optionally compute Markdown
    if include_markdown:
        response.markdown = format_as_markdown(response)

    return format_as_json(response)


if __name__ == "__main__":
    mcp.run()
