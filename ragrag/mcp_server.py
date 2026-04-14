"""Minimal MCP server for ragrag (Phase F).

Exposes a single ``search_documentation`` tool that wraps the
daemon's JSON-RPC ``search`` method (falling back to an in-process
``SearchEngine`` when the daemon is not available). The tool
serialises the result into a compact, LLM-friendly shape and
optionally embeds hero page images as base64 so the MCP client
can render them without filesystem access.

The ``mcp`` package is an optional extra; this module imports it
lazily so non-MCP users do not have to install it. Install with:

    pip install -e ".[mcp]"

Then run:

    ragrag mcp --index-path /path/to/project

The server speaks over stdio by default — the standard
Anthropic MCP transport for local tools.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve ragrag as an MCP tool over stdio.",
    )
    parser.add_argument(
        "--index-path",
        default=os.getcwd(),
        help="Index root (the directory containing .ragrag/ or a ragrag.json).",
    )
    parser.add_argument(
        "--default-top-k",
        type=int,
        default=5,
        help="Default number of hits to return when the client does not specify top_k.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("none", "path", "base64"),
        default="base64",
        help="How to deliver page images to the MCP client.",
    )
    parser.add_argument(
        "--max-inline-image-kb",
        type=int,
        default=4096,
        help="Hard cap on the total bytes of inlined base64 images per response.",
    )
    args = parser.parse_args(argv)

    try:
        import mcp  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "The 'mcp' optional extra is not installed. Install it with:\n"
            "    pip install -e '.[mcp]'\n"
        )
        return 2

    try:
        return asyncio.run(_serve(args))
    except KeyboardInterrupt:
        return 0


# --------------------------------------------------------------------------- #
# Server implementation
# --------------------------------------------------------------------------- #


async def _serve(args) -> int:
    # Heavy MCP imports only when actually serving — keeps unit tests fast.
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as mcp_types

    index_path = os.path.abspath(args.index_path)
    logger.info("ragrag MCP server starting against %s", index_path)

    server = Server("ragrag")

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="search_documentation",
                description=(
                    "Search the local ragrag knowledge base for topics relevant to a "
                    "natural-language query. Returns the top-K topic chunks as a list "
                    "of objects with title, summary, page refs, a directory-level "
                    "location block, and (when available) base64-encoded page images "
                    "for the hero page of each result."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional subset of paths to search. Defaults to the index root.",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        if name != "search_documentation":
            raise ValueError(f"unknown tool: {name!r}")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        top_k = int(arguments.get("top_k") or args.default_top_k)
        paths = arguments.get("paths") or [index_path]

        # Run the blocking search in a worker thread so the MCP event
        # loop stays responsive.
        result = await asyncio.to_thread(
            _run_search,
            index_path=index_path,
            query=query,
            top_k=top_k,
            paths=paths,
            image_mode=args.image_mode,
            max_inline_image_kb=args.max_inline_image_kb,
        )
        return [
            mcp_types.TextContent(
                type="text",
                text=json.dumps(result, indent=2),
            )
        ]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
    return 0


# --------------------------------------------------------------------------- #
# Search dispatch (daemon-first, in-process fallback)
# --------------------------------------------------------------------------- #


def _run_search(
    *,
    index_path: str,
    query: str,
    top_k: int,
    paths: list[str],
    image_mode: str,
    max_inline_image_kb: int,
) -> dict[str, Any]:
    """Execute one search and return a compact dict for the MCP client."""
    from ragrag.daemon.client import DaemonClient, DaemonError

    response: Optional[dict[str, Any]] = None
    try:
        client = DaemonClient(index_path)
        response = client.search(
            query=query,
            paths=paths,
            top_k=top_k,
        )
    except (DaemonError, OSError) as exc:
        logger.warning("Daemon unreachable (%s); falling back to in-process", exc)
        response = _inprocess_search(index_path, query, top_k, paths)

    return _trim_response(response or {}, image_mode, max_inline_image_kb)


def _inprocess_search(
    index_path: str,
    query: str,
    top_k: int,
    paths: list[str],
) -> dict[str, Any]:
    from ragrag.config import get_settings
    from ragrag.embedding.colqwen_embedder import ColQwenEmbedder
    from ragrag.index.ingest_manager import IngestManager
    from ragrag.index.qdrant_store import COLLECTION_NAME, QdrantStore
    from ragrag.models import SearchRequest
    from ragrag.retrieval.search_engine import SearchEngine

    settings = get_settings(index_path)
    embedder = ColQwenEmbedder(
        settings.model_id,
        settings.max_visual_tokens,
        quantization=settings.quantization,
        defer_load=True,
    )
    store = QdrantStore(settings.index_path, COLLECTION_NAME, embedder.embedding_dim)
    ingest = IngestManager(embedder, store, settings)
    engine = SearchEngine(embedder, store, ingest, settings)
    request = SearchRequest(query=query, paths=paths, top_k=top_k)
    response = engine.search(request)
    return response.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Response trimming + base64 image inlining
# --------------------------------------------------------------------------- #


def _trim_response(
    response: dict[str, Any],
    image_mode: str,
    max_inline_image_kb: int,
) -> dict[str, Any]:
    """Strip oversized fields and optionally inline base64 hero images."""
    results = response.get("results") or []
    budget_bytes = max_inline_image_kb * 1024
    used = 0

    compact_results: list[dict[str, Any]] = []
    for r in results:
        # Keep only the hero page's image; drop the rest to avoid flooding
        # the MCP client context.
        context_pages = r.get("context_pages") or []
        hero_page = None
        if context_pages:
            hero_page = context_pages[0]
            for ctx in context_pages:
                if ctx.get("page") == r.get("page") or ctx.get("page") == (r.get("page_refs") or [None])[0]:
                    hero_page = ctx
                    break

        hero_b64: Optional[str] = None
        hero_path: Optional[str] = None
        if hero_page:
            hero_path = hero_page.get("page_image_path")
            if image_mode == "base64" and hero_path:
                try:
                    data = Path(hero_path).read_bytes()
                except OSError:
                    data = None
                if data and used + len(data) <= budget_bytes:
                    hero_b64 = base64.b64encode(data).decode("ascii")
                    used += len(data)

        compact_results.append(
            {
                "rank": r.get("rank"),
                "score": r.get("score"),
                "path": r.get("path"),
                "title": r.get("title"),
                "summary": r.get("summary"),
                "page_refs": r.get("page_refs"),
                "excerpt": (r.get("excerpt") or "")[:400],
                "rerank_reason": r.get("rerank_reason"),
                "location": r.get("location"),
                "hero_page": (hero_page or {}).get("page") if hero_page else None,
                "hero_page_image_path": hero_path if image_mode != "none" else None,
                "hero_page_image_b64": hero_b64,
            }
        )

    return {
        "query": response.get("query"),
        "status": response.get("status"),
        "total_ms": (response.get("timing_ms") or {}).get("total_ms"),
        "results": compact_results,
    }


if __name__ == "__main__":
    raise SystemExit(main())
