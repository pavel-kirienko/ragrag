#!/usr/bin/env python3
"""CLI search tool for ragrag.

Usage:
    python scripts/search_cli.py --query QUERY --paths PATH [PATH ...] [--top-k N] [--json] [--model MODEL_ID]

Examples:
    python scripts/search_cli.py --query "machine learning" --paths ./docs ./src
    python scripts/search_cli.py --query "error handling" --paths . --top-k 20 --json
    python scripts/search_cli.py --query "authentication" --paths ./src --model "TomoroAI/tomoro-colqwen3-embed-4b"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from src.config import get_settings
from src.embedding.colqwen_embedder import ColQwenEmbedder
from src.index.ingest_manager import IngestManager
from src.index.qdrant_store import QdrantStore
from src.models import SearchRequest
from src.retrieval.result_formatter import format_as_json, format_as_markdown
from src.retrieval.search_engine import SearchEngine


logger = logging.getLogger(__name__)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="search_cli.py",
        description="Search documents using ColQwen3 embeddings and Qdrant vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural-language search query",
    )

    parser.add_argument(
        "--paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more file paths or directories to search",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10, max: 50)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of Markdown",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model ID (default from settings)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load settings
        settings = get_settings()

        # Override model if provided
        if args.model:
            settings.MODEL_ID = args.model

        # Validate top_k
        if args.top_k < 1 or args.top_k > settings.TOP_K_MAX:
            logger.error(
                f"top_k must be between 1 and {settings.TOP_K_MAX}, got {args.top_k}"
            )
            return 1

        # Initialize embedder
        logger.info(f"Initializing embedder with model: {settings.MODEL_ID}")
        embedder = ColQwenEmbedder(settings.MODEL_ID, settings.MAX_VISUAL_TOKENS)

        # Initialize Qdrant store
        logger.info(f"Initializing Qdrant store at {settings.QDRANT_PATH}")
        store = QdrantStore(
            settings.QDRANT_PATH,
            settings.QDRANT_COLLECTION,
            embedder.embedding_dim,
        )

        # Initialize ingest manager
        ingest_mgr = IngestManager(embedder, store, settings)

        # Initialize search engine
        engine = SearchEngine(embedder, store, ingest_mgr, settings)

        # Build search request
        request = SearchRequest(
            paths=args.paths,
            query=args.query,
            top_k=args.top_k,
            include_markdown=not args.json,
        )

        # Execute search
        logger.info(f"Searching for: {args.query}")
        response = engine.search(request)

        # Format and print results
        if args.json:
            output = format_as_json(response)
        else:
            output = format_as_markdown(response)

        print(output)

        # Return success if no errors, partial success if errors occurred
        return 0 if response.status == "complete" else 1

    except Exception as exc:
        logger.exception(f"Search failed: {exc}")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
