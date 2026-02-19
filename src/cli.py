"""ragrag — local multimodal semantic search CLI.

Usage:
    ragrag <query> [paths...] [options]

Examples:
    ragrag "clock tree configuration" ./docs ./datasheets
    ragrag "SPI timing diagram" .
    ragrag "GPIO initialization" --top-k 20
    ragrag "motor controller specs" ./pdfs --markdown
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from src.config import get_settings
from src.embedding.colqwen_embedder import ColQwenEmbedder
from src.index.ingest_manager import IngestManager
from src.index.qdrant_store import QdrantStore, COLLECTION_NAME
from src.models import SearchRequest
from src.retrieval.result_formatter import format_as_json, format_as_markdown
from src.retrieval.search_engine import SearchEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragrag",
        description="Local multimodal semantic search for embedded development docs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "query",
        help="Search query string.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        metavar="path",
        help="Files or directories to search (default: current directory).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="N",
        help="Number of results to return (default: from config, usually 10).",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        default=True,
        help="Output results as JSON (default).",
    )
    output_group.add_argument(
        "--markdown",
        dest="output_markdown",
        action="store_true",
        default=False,
        help="Output results as Markdown.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Override the embedding model ID.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING). Logs go to stderr.",
    )
    return parser


def main() -> int:
    """Entry point for the ragrag CLI. Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging to stderr only
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Load settings, apply CLI overrides
    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"model_id": args.model})

    top_k = args.top_k if args.top_k is not None else settings.top_k
    use_markdown = args.output_markdown

    try:
        # Progress to stderr for long-running ops
        print("Loading model (this may take a few minutes on first run)...", file=sys.stderr)
        embedder = ColQwenEmbedder(settings.model_id, settings.max_visual_tokens)
        print("Model loaded.", file=sys.stderr)

        store = QdrantStore(settings.index_path, COLLECTION_NAME, embedder.embedding_dim)
        ingest_mgr = IngestManager(embedder, store, settings)
        engine = SearchEngine(embedder, store, ingest_mgr, settings)

        request = SearchRequest(
            paths=args.paths,
            query=args.query,
            top_k=top_k,
            include_markdown=use_markdown,
        )

        response = engine.search(request)

        # Output to stdout
        if use_markdown:
            print(format_as_markdown(response))
        else:
            print(format_as_json(response))

        return 0 if response.status == "complete" else 1

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        logging.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
