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
import logging
import os
import sys

from src.config import find_index_root, get_settings


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
        "--new",
        action="store_true",
        default=False,
        help="Create a new index in the current directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO). Logs go to stderr.",
    )
    return parser


def main() -> int:
    """Entry point for the ragrag CLI. Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args()

    from src.embedding.colqwen_embedder import ColQwenEmbedder
    from src.index.ingest_manager import IngestManager
    from src.index.qdrant_store import QdrantStore, COLLECTION_NAME
    from src.models import SearchRequest
    from src.retrieval.result_formatter import format_as_json, format_as_markdown
    from src.retrieval.search_engine import SearchEngine

    # Configure logging to stderr only
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    for noisy in ("transformers", "httpx", "urllib3", "qdrant_client", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print("ragrag: starting up...", file=sys.stderr, flush=True)

    if args.new:
        root_dir = os.getcwd()
        settings = get_settings(root_dir)
        index_path = os.path.abspath(os.path.join(root_dir, ".ragrag"))
        os.makedirs(index_path, exist_ok=True)
        settings = settings.model_copy(update={"index_path": index_path})
    else:
        _, settings = find_index_root()

    logging.info("Using index path: %s", settings.index_path)

    if args.model:
        settings = settings.model_copy(update={"model_id": args.model})

    top_k = args.top_k if args.top_k is not None else settings.top_k
    use_markdown = args.output_markdown

    try:
        logging.info("Initializing model '%s' (first run may take a long time)...", settings.model_id)
        embedder = ColQwenEmbedder(settings.model_id, settings.max_visual_tokens)

        logging.info("Opening local vector store...")
        store = QdrantStore(settings.index_path, COLLECTION_NAME, embedder.embedding_dim)
        ingest_mgr = IngestManager(embedder, store, settings)
        engine = SearchEngine(embedder, store, ingest_mgr, settings)

        request = SearchRequest(
            paths=args.paths,
            query=args.query,
            top_k=top_k,
            include_markdown=use_markdown,
        )

        logging.info("Running indexing + search over %d path(s)...", len(args.paths))
        response = engine.search(request)
        logging.info(
            "Search complete: status=%s results=%d added=%d updated=%d skipped_unchanged=%d",
            response.status,
            len(response.results),
            response.indexed_now.files_added,
            response.indexed_now.files_updated,
            response.indexed_now.files_skipped_unchanged,
        )

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
