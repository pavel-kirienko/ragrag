#!/usr/bin/env python3
"""Automated validation harness for ragrag semantic search pipeline.

Loads queries from a YAML file, runs each through the full pipeline, and
checks that results meet the expected criteria.

Usage:
    python scripts/validate_search.py [--queries-file PATH] [--verbose] [--model MODEL_ID]

Examples:
    python scripts/validate_search.py
    python scripts/validate_search.py --queries-file validation/expected/queries.yaml
    python scripts/validate_search.py --verbose
    python scripts/validate_search.py --model "TomoroAI/tomoro-colqwen3-embed-4b" --verbose
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src.config import get_settings
from src.embedding.colqwen_embedder import ColQwenEmbedder
from src.index.ingest_manager import IngestManager
from src.index.qdrant_store import QdrantStore, COLLECTION_NAME
from src.models import SearchRequest
from src.retrieval.search_engine import SearchEngine


logger = logging.getLogger(__name__)

DEFAULT_QUERIES_FILE = "validation/expected/queries.yaml"


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_queries(queries_file: str) -> list[dict[str, Any]]:
    """Load and validate queries from a YAML file.

    Args:
        queries_file: Path to the YAML file containing queries.

    Returns:
        List of query dicts.

    Raises:
        FileNotFoundError: If the queries file does not exist.
        ValueError: If the YAML is malformed or missing required fields.
    """
    path = Path(queries_file)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "queries" not in data:
        raise ValueError(f"Invalid queries file: expected top-level 'queries' key in {queries_file}")

    queries = data["queries"]
    if not isinstance(queries, list):
        raise ValueError(f"'queries' must be a list in {queries_file}")

    # Validate required fields
    for i, q in enumerate(queries):
        for field in ("id", "query", "paths"):
            if field not in q:
                raise ValueError(f"Query [{i}] is missing required field '{field}'")

    return queries


# ---------------------------------------------------------------------------
# Per-query validation
# ---------------------------------------------------------------------------

def run_query_checks(
    query: dict[str, Any],
    engine: SearchEngine,
    verbose: bool,
) -> tuple[int, int, float]:
    """Run a single query through the pipeline and validate the results.

    Args:
        query: The query dict loaded from YAML.
        engine: The initialized SearchEngine.
        verbose: Whether to print detailed output.

    Returns:
        Tuple of (passed_checks, total_checks, elapsed_ms).
    """
    qid = query.get("id", "<unnamed>")
    description = query.get("description", "")
    raw_query = query["query"]
    paths = query["paths"]
    min_results = query.get("min_results", 0)
    expected_paths_contain: list[str] = query.get("expected_paths_contain", [])
    expect_skipped: bool = query.get("expect_skipped", False)
    top_k = 10

    print(f"\n{'─' * 60}")
    print(f"Query: {qid}")
    if description:
        print(f"  Description : {description}")
    if verbose:
        print(f"  Query text  : {raw_query}")
        print(f"  Paths       : {paths}")

    t0 = time.time()
    request = SearchRequest(paths=paths, query=raw_query, top_k=top_k)
    response = engine.search(request)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"  Timing      : {elapsed_ms:.0f} ms")
    if verbose:
        print(f"  Results     : {len(response.results)} (top_k={top_k})")
        print(f"  Skipped     : {len(response.skipped_files)}")
        print(f"  Status      : {response.status}")
        if response.errors:
            print(f"  Errors      : {response.errors}")
        if response.results:
            print("  Top results :")
            for r in response.results[:3]:
                print(f"    [{r.rank}] score={r.score:.4f}  {r.path}")

    passed = 0
    total = 0

    # Check 1: min_results
    total += 1
    got = len(response.results)
    if got >= min_results:
        passed += 1
        print(f"  [PASS] min_results: got {got} >= {min_results}")
    else:
        print(f"  [FAIL] min_results: got {got} < {min_results}")

    # Check 2: expected_paths_contain (only if list is non-empty)
    if expected_paths_contain:
        total += 1
        result_paths = [r.path for r in response.results]
        matched = any(
            any(substr in rpath for rpath in result_paths)
            for substr in expected_paths_contain
        )
        if matched:
            passed += 1
            print(f"  [PASS] expected_paths_contain: found match for {expected_paths_contain}")
        else:
            print(f"  [FAIL] expected_paths_contain: none of {expected_paths_contain} found in result paths")
            if verbose:
                print(f"         Result paths: {result_paths}")

    # Check 3: expect_skipped
    if expect_skipped:
        total += 1
        skipped_count = len(response.skipped_files)
        if skipped_count > 0:
            passed += 1
            print(f"  [PASS] expect_skipped: {skipped_count} file(s) skipped")
        else:
            print(f"  [FAIL] expect_skipped: no files were skipped but expect_skipped=True")
            if verbose:
                print(f"         Paths searched: {paths}")

    return passed, total, elapsed_ms


# ---------------------------------------------------------------------------
# Pipeline initialization
# ---------------------------------------------------------------------------

def build_engine(model_id: str | None = None) -> SearchEngine:
    """Initialize the full pipeline: embedder → store → ingest_mgr → engine.

    Args:
        model_id: Optional model ID override.

    Returns:
        Initialized SearchEngine.
    """
    settings = get_settings()

    if model_id:
        # Pydantic settings are normally immutable; use model_copy to override
        settings = settings.model_copy(update={"model_id": model_id})

    logger.info(f"Loading embedder: {settings.model_id}")
    embedder = ColQwenEmbedder(settings.model_id, settings.max_visual_tokens)

    logger.info(f"Connecting to index at {settings.index_path}")
    store = QdrantStore(
        settings.index_path,
        COLLECTION_NAME,
        embedder.embedding_dim,
    )

    ingest_mgr = IngestManager(embedder, store, settings)
    engine = SearchEngine(embedder, store, ingest_mgr, settings)

    return engine


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="validate_search.py",
        description=(
            "Validation harness for ragrag. "
            "Runs semantic search queries and checks results against expectations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--queries-file",
        type=str,
        default=DEFAULT_QUERIES_FILE,
        metavar="PATH",
        help=f"Path to YAML file with validation queries (default: {DEFAULT_QUERIES_FILE})",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each query (top results, skipped files, etc.)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL_ID",
        help="Override HuggingFace model ID (default from settings)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: WARNING)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the validation harness.

    Returns:
        0 if all checks pass, 1 if any fail.
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load queries
    try:
        queries = load_queries(args.queries_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading queries: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(queries)} queries from {args.queries_file}")

    # Build pipeline
    print("Initializing pipeline (this may take several minutes on CPU)...")
    try:
        engine = build_engine(model_id=args.model)
    except Exception as exc:
        print(f"Error initializing pipeline: {exc}", file=sys.stderr)
        logger.exception("Pipeline init failed")
        return 1

    print("Pipeline ready.\n")

    # Run queries
    total_passed = 0
    total_checks = 0
    total_elapsed_ms = 0.0

    for query in queries:
        try:
            passed, checks, elapsed_ms = run_query_checks(query, engine, verbose=args.verbose)
        except Exception as exc:
            qid = query.get("id", "<unnamed>")
            print(f"\n  [ERROR] Query '{qid}' raised an exception: {exc}", file=sys.stderr)
            logger.exception(f"Query '{qid}' failed with exception")
            # Count all expected checks as failed
            checks = 1 + (1 if query.get("expected_paths_contain") else 0) + (1 if query.get("expect_skipped") else 0)
            passed = 0
            elapsed_ms = 0.0

        total_passed += passed
        total_checks += checks
        total_elapsed_ms += elapsed_ms

    # Summary
    print(f"\n{'═' * 60}")
    print(f"SUMMARY: {total_passed}/{total_checks} checks passed  "
          f"(total time: {total_elapsed_ms:.0f} ms)")
    print("═" * 60)

    if total_passed == total_checks:
        print("ALL CHECKS PASSED ✓")
        return 0
    else:
        failed = total_checks - total_passed
        print(f"{failed} CHECK(S) FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
