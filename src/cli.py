"""ragrag — local multimodal semantic search CLI."""
from __future__ import annotations

import argparse
import logging
import os
import sys

from src.config import find_index_root, get_settings


_DESCRIPTION = (
    "ragrag - local multimodal semantic search for technical documentation. "
    "Indexes text files, PDFs (including diagrams via page rendering), and images "
    "into a local vector store, and answers natural-language queries using "
    "late-interaction (ColQwen3) embeddings. Runs 100% locally after the first "
    "model download."
)


_EPILOG = """\
CORE CONCEPTS
  * Lazy index-on-demand: every invocation first re-indexes new or changed
    files under <path...> before running the query. Unchanged files are
    skipped via SHA-256 content hash. There is no separate "index" subcommand.
  * Index location: a directory named `.ragrag/` discovered by walking from
    the current directory upward. If none is found and no config file
    (`ragrag.json` / `.ragrag.json`) exists on the path, the tool exits with
    an error asking you to re-run with `--new`.
  * `--new` forces creation of `.ragrag/` in the *current* working directory,
    ignoring any index that might exist in a parent. Use this exactly once
    per project root. Subsequent calls from that directory (or any
    subdirectory) will reuse it automatically - do NOT pass `--new` again.
  * Stdout carries ONLY the formatted result (JSON or Markdown). Stderr
    carries logs, progress, and errors. Parse stdout; watch stderr for
    diagnostics. Exit code 0 = complete, 1 = partial or error.
  * First run downloads the ColQwen3 model from HuggingFace (several GB) and
    may take many minutes. Subsequent runs are fully offline.

EXAMPLES
  # Search the current directory (uses/creates index in nearest ancestor)
  ragrag "clock tree configuration"

  # First-time setup in a project root: create a new index here
  ragrag --new "GPIO pin mux" ./docs ./datasheets

  # Restrict search to two directories, return 20 results, human-readable
  ragrag "SPI timing diagram" ./pdfs ./refman --top-k 20 --markdown

  # Quiet run - suppress INFO logs on stderr
  ragrag "DMA configuration" --log-level WARNING

  # Override the embedding model (advanced)
  ragrag "reset vector" --model TomoroAI/tomoro-colqwen3-embed-4b

OUTPUT FORMAT (stdout)
  Default: JSON, pretty-printed with 2-space indent. Top-level keys:
    query          (str)  - the query you passed
    status         (str)  - "complete" or "partial" (partial => see errors)
    indexed_now    (obj)  - {files_added, files_updated, files_skipped_unchanged}
    skipped_files  (list) - [{path, reason}, ...]
    errors         (list) - human-readable error strings
    results        (list) - ranked SearchResult objects (see below)
    timing_ms      (obj)  - {discovery_ms, indexing_ms, query_embedding_ms,
                             retrieval_ms, formatting_ms, total_ms}
  Each entry in `results` has:
    rank (int), score (float), path (str),
    file_type ("text"|"pdf"|"image"),
    modality  ("text"|"image"),
    page (int|null)       - 1-indexed PDF page, null for non-PDF
    start_line, end_line  - 1-indexed line range, set for text hits
    excerpt (str)         - short human-readable snippet
  `--markdown` replaces the JSON on stdout with a human-readable report.
  Use JSON (the default) for programmatic parsing.

SUPPORTED FILE TYPES
  Text:  .txt .md .rst .py .js .ts .c .h .cpp .hpp .cc .cxx
         .json .yaml .yml .toml .ini .cfg
  PDF:   .pdf   (pages rendered as images; Tesseract OCR fallback kicks in
                 when a page's native text is below `ocr_threshold` chars)
  Image: .png .jpg .jpeg .bmp .tiff .tif .webp
  Detection uses libmagic MIME types; the list above is the practical set.
  Other files are silently ignored during discovery.

CONFIGURATION FILE
  Optional `ragrag.json` or `.ragrag.json` (first-found-wins), discovered by
  walking up from CWD. `index_path` in the config is resolved relative to
  the config file's directory, not CWD. All fields optional; defaults:
    index_path          ".ragrag"
    model_id            "TomoroAI/tomoro-colqwen3-embed-4b"
    max_visual_tokens   16384
    quantization        "auto"     # auto | none | 8bit | 4bit (GPU only)
    top_k               10
    max_top_k           50         # hard cap on --top-k
    chunk_size          900        # chars per text chunk
    chunk_overlap       200        # chars of overlap between chunks
    text_batch_size     8          # text chunks per forward pass (1..64)
    pdf_dpi             250
    ocr_threshold       50         # native chars required to skip OCR
    include_hidden      false
    follow_symlinks     true
    indexing_timeout    100000     # soft cap in seconds

RECOMMENDED WORKFLOW FOR AGENTS
  1. Run once with `--new` at the project root to create `.ragrag/` unless it already exists.
  2. Thereafter, call `ragrag "<query>" [paths...]` from anywhere in the
     tree - the index is auto-discovered and re-indexing is incremental as needed.
  3. Parse the `results` array from stdout JSON. Use `page` (PDFs) or
     `start_line`/`end_line` (text) from each hit to locate evidence.
  4. On `status == "partial"`, inspect `errors` and `skipped_files` before
     trusting the ranking.

EXIT CODES
  0  search completed (status == "complete")
  1  partial result, interrupted, or unhandled error
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragrag",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "query",
        help="Natural-language search query. Quote it if it contains spaces.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        metavar="path",
        help=(
            "Files or directories to search under. Default: current directory. "
            "Multiple paths allowed; directories are walked recursively."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max results to return. Default: `top_k` from config (10 if unset). "
            "Capped by `max_top_k` from config (hard max 50)."
        ),
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        default=True,
        help="Emit results as JSON on stdout (default). See OUTPUT FORMAT below.",
    )
    output_group.add_argument(
        "--markdown",
        dest="output_markdown",
        action="store_true",
        default=False,
        help="Emit a human-readable Markdown report on stdout instead of JSON.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=(
            "Override the embedding model HuggingFace ID for this call. "
            "Advanced; the default ColQwen3 model is recommended."
        ),
    )
    parser.add_argument(
        "--new",
        action="store_true",
        default=False,
        help=(
            "Create a fresh index `.ragrag/` in the current directory, "
            "ignoring any parent index. Use once per project root; omit on "
            "subsequent calls (the index is auto-discovered)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=(
            "Stderr log verbosity (default: INFO). Results always go to "
            "stdout regardless of log level."
        ),
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
        embedder = ColQwenEmbedder(
            settings.model_id,
            settings.max_visual_tokens,
            quantization=settings.quantization,
        )

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
