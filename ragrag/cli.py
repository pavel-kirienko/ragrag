"""ragrag — local multimodal semantic search CLI."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Enable expandable segments in torch's CUDA allocator. Must be set BEFORE
# torch is imported anywhere. Reduces fragmentation when the same process
# loads/unloads several large models (ColQwen3 <-> Qwen2.5-VL-3B swap during
# two-pass indexing on 8 GB GPUs).
# Avoid 10-second HEAD requests to huggingface.co on every model load.
# The ColQwen3 + Qwen2.5-VL weights are cached locally; if a user really
# needs a fresh fetch they can unset these explicitly.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ragrag import __version__  # noqa: E402
from ragrag.config import find_index_root, get_settings  # noqa: E402


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
        version=f"%(prog)s {__version__}",
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
        "--format",
        default=None,
        choices=["json", "compact-json", "markdown", "markdown-rich"],
        help=(
            "Explicit output format. Overrides --json / --markdown. "
            "'compact-json' trims excerpts and base64 image payloads for "
            "LLM consumers with tight context budgets; 'markdown-rich' is "
            "the topic-aware Markdown report with embedded page images "
            "and Location blocks."
        ),
    )
    parser.add_argument(
        "--include-page-images",
        default=None,
        choices=["none", "path", "base64"],
        help=(
            "Delivery mode for PDF page images in each result's context_pages. "
            "Default comes from config (`include_page_images_default`, 'path')."
        ),
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
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        default=False,
        help=(
            "Run the search engine in the current process instead of "
            "auto-starting the ragrag daemon. Useful in sandboxed "
            "environments and for debugging. Same effect as setting "
            "RAGRAG_NO_DAEMON=1."
        ),
    )
    return parser


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    for noisy in ("transformers", "httpx", "urllib3", "qdrant_client", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _resolve_settings(args) -> tuple[str, "Settings"]:  # noqa: F821
    """Return (root_dir, settings) for the given CLI args."""
    from ragrag.config import Settings  # noqa: F401  — for type only

    if args.new:
        root_dir = os.getcwd()
        settings = get_settings(root_dir)
        index_path = os.path.abspath(os.path.join(root_dir, ".ragrag"))
        os.makedirs(index_path, exist_ok=True)
        settings = settings.model_copy(update={"index_path": index_path})
    else:
        root_dir, settings = find_index_root()
    if args.model:
        settings = settings.model_copy(update={"model_id": args.model})
    return root_dir, settings


def _daemon_disabled(args) -> bool:
    if getattr(args, "no_daemon", False):
        return True
    if os.environ.get("RAGRAG_NO_DAEMON"):
        return True
    return False


def _resolve_format(args) -> str:
    if args.format:
        return args.format
    if args.output_markdown:
        return "markdown-rich"
    return "json"


def _render_response(result: dict, fmt: str) -> None:
    from ragrag.models import SearchResponse
    from ragrag.retrieval.result_formatter import (
        format_as_compact_json,
        format_as_json,
        format_as_markdown,
        format_as_markdown_rich,
    )

    if fmt == "json":
        print(json.dumps(result, indent=2))
        return
    try:
        response = SearchResponse(**result)
    except Exception:
        print(json.dumps(result, indent=2))
        return
    if fmt == "compact-json":
        print(format_as_compact_json(response))
    elif fmt == "markdown":
        print(format_as_markdown(response))
    else:  # markdown-rich
        print(format_as_markdown_rich(response))


def _run_via_daemon(args, settings) -> int | None:
    """Try to run the query through the daemon. Returns exit code on success,
    or ``None`` when the daemon path is unusable and the caller should fall back.
    """
    from ragrag.daemon.client import DaemonClient, DaemonError, DaemonStartupError

    client = DaemonClient(settings.index_path)
    try:
        client.ensure_daemon()
    except DaemonStartupError as exc:
        logging.warning("Daemon unavailable (%s); falling back to in-process engine", exc)
        return None
    try:
        result = client.search(
            args.query,
            paths=args.paths,
            top_k=args.top_k if args.top_k is not None else settings.top_k,
            include_markdown=args.output_markdown,
        )
    except DaemonError as exc:
        logging.warning("Daemon RPC failed (%s); falling back to in-process engine", exc)
        return None

    _render_response(result if isinstance(result, dict) else {}, _resolve_format(args))
    status = result.get("status") if isinstance(result, dict) else None
    return 0 if status == "complete" else 1


def _run_inprocess(args, settings) -> int:
    """Original in-process search path. Used as fallback / explicit --no-daemon mode."""
    from ragrag.embedding.colqwen_embedder import ColQwenEmbedder
    from ragrag.embedding.vlm_loader import load_vlm
    from ragrag.extractors.vlm_topic_client import VLMTopicClient
    from ragrag.index.ingest_manager import IngestManager
    from ragrag.index.qdrant_store import QdrantStore, COLLECTION_NAME
    from ragrag.models import SearchRequest
    from ragrag.retrieval.result_formatter import format_as_json, format_as_markdown
    from ragrag.retrieval.search_engine import SearchEngine

    top_k = args.top_k if args.top_k is not None else settings.top_k
    fmt = _resolve_format(args)

    logging.info("Initializing model '%s' (first run may take a long time)...", settings.model_id)
    # Defer the embedder load. On a tight 8 GB card the bnb 4-bit VLM
    # leaves ~1.5 GiB of non-PyTorch CUDA context allocations behind
    # after unload, which breaks a subsequent embedder reload. Loading
    # the embedder only AFTER the VLM plan phase has finished avoids
    # that entirely on the first ingest pass.
    embedder = ColQwenEmbedder(
        settings.model_id,
        settings.max_visual_tokens,
        quantization=settings.quantization,
        defer_load=True,
    )

    # VLM topic client is loaded lazily per indexing call via a factory,
    # so searches over an already-indexed corpus never pay the VLM load cost.
    def _vlm_factory(device: str | None = None) -> VLMTopicClient:
        logging.info(
            "Loading VLM topic client '%s' (device=%s) ...",
            settings.vlm_model_id, device or "auto",
        )
        handle = load_vlm(
            settings.vlm_model_id,
            quantization=settings.vlm_quantization,
            device=device,
        )
        return VLMTopicClient(handle, image_max_side=settings.chunker_vlm_image_max_side)

    logging.info("Opening local vector store...")
    store = QdrantStore(settings.index_path, COLLECTION_NAME, embedder.embedding_dim)
    ingest_mgr = IngestManager(embedder, store, settings, vlm_factory=_vlm_factory)

    reranker = None
    if settings.reranker_model and settings.reranker_model.lower() not in {"none", ""}:
        try:
            from ragrag.retrieval.reranker import VLMReranker

            reranker = VLMReranker(settings)
            logging.info("VLM reranker armed (model=%s)", settings.vlm_model_id)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Reranker init failed: %s — falling back to MaxSim-only", exc)
            reranker = None

    engine = SearchEngine(embedder, store, ingest_mgr, settings, reranker=reranker)

    request = SearchRequest(
        paths=args.paths,
        query=args.query,
        top_k=top_k,
        include_markdown=(fmt.startswith("markdown")),
        include_page_images=args.include_page_images,
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

    _render_response(response.model_dump(), fmt)
    return 0 if response.status == "complete" else 1


def _run_daemon_subcommand(argv: list[str]) -> int:
    """ragrag daemon … — start a daemon process. Delegates to ragrag.daemon.server.main."""
    from ragrag.daemon.server import main as daemon_main

    return daemon_main(argv)


def _run_status_subcommand() -> int:
    """ragrag status — print the daemon's /status JSON."""
    from ragrag.daemon.client import DaemonClient, DaemonError, DaemonStartupError

    try:
        root_dir, settings = find_index_root()
    except SystemExit:
        print("ragrag: no index found in this directory tree", file=sys.stderr)
        return 1
    client = DaemonClient(settings.index_path)
    if not client.socket_path.exists():
        print("ragrag: no daemon running for this index", file=sys.stderr)
        return 1
    try:
        snapshot = client.status()
    except DaemonError as exc:
        print(f"ragrag: daemon error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(snapshot, indent=2))
    return 0


def _run_shutdown_subcommand() -> int:
    """ragrag shutdown — politely shut down the daemon for the current index."""
    from ragrag.daemon.client import DaemonClient, DaemonError

    try:
        root_dir, settings = find_index_root()
    except SystemExit:
        print("ragrag: no index found in this directory tree", file=sys.stderr)
        return 1
    client = DaemonClient(settings.index_path)
    if not client.socket_path.exists():
        print("ragrag: no daemon running for this index", file=sys.stderr)
        return 0
    try:
        client.shutdown()
    except DaemonError as exc:
        print(f"ragrag: daemon error: {exc}", file=sys.stderr)
        return 1
    print("ragrag: daemon shut down")
    return 0


def main() -> int:
    """Entry point for the ragrag CLI. Returns exit code."""
    # Sub-command dispatch (peeks at argv[1] before letting argparse parse the search form).
    argv = sys.argv[1:]
    if argv and argv[0] in {"daemon", "status", "shutdown"}:
        sub, rest = argv[0], argv[1:]
        # Minimal logging for sub-commands (the daemon configures its own logger).
        if sub == "daemon":
            return _run_daemon_subcommand(rest)
        _setup_logging("WARNING")
        if sub == "status":
            return _run_status_subcommand()
        if sub == "shutdown":
            return _run_shutdown_subcommand()

    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.log_level)

    print("ragrag: starting up...", file=sys.stderr, flush=True)

    try:
        _root_dir, settings = _resolve_settings(args)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    logging.info("Using index path: %s", settings.index_path)

    try:
        if not _daemon_disabled(args) and settings.daemon_autostart:
            exit_code = _run_via_daemon(args, settings)
            if exit_code is not None:
                return exit_code
        return _run_inprocess(args, settings)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        logging.debug("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
