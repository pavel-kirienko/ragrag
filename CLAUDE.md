# For agents

Ragrag is a local multimodal semantic search CLI for technical documentation (text, PDFs with diagrams, images). It uses late-interaction embeddings and vector search, running fully locally. Read README.md to understand the context.

ALWAYS UPDATE AFFECTED DOCUMENTATION SOURCES WHEN THE IMPLEMENTATION IS ALTERED: THIS FILE, README.md, BUILT-IN HELP STRINGS, etc.

## Commands

```bash
pip install -e ".[dev]"                 # Install with dev extras

# Tests via nox; see noxfile.py for further information.
nox

# Validation corpus (real PDFs/fixtures)
python scripts/fetch_validation_data.py
python scripts/validate_search.py
```

The `unit` nox session excludes `tests/test_e2e.py` and runs without needing the embedding model. `test_e2e.py` requires the real ColQwen3 model from HuggingFace and is slow.

## Architecture

The pipeline is split into extraction → embedding → indexing → retrieval, orchestrated by the CLI.

- **`ragrag/cli.py`** — argparse CLI. Resolves config, constructs `SearchEngine`, prints JSON or Markdown. Logs go to stderr; results to stdout.
- **`ragrag/config.py`** — `Settings` (pydantic) loaded from `ragrag.json`/`.ragrag.json`, walking up the directory tree. Also handles index root discovery (`.ragrag/`).
- **`ragrag/path_discovery.py`** — walks input paths honoring `include_hidden`/`follow_symlinks`, classifies files via `ragrag/models.py:get_file_type` (text/PDF/image).
- **`ragrag/file_state.py`** — `FileStateTracker` persists file mtime/hash under the index dir so unchanged files are skipped on re-ingest.
- **`ragrag/extractors/`** — per-modality extraction:
  - `text_extractor.py` chunks plain text using `chunk_size`/`chunk_overlap`.
  - `pdf_extractor.py` renders PDF pages at `pdf_dpi`, extracts native text, and falls back to `ocr.py` (Tesseract) when page text is below `ocr_threshold`.
  - `image_extractor.py` loads standalone images.
- **`ragrag/embedding/colqwen_embedder.py`** — wraps the ColQwen3 model (`TomoroAI/tomoro-colqwen3-embed-4b` by default) and produces MultiVector (late-interaction) embeddings for both text and images. Respects `max_visual_tokens`. Metal/CPU/CUDA device selection lives here.
- **`ragrag/index/qdrant_store.py`** — thin Qdrant wrapper using a local on-disk collection under `index_path`. Stores MultiVector points with payload metadata (source path, modality, page, chunk id).
- **`ragrag/index/ingest_manager.py`** — drives discovery → extract → embed → upsert. Enforces `indexing_timeout` as a soft cap, emitting `SkippedFile` entries for anything deferred.
- **`ragrag/retrieval/search_engine.py`** — the orchestrator: runs lazy ingest first, embeds the query, calls Qdrant MaxSim, returns `SearchResponse` with `TimingInfo` and indexing stats.
- **`ragrag/retrieval/result_formatter.py`** — converts Qdrant `ScoredPoint`s into `SearchResult` objects for CLI output.
- **`ragrag/models.py`** — shared pydantic models (`Segment`, `Modality`, `FileType`, `SearchRequest/Response/Result`, `IndexingStats`, `SkippedFile`, `TimingInfo`). This is the contract between layers; changes here ripple widely.

Update this section when the architecture is changed.

### Key cross-cutting behaviors

- **Lazy index-on-demand**: every search call first runs `IngestManager.ingest_paths` so new/changed files are indexed before query embedding. Tests often mock the embedder to avoid this cost.
- **Config resolution** is hierarchical (CWD up to root), and `index_path` in the config is resolved relative to the config file, not CWD. Tests covering this live in `tests/test_cli.py` and `tests/test_todo_cases.py`.
- **Modalities are unified** in Qdrant under one collection — text chunks and rendered PDF page images share the vector space so a single query embedding retrieves both.
