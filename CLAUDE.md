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

The pipeline is split into extraction → plan (VLM topic chunking) → embed → index → retrieval, orchestrated by the CLI or the daemon.

- **`ragrag/cli.py`** — argparse CLI. Auto-spawns the daemon when available, falls back to in-process when not. Logs go to stderr; results to stdout. Sets `HF_HUB_OFFLINE=1` before importing torch.
- **`ragrag/config.py`** — `Settings` (pydantic) loaded from `ragrag.json` / `.ragrag.json`, walking up the directory tree.
- **`ragrag/path_discovery.py`**, **`ragrag/file_state.py`** — file enumeration + SHA-256 based staleness tracking under `<index>/file_state/`.
- **`ragrag/extractors/pdf_extractor.py`** — renders pages at `pdf_dpi`, yields `(Segment, PIL.Image)` pairs, OCR fallback via `ocr.py` when page text is too short.
- **`ragrag/extractors/vlm_topic_client.py`** — single shared VLM prompt/parser for PDF + text chunking. Compact one-level JSON schema used when `chunker_stride_pages == 1` (the default); the parser has a salvage path that recovers truncated JSON.
- **`ragrag/extractors/vlm_topic_chunker.py`**, **`vlm_topic_segmenter.py`** — wrap the client to emit `Chunk` objects per file.
- **`ragrag/extractors/vlm_topic_worker.py`** — **runs in a subprocess**. The parent process spawns `python -m ragrag.extractors.vlm_topic_worker`, pipes in a JSON request (list of files + settings), and reads JSON-Line chunks on stdout. The child loads the VLM, plans every requested file, and exits. Subprocess isolation is what keeps the embedder reliable on tight GPUs: `bnb` 4-bit leaves non-PyTorch CUDA context state behind after unload, and the only way to fully reclaim it is for the child process to exit.
- **`ragrag/extractors/vlm_topic_subprocess.py`** — parent-side wrapper (`SubprocessVLMPlanner`) that `subprocess.run`s the worker.
- **`ragrag/embedding/colqwen_embedder.py`** — ColQwen3 late-interaction embedder. Supports `defer_load=True` so the CLI can construct the handle before the plan phase and only load weights after the VLM subprocess exits. Uses `device_map={"": "cuda:0"}` to bypass accelerate's memory-based dispatch heuristic (which undercounts free VRAM on this GPU profile).
- **`ragrag/embedding/vlm_loader.py`** — shared HF loader for any VLM. Auto-sets `HF_HUB_OFFLINE`, tries `local_files_only=True` first with a network-allowed fallback, and forces `attn_implementation="eager"` to sidestep cuDNN kernel-lookup failures with bnb-4bit compute-dtype combos.
- **`ragrag/index/page_cache.py`** — `PageImageCache`: WebP-backed, SHA-addressed LRU at `<index>/page_cache/<sha[:2]>/<sha>/<page>.webp`.
- **`ragrag/index/qdrant_store.py`** — thin Qdrant wrapper; variable-length multivectors via `offsets.bin`.
- **`ragrag/index/ingest_manager.py`** — two-phase pipeline. **Plan phase**: route stale PDFs / text files to `SubprocessVLMPlanner` in one batch, collect `Chunk` objects. Images bypass the VLM. **Embed phase**: load ColQwen3 (fresh, clean CUDA context), embed per-chunk text + per-page images, upsert two points per chunk (text + image modalities) that share a `chunk_id`.
- **`ragrag/retrieval/search_engine.py`** — orchestrator. Runs lazy ingest first, then embeds the query, does MaxSim retrieval, rolls up duplicate `chunk_id`s (text+image collapse into one result), attaches `Location` block + `context_pages` from the page cache.
- **`ragrag/retrieval/location_builder.py`** — builds the `Location` payload: file path, containing directory, a directory listing (head/tail truncated via `location_directory_listing_max`). Respects `.gitignore` when `location_respect_gitignore=True`.
- **`ragrag/retrieval/result_formatter.py`** — `json` / `compact-json` / `markdown` / `markdown-rich` output formats.
- **`ragrag/daemon/server.py`** — long-lived JSON-RPC server over Unix socket + HTTP status server. Holds one `SearchEngine` per index.
- **`ragrag/daemon/http_status.py`** + **`static.py`** — the bundled dashboard. Static HTML polls `/status` every 2 s. Also serves `/pages/<sha>/<n>.webp` from the engine's `PageImageCache` and a `POST /shutdown` gate.
- **`ragrag/models.py`** — shared pydantic contracts: `Chunk`, `ChunkKind`, `PageContext`, `Location`, `SearchRequest/Response/Result`, `IndexingStats`, `SkippedFile`, `TimingInfo`.

Update this section when the architecture is changed.

### Key cross-cutting behaviors

- **Subprocess VLM isolation**: the VLM topic chunker runs in a child Python process. Do NOT try to re-introduce in-process unload/reload; bnb 4-bit fragmentation on shared CUDA contexts will break the embedder's forward pass. If you need to add another VLM use site (e.g. Phase D reranker), either piggy-back on the same worker protocol or spawn a second child.
- **Deferred embedder load**: `ColQwenEmbedder(defer_load=True)` defers weight loading until `ensure_loaded()` is called. CLI and daemon use this so the VLM subprocess owns the CUDA context during the plan phase; the embedder only loads afterwards, in a pristine context.
- **Lazy index-on-demand**: every search call first runs `IngestManager.ingest_paths` so new/changed files are indexed before query embedding. Tests often mock the embedder and the VLM factory to avoid this cost.
- **Config resolution** is hierarchical (CWD up to root), and `index_path` in the config is resolved relative to the config file, not CWD.
- **Chunks are topics, not pages**: one `Chunk` describes a semantic topic that may reference multiple pages (possibly non-contiguous) and may overlap with other chunks (a page can belong to more than one topic). Storage doubles each topic as a text point + an image point sharing a `chunk_id`; the search rollup re-unifies them.
- **Modalities are unified** in Qdrant under one collection.
