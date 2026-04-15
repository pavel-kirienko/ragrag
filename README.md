<div align="center">

# Ragrag — local multimodal RAG for documents

_RTFM for AI_

[![PyPI](https://img.shields.io/pypi/v/ragrag.svg)](https://pypi.org/project/ragrag/)

</div>

Local *multimodal* semantic search using ColQwen3 late-interaction embeddings + Qdrant MaxSim retrieval.
Indexes text files, PDFs with images/diagrams, and standalone images.

Ragrag is originally designed to allow AI agents to read complex technical documentation when doing embedded development, where simple text-based indexing won't work due to abundance of diragrams, schematics, and complex tabular data.

## Usage

Install:

```bash
pip install ragrag
```

⚠️ The very first run will take a **very long time** because the tool will download the model from Huggingface. Despite downloading from the internet, indexing and search run 100% locally.

Search all supported documents in the current directory and subdirectories that are related to clock tree configuration:

```bash
ragrag "clock tree configuration"
```

When a new file is found or an existing file is changed, the model will automatically re-index it (no need to tell it to index manually), which may take anywhere from a few seconds to who knows how long depending on the documents and the performance of your computer (everything is done locally).

The index is stored in `.ragrag/`; the tool will attempt to locate an existing index in the current or parent directories. If none is found, it will attempt to guess where to create the index; if it cannot guess reliably, it will ask you to confirm using `--new`.

The tool may log in stderr, while the search results go to stdout.

Search specific directories with more results and with Markdown output:

```bash
ragrag "GPIO initialization" --top-k 20 --markdown
```

For more options see `ragrag --help`.

### Daemon, dashboard, and MCP

Ragrag ships with a local daemon that keeps models loaded between
queries and cuts the typical warm-query latency to under two
seconds. The first CLI invocation auto-spawns it; subsequent calls
reuse it over a Unix socket at `<index>/.ragrag/daemon.sock`.

- `ragrag status` — print a JSON snapshot of the running daemon.
- `ragrag shutdown` — tell the daemon to exit.
- `ragrag daemon --idle` — start a daemon explicitly in the background.
- A minimal HTML dashboard is served on `http://127.0.0.1:27272/`
  (port configurable) showing models loaded, VRAM/RAM use, current
  indexing progress, and the last 20 queries. `/pages/<sha>/<n>.webp`
  serves cached page images directly.

Ragrag can also expose itself as an MCP tool over stdio for agent
clients:

```bash
pip install -e '.[mcp]'
ragrag mcp --index-path /path/to/docs
```

The single `search_documentation` tool accepts a natural-language
query and returns topic chunks with titles, summaries, page refs,
a directory-level location block, and (optionally) base64-encoded
hero page images.

### Topic chunking and reranking

Each indexable unit is a **topic** identified by a local VLM
(Qwen2.5-VL-3B by default), not a fixed-size text chunk or a
physical page. A topic may span multiple pages, possibly
non-contiguous, and two topics may overlap on the same page when
the content genuinely belongs to both. The chunker runs in an
isolated subprocess so its bnb 4-bit CUDA context state never
contaminates the embedder's allocator.

An optional listwise VLM reranker can be armed by setting
`"reranker_model": "vlm"` in `ragrag.json`. It spawns a second
persistent subprocess that takes the top-K candidates from the
MaxSim retrieval and reorders them by directly inspecting the
hero page image plus the topic title and summary. Opt-in because
it takes another ~2.5 GiB of VRAM on top of ColQwen3 —
tight-8-GB hosts should leave it off.

### Configuration files

It is possible to set the defaults per directory via the config file. Ragrag will look for `ragrag.json` or `.ragrag.json` in the current working directory in that order; if not found, it will climb directory tree until one is found. All fields are optional.

```json
{
  "index_path": ".ragrag",
  "model_id": "TomoroAI/tomoro-colqwen3-embed-4b",
  "max_visual_tokens": 4096,
  "top_k": 10,
  "max_top_k": 50,
  "pdf_dpi": 250,
  "ocr_threshold": 50,
  "chunk_size": 900,
  "chunk_overlap": 200,
  "include_hidden": false,
  "follow_symlinks": true,
  "indexing_timeout": 100000
}
```

#### Field Descriptions

- **`index_path`**: Directory where the vector index and metadata are stored. Defaults to `.ragrag`. If a relative path is provided, it's resolved relative to the configuration file location.
- **`model_id`**: The HuggingFace model identifier for the ColQwen3 embedding model. Defaults to `TomoroAI/tomoro-colqwen3-embed-4b`.
- **`max_visual_tokens`**: Maximum number of visual tokens processed per image by the embedding model. Higher values capture more visual detail but use more GPU or CPU memory and increase embedding time.
- **`top_k`**: The default number of search results to return for each query.
- **`max_top_k`**: The maximum value that `top_k` can be set to. Requests for more results are capped at this value.
- **`pdf_dpi`**: Resolution in dots per inch used when rendering PDF pages into images for multimodal indexing. Higher DPI improves detail for small text and diagrams but increases processing time.
- **`ocr_threshold`**: Minimum character count for native PDF text before OCR fallback is skipped. Pages with fewer characters than this threshold are re-processed with Tesseract OCR to ensure visual content is indexed.
- **`chunk_size`**: Target size for text chunks in characters. Large documents are split into these chunks to fit within the model context window.
- **`chunk_overlap`**: The number of characters that overlap between consecutive text chunks. This ensures content spanning a chunk boundary is captured in both adjacent chunks, improving search recall for concepts that straddle chunk edges. A value of 0 means no overlap, which risks missing boundary content. 100-200 is typical.
- **`include_hidden`**: Whether to include hidden files and directories, those starting with a dot, during indexing. Defaults to `false`.
- **`follow_symlinks`**: Whether to follow symbolic links when discovering files. Default is true, meaning symlinks are followed. Set to false to prevent following symlinks and avoid cycles in recursive directory structures.
- **`indexing_timeout`**: Soft timeout in seconds for the full indexing phase. When elapsed, remaining files are skipped. Default is 100000, which is effectively unlimited for normal use. Set lower for time-bounded operations.

## Architecture

Ragrag is a single Python package (`ragrag/`) that runs entirely
locally. There is no network at query time and no cloud fallback.
The pipeline has five stages:

```
┌──────────────┐   ┌─────────────┐   ┌────────────┐   ┌───────────┐   ┌────────────┐
│ 1. discovery │──▶│ 2. planning │──▶│ 3. embed   │──▶│ 4. index  │──▶│ 5. retrieve│
│ walk paths   │   │ VLM chunker │   │ ColQwen3   │   │ mmap      │   │ MaxSim +   │
│ (SHA-based   │   │ (subprocess)│   │ multivec   │   │ vector    │   │ optional   │
│  staleness)  │   │             │   │ per topic  │   │ store     │   │ VLM rerank │
└──────────────┘   └─────────────┘   └────────────┘   └───────────┘   └────────────┘
```

Stages 1-4 run once per file SHA change; stage 5 runs on every
query. The whole thing is orchestrated from two entry points:

- **`ragrag/cli.py`** — argparse CLI. Auto-spawns the daemon when
  available, falls back to in-process mode when not. Logs go to
  stderr, results to stdout. Sets `HF_HUB_OFFLINE=1` before torch
  is imported so a flaky network never blocks startup.
- **`ragrag/daemon/server.py`** — long-lived JSON-RPC server over
  a Unix socket at `<index>/.ragrag/daemon.sock`. Holds one
  `SearchEngine` per index root, owns the loaded models, serves
  the HTTP dashboard on a separate thread.

### The unit of retrieval is a topic, not a page

The most consequential design decision: a **chunk is a topic**, a
semantic unit the VLM chunker identifies from the rendered page
images + native PDF text. A topic can reference multiple pages,
possibly non-contiguous, and two topics can reference the same
page when the content genuinely belongs to both.

```python
class Chunk(BaseModel):
    chunk_id: str                      # UUIDv4
    path: str                          # absolute source file path
    file_sha256: str
    kind: ChunkKind                    # pdf_topic | text_topic | image
    title: str                         # VLM-supplied, e.g. "ADC sampling time"
    summary: str                       # one-sentence VLM summary
    page_refs: list[int] = []          # 1-indexed page numbers (PDF topics)
    line_ranges: list[tuple[int,int]] = []  # (text topics)
    hero_page: int | None = None       # representative page for rerank prompts
    excerpt: str                       # ≤2 KiB of display text
    order_key: int                     # monotone within a file
```

Each topic is stored as **two points** in the vector store — a
text-modality point (the topic's consolidated text, embedded once
via `ColQwenEmbedder.embed_text_chunks`) and an image-modality
point (per-page image embeddings for every referenced page,
concatenated). Both points share the same `chunk_id` in their
payloads. At search time the engine rolls up duplicate
`chunk_id`s keeping the higher-scoring modality, so one topic
appears at most once in the final top-k regardless of which
modality matched.

Overlap has a cost: a page referenced by two topics gets embedded
twice. Indexing wall-clock grows sub-linearly with the overlap
rate and stays correct even when the VLM emits two topics that
happen to share half their pages. The storage format
(`ragrag/index/qdrant_store.py` with `offsets.bin`) already handles
variable-length multivectors, so a 50-page topic is just a
bigger row.

### Subprocess isolation for VLMs

The central infrastructure pattern. Any time the project loads a
VLM (topic chunker, text segmenter, reranker) it happens in a
child Python process, never in the main CLI / daemon process.
The reason, which is documented in capital letters in `CLAUDE.md`
and has been rediscovered twice:

> **bnb 4-bit leaves non-PyTorch CUDA context state behind after
> unload. Attempting to unload-and-reload a quantized model
> in-process fragments the CUDA allocator and the next forward
> pass silently OOMs or returns garbage. The only way to fully
> reclaim that state is for the child process to exit.**

Concretely:

- **`ragrag/extractors/vlm_topic_worker.py`** is the chunker/text
  segmenter worker. The parent passes it a JSON request over
  stdin (list of files + settings), it loads Qwen2.5-VL-3B once,
  emits one JSONL `Chunk` per line on stdout, then exits.
  `SubprocessVLMPlanner` (`vlm_topic_subprocess.py`) is the thin
  parent-side wrapper that `subprocess.run`s the worker and
  collects chunks.
- **`ragrag/retrieval/vlm_rerank_worker.py`** is the rerank
  worker. A **persistent** line-framed JSON-RPC over stdin/stdout:
  `{"cmd":"ping"}`, `{"cmd":"rerank",...}`, `{"cmd":"shutdown"}`.
  The parent keeps one worker alive per daemon session and
  serialises calls under a lock.
- **`ragrag/embedding/colqwen_embedder.py`** uses `defer_load=True`
  so the CLI can construct the handle before the planner runs and
  only load weights after the VLM subprocess has exited. That
  way the embedder always comes up in a pristine CUDA context.

### Query-time order of operations (with Phase D reranker armed)

The search pipeline in `ragrag/retrieval/search_engine.py` is:

```
Phase 0 — reranker prewarm  (if reranker_model="vlm")
          spawn vlm_rerank_worker, wait for {"status":"ready"}
Phase 1 — lazy ingest
          walk paths → stale detection → planner subprocess →
          embedder ensure_loaded → per-topic embed + upsert
Phase 2 — query embedding
          embedder.embed_query_text(query) → MultiVector
Phase 3 — retrieval
          qdrant_store.search(query_vec, top_k * oversample)
Phase 4 — rollup + format
          dedupe by chunk_id; build SearchResult objects;
          attach Location block and context_pages from page cache
Phase 5 — optional rerank
          send {"cmd":"rerank",...} to the already-running worker;
          parse response; reorder; stamp rerank_reason; truncate to top_k
```

Phase 0 runs **before Phase 1** on purpose. The Phase 1 ingest
manager is what first loads ColQwen3 into the parent process, and
once bnb 4-bit has initialised its CUDA context the parent pins
~3.5 GiB of VRAM that `unload()` cannot release in-process. If the
rerank worker has to load *after* that, it only sees ~40 MiB free
on an 8 GiB card and OOMs. Prewarming first lets the worker's
VLM claim its ~2.5 GiB slice while the parent process is still
torch-virgin. This is what `VLMReranker.prewarm()` exists for.

### Memory budget on 8 GiB cards

The reference GPU profile is an RTX 3050 8 GB with a busy X11
desktop. Baseline:

| Component                          | VRAM  |
|------------------------------------|------:|
| X11 / browser baseline             | ~1.7 GiB |
| ColQwen3-4B @ bnb 4-bit            | ~2.5 GiB |
| ColQwen3 activations (fwd pass)    | ~0.8 GiB |
| Qwen2.5-VL-3B @ bnb 4-bit          | ~2.5 GiB |
| Qwen2.5-VL-3B rerank activations   | ~0.8 GiB |
| **Total peak if both resident**    | **~8.3 GiB** |

That's ~0.5 GiB over the ceiling. When the reranker is armed,
the search engine sets `RAGRAG_EMBEDDER_DEVICE=cpu` so the
embedder loads on CPU and leaves the GPU to the rerank worker's
VLM. Text-only query embedding on CPU is ~30 s per call — slow,
but acceptable given the rerank forward pass is ~25 s on GPU, so
it was never the bottleneck.

Tighter cards can set `reranker_model="none"` and keep the
embedder on GPU for fast query embedding at the cost of
MaxSim-only ranking.

`ragrag/embedding/vlm_loader.py` walks a factory chain
(`AutoModelForImageTextToText → Vision2Seq → CausalLM →
AutoModel`) so custom-config VLMs still load, pins loaded models
to `device_map={"":"cuda:0"}` to bypass `accelerate`'s
undercounting of free VRAM on tight cards, and forces
`attn_implementation="eager"` so cuDNN kernel lookups never
fail at bf16.

### Storage, caching, and location

- **`ragrag/index/qdrant_store.py`** — custom mmap-backed vector
  store. `offsets.bin` indexes variable-length multivector rows;
  `vectors.bin` holds the raw bytes. A 140 MiB topic
  multivector is just a big row — mmap reads it page-by-page on
  demand, RSS never spikes.
- **`ragrag/index/page_cache.py`** — SHA-addressed WebP cache of
  rendered PDF pages at `<index>/page_cache/<sha[:2]>/<sha>/<n>.webp`.
  LRU-evicted at `page_cache_max_mb` (default 1 GiB). Written by
  the ingest manager as pages are rendered; served by the daemon
  dashboard at `/pages/<sha>/<n>.webp` and consumed at search
  time to populate `SearchResult.context_pages`.
- **`ragrag/file_state.py`** — per-file SHA-256 tracker at
  `<index>/file_state/`. Staleness detection is purely
  content-based, so moving a file is free and timestamp-based
  touch-without-change is free too.
- **`ragrag/retrieval/location_builder.py`** — on every search
  hit, builds a `Location` block: path, containing directory,
  head+tail directory listing capped at
  `location_directory_listing_max`. This is the *only* "related
  files" information a hit carries — by design. The project
  explicitly does not build a header/impl or import graph; the
  filesystem hierarchy is the signal, and the LLM consumer can
  follow up with another ragrag query if it wants to open
  something nearby.

### Daemon and dashboard

- **JSON-RPC over Unix socket** at `<index>/.ragrag/daemon.sock`,
  one request per connection, line-framed. The CLI
  (`ragrag/daemon/client.py`) auto-spawns the daemon on
  `ECONNREFUSED`, polls for the socket for up to 60 s, and falls
  back to in-process mode if the daemon can't start (sandboxed
  environments, `RAGRAG_NO_DAEMON=1`, `--no-daemon`).
- **Idle timeout 12 h** (`daemon_idle_timeout_s`), then the daemon
  unloads everything and exits. The next CLI call spins up a
  fresh one.
- **Serialised queries** — one request at a time under a
  `threading.RLock`; GPU can't meaningfully parallelise forward
  passes anyway.
- **HTTP status server** on a separate thread (default
  `http://127.0.0.1:27272/`, port recorded in `daemon.pid`).
  `/status` returns a JSON snapshot (uptime, models loaded, VRAM
  used, recent queries). `/pages/<sha>/<n>.webp` streams page
  images from the cache. `/` serves a single static HTML file
  that polls `/status` every 2 s. The status server is
  read-only and never touches the RPC lock.

### Phase D reranker status (as of 2026-04)

The Phase D VLM listwise reranker is **implemented and tested**
but defaults to **off**. On the STM32H743VI reference benchmark
the best reranker we could fit (Qwen2.5-VL-3B at bnb 4-bit, same
model as the chunker) **regresses** `semantic_at_5` by -0.08 and
costs +77 s per query vs MaxSim-only, because its listwise
ranking confuses adjacent parameter tables (it flips "ADC INL"
from "ADC accuracy" to "DAC accuracy" on a dense datasheet).
Full numbers at `validation/benchmarks/phase-D.json`.

The infrastructure (prewarm hook, subprocess worker, require-GPU
flag, JSON salvage parser, embedder-to-CPU override) all works
and ships. Operators can still opt in via
`"reranker_model": "vlm"` in `ragrag.json` if it helps on their
own corpus. Better levers for a future attempt:

- bigger reranker (Qwen2.5-VL-7B-AWQ) on a card with more
  headroom than 8 GiB
- pointwise per-candidate scoring instead of listwise (smaller
  activation cost, fewer degrees of freedom for the model to
  confuse itself)
- text-only cross-encoder (BGE-reranker-v2-m3, ~500 MB) that
  coexists with ColQwen3 at near-zero VRAM cost

### Directory map

| Path | What lives here |
|---|---|
| `ragrag/cli.py` | argparse CLI, daemon auto-start, in-process fallback |
| `ragrag/config.py` | `Settings` pydantic model + `ragrag.json` discovery |
| `ragrag/models.py` | Shared pydantic contracts: `Chunk`, `SearchRequest/Response/Result`, `Location`, etc. |
| `ragrag/path_discovery.py` | Filesystem walk, extension/hidden-file filters, skips `ragrag.json` |
| `ragrag/file_state.py` | SHA-based staleness tracker |
| `ragrag/extractors/pdf_extractor.py` | PDF rendering (pdf_dpi) + OCR fallback |
| `ragrag/extractors/vlm_topic_client.py` | VLM prompt + response parser for PDF and text topic discovery |
| `ragrag/extractors/vlm_topic_chunker.py` | Rolling-window topic state machine for PDFs |
| `ragrag/extractors/vlm_topic_segmenter.py` | Same for text files |
| `ragrag/extractors/vlm_topic_worker.py` | **Subprocess entry point** — loads VLM, streams chunks over stdin/stdout |
| `ragrag/extractors/vlm_topic_subprocess.py` | Parent-side wrapper (`SubprocessVLMPlanner`) |
| `ragrag/embedding/colqwen_embedder.py` | ColQwen3 late-interaction embedder with `defer_load` + CPU-fallback auto-detect |
| `ragrag/embedding/vlm_loader.py` | Shared HF VLM loader (factory chain, `device_map={"":"cuda:0"}`, eager attention) |
| `ragrag/index/ingest_manager.py` | Plan → embed → upsert pipeline |
| `ragrag/index/qdrant_store.py` | mmap-backed vector store with variable-length multivectors |
| `ragrag/index/page_cache.py` | WebP-backed LRU page image cache |
| `ragrag/retrieval/search_engine.py` | Orchestrator: prewarm → ingest → query embed → retrieve → rollup → rerank |
| `ragrag/retrieval/reranker.py` | `VLMReranker` — persistent subprocess lifecycle, serialised rerank |
| `ragrag/retrieval/vlm_rerank_worker.py` | **Subprocess entry point** — loads VLM, handles rerank RPCs, returns JSON ranks |
| `ragrag/retrieval/location_builder.py` | `Location` block for every search hit |
| `ragrag/retrieval/result_formatter.py` | JSON / compact-json / markdown / markdown-rich output formats |
| `ragrag/daemon/server.py` | JSON-RPC daemon, idle timeout, signal handling |
| `ragrag/daemon/client.py` | `DaemonClient` + auto-spawn |
| `ragrag/daemon/http_status.py` + `static.py` | Dashboard HTTP server |
| `ragrag/mcp_server.py` | MCP tool adapter over stdio |

## Rationale

LLM coding agents working on embedded projects need fast semantic lookup over large local documentation sets:
- source trees
- PDF datasheets and reference manuals
- image files with text/diagrams

Traditional text-only RAG misses diagram-heavy content (clock trees, pin mux diagrams, timing plots, block diagrams). We need a single search system that can retrieve both textual and visual evidence.

### Why multimodal is mandatory

Electronics documents contain content that is not faithfully representable as plain text:
- schematics
- signal timing diagrams
- clock trees
- annotated block diagrams

Therefore, the MVP must index:
- text
- PDF pages as images
- standalone images

### Goals

- Single local process, works anywhere even without GPU (given enough RAM).
- No dependency on cloud inference providers.
- Best feasible retrieval quality for visually rich technical docs.
- Lazy indexing on first query and on change (index-on-demand).

### Non-goals

- Distributed indexing.
- Guaranteed real-time indexing of all filesystem changes.
- Cloud fallback.

## Development

```bash
# Download validation corpus (PDFs, fixtures)
python scripts/fetch_validation_data.py

# Run tests (no model required)
pytest tests/test_validation.py -v
```
