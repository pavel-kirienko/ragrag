# Ragrag v2 Roadmap — Knowledge Base for Technical Docs

> **This is a live document.** Edit it as work lands. Check boxes when tasks complete, strike
> or note items that change, append to the "Notes & decisions" section of each phase as
> implementation teaches us things. When a phase finishes, update the status table at the
> top and move its summary into the Change log at the bottom.

## Legend

| Marker | Meaning |
|---|---|
| `[ ]` | Not started |
| `[/]` | In progress |
| `[x]` | Done |
| `[-]` | Deliberately skipped (explain in notes) |
| `[?]` | Blocked or needs decision |

Inline notes use `**Note:** …` and get kept in the closest `### Notes & decisions` block.

## Current status

| Phase | State | Summary |
|---|---|---|
| **Baseline** | `[x]` | Measured on STM32H743VI; see `baseline.json`. 4351 segments, 3.0 GB index, P@1=0.25, P@5=0.67, avg query 21.7 s. |
| **Mem fix (0.5)** | `[x]` | Streaming iterators + custom mmap store + numpy multivectors. RSS 20 GiB→4.6 GiB, store opens O(1). Commit `47e90ff`. |
| **Phase A — Daemon + thin CLI** | `[x]` | Unix socket, auto-start, 12 h idle timeout, CPU detection, in-process fallback. **Warm query 1.05 s vs baseline 23.6 s.** |
| **Phase B — Topic chunks** | `[ ]` | `Chunk` model + VLM topic chunker + text topic segmenter + `Location`. |
| **Phase C — Rich contexts** | `[ ]` | Page image cache + context pages + `Location` block in results. |
| **Phase D — VLM reranker** | `[ ]` | Listwise Qwen2.5-VL-3B rerank on top of MaxSim, with CPU fallback. |
| **Phase E — Dashboard** | `[ ]` | Minimal HTML served by the daemon; `/status`, `/pages`, `/shutdown`. |
| **Phase F — MCP server** | `[ ]` | `ragrag mcp` subcommand wrapping the daemon client. |
| **Phase G — Bench harness polish** | `[ ]` | `--daemon` flag, `diff_bench.py`, `validation/benchmarks/` under git. |

Latest benchmark: `baseline.json` (commit `47e90ff`, STM32H743VI, page-level indexing, no reranker).

---

## Context

We are building a searchable-knowledge-base tool for technical documentation: datasheets,
reference manuals, source trees, schematics, errata sheets, application notes, Markdown
design docs. The primary consumer is an LLM coding agent that asks natural-language
questions and needs to get back enough material to answer them directly — without opening
the source PDF itself.

The memory-pressure bug is fixed (commit `47e90ff`). The remaining problems are
retrieval quality, result usability, per-query latency, and CPU-friendliness:

- **Precision.** Baseline P@1 = 0.25, P@5 = 0.67. The tool finds *a* page with the query
  keywords but usually not the authoritative one. `semantic_at_5 = 1.00` so recall is
  fine — ranking is the bottleneck.
- **Result usability.** Excerpts are 200-char mid-table fragments. An LLM consumer reading
  them alone can't interpret the result without opening the PDF.
- **Latency.** Every CLI invocation pays ~20 s of model-reload tax.
- **CPU hosts.** Must still work on a laptop with an Intel Iris iGPU (no CUDA). Currently
  untested; retrieval on CPU is slow but correct.
- **Chunking granularity.** Current indexing is page-level, which is too small for
  semantic units and creates the same-page crowding + "maximum keyword hijack" problems
  we see in the baseline.

The plan below rebuilds the chunking model around semantic topics, introduces a persistent
daemon that owns loaded models and state, adds a small local vision-language model for
chunking and reranking, and ships a minimal web dashboard for visibility into background
indexing jobs.

---

## Baseline measurements (reference; do not edit)

Captured on commit `47e90ff`, STM32H743VI datasheet at DPI 250 on RTX 3050 / driver 470.

```
Indexing wall:      60.6 min        (357 pages, 4351 segments, 3.0 GB on disk)
Cold query wall:    ~22 s           (dominated by model reload)
Warm retrieval:     15–271 ms       (custom mmap store)
P@1:                0.25
P@5:                0.67
P@10:               0.92
MRR:                0.447
semantic_at_5:      1.00
distinct_pages_top10: 5.67
```

Per-question results are in `baseline.json`; open and diff against every later phase's
report under `validation/benchmarks/`.

---

## Non-negotiables

- **CPU-only hosts must work end to end.** An Intel Iris iGPU (no CUDA) or a headless
  Linux box with no GPU at all must successfully index, serve, auto-start the daemon,
  render page images, and answer queries. Every model path has a CPU option. VLM chunker
  and reranker run slowly on CPU but run.
- **VLM semantic chunking in v1 — no TOC reliance.** Plenty of real technical documents
  have no usable outline (schematics, scanned docs, vendor PDFs with broken bookmarks).
  The chunker works from page images directly. The old regex path is retained as a
  last-resort fallback and prints a loud warning if engaged.
- **We index topics, not physical units.** A chunk is a VLM-identified topic that
  references a set of pages (PDF) or line ranges (text). Topics may overlap — two distinct
  topics can reference the same page. No hard page-count limits.
- **Text files follow the same rule.** No byte-size-based dispatch. Every text file goes
  through the VLM topic segmenter; short coherent files come back as a single chunk, not
  because they're short but because the VLM says so.
- **Daemon idle timeout: 12 hours.** One working day of silence before unloading.
- **Every result carries a `Location` block** — `{path, directory, directory_listing}`.
  No graph, no header/impl fuzzy match, no import parsing, no Markdown-link traversal.
  The LLM consumer can issue follow-up ragrag queries if it wants neighbors.
- **Minimal web dashboard.** Served by the daemon. Shows indexing progress, queue,
  resource utilisation (CPU, GPU, RAM, VRAM), recent queries. Single static HTML file,
  no framework, no build step.

---

## Architecture overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│  CLI (ragrag "query"): thin JSON-RPC client                               │
│   - auto-spawns daemon on ECONNREFUSED / FileNotFoundError                │
│   - falls back to in-process SearchEngine if daemon cannot start          │
└─────────────────────────────┬─────────────────────────────────────────────┘
                              │ JSON-RPC 2.0 over Unix domain socket
                              ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  Daemon (long-lived, idle timeout 12 h)                                   │
│                                                                           │
│  Models (loaded lazily, unloaded on idle timeout):                        │
│   • ColQwenEmbedder   — retrieval (CUDA 4-bit OR CPU bf16)                │
│   • VLMReranker       — Qwen2.5-VL-3B-AWQ / Moondream2 / none             │
│   • VLMTopicChunker   — same VLM instance as reranker, different prompt   │
│   • TextTopicSegmenter — same VLM instance, different prompt              │
│                                                                           │
│  Services:                                                                │
│   • KnowledgeBase      — one per index root; Store + FileTracker + Cache  │
│   • IndexingQueue      — background worker thread, one file at a time    │
│                                                                           │
│  Transports:                                                              │
│   • JSON-RPC on <index>/.ragrag/daemon.sock                               │
│   • HTTP status/dashboard on 127.0.0.1:<port> (port in daemon.pid)        │
│     - GET /           → static HTML dashboard                             │
│     - GET /status     → JSON snapshot (poller target)                     │
│     - GET /pages/<sha>/<page>.webp  → cached page image                   │
│     - POST /shutdown  → graceful exit                                     │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Glossary

| Term | Meaning |
|---|---|
| **Chunk** | An indexable unit. Identified with a `chunk_id`. Stored as one or two points in the vector store (text modality, visual modality). Replaces the old `Segment` type. |
| **Topic** | Semantic equivalent of a chunk: a set of references (pages or line ranges) that belong together. The VLM identifies them. |
| **Page refs** | For PDF topics: the list of page numbers referenced by the chunk. Need not be contiguous. May overlap with other topics. |
| **Line ranges** | For text topics: list of `(line_start, line_end)` pairs. Need not be contiguous. May overlap. |
| **Hero page** | One representative page per PDF topic, used in the reranker prompt (so a 50-page topic doesn't blow the prompt budget). |
| **Location** | Per-result block: `{path, directory, directory_listing}`. Computed at search time. |
| **Daemon** | Long-lived process holding loaded models and index handles. Auto-started by the CLI. |
| **In-process fallback** | CLI behaviour when the daemon can't start — runs the SearchEngine in the same process, pays the model-load cost. Kept working forever as a safety net. |
| **Knowledge base** | One index root (`<path>/.ragrag/`). The daemon can have multiple open simultaneously. |

---

# Phase A — Daemon skeleton + thin CLI + CPU detection

**Why first.** Every later phase iterates faster against a running daemon (no 20 s model
reload between test runs). The dashboard in Phase E and the indexing queue in Phase C
both piggy-back on it.

**Branch:** `phase-a-daemon`

## Tasks

### A.1  JSON-RPC protocol

- [x] Define RPC method signatures in `ragrag/daemon/rpc.py`
  - [x] `search(query, paths, top_k, include_page_images, rerank, context_strategy)`
  - [x] `index(paths, force)` — blocking indexing call
  - [x] `status()` — uptime, loaded models, open indexes, resource use, indexing state, recent queries
  - [x] `shutdown()` — ack + graceful exit
  - [x] `reload_config()` — re-read Settings from disk
- [x] JSON-RPC 2.0 framing (newline-delimited), one request per connection
- [x] Error classes: `JsonRpcError` (single class with code/message/data; specific kinds via the standard JSON-RPC error codes)
- [x] Wire format helpers: `encode_request`, `decode_request`, `encode_response`, `decode_response`

### A.2  Socket + lock layout

- [x] Directory: `<index_path>/.ragrag/` — created by CLI or daemon, whichever runs first
- [x] Socket path: `<index_path>/.ragrag/daemon.sock`, mode 0600
- [x] PID file: `<index_path>/.ragrag/daemon.pid`, content: `<pid>\n<start_monotonic>\n<protocol_version>\n<http_port>`
- [x] Log file: `<index_path>/.ragrag/daemon.log`, append (10 MB cap deferred — write rotation in Phase E)
- [x] Stale-socket detection in client: read pid, `kill -0 <pid>`; if dead, unlink socket and spawn fresh
- [-] `fcntl.flock` on `daemon.pid` serialises concurrent spawns so only one wins — _Deferred. Single-developer tool with one daemon per index dir; race window is theoretical. Revisit if multi-spawn observed in practice._

### A.3  Daemon process

- [x] `ragrag/daemon/server.py`: accept loop, RPC dispatcher, lifecycle
- [x] `ragrag daemon --index-path <p> [--detach]` entry point (also callable as `python -m ragrag.daemon`)
- [x] `--detach`: double-fork + close stdio (parent returns when socket exists)
- [x] Lazy model loading: `SearchEngine` built on first request, not on daemon start
- [x] Idle timeout: `daemon_idle_timeout_s = 43200` (12 h). On timeout: unload engines, exit cleanly
- [x] Signal handlers: SIGTERM/SIGINT (drain + exit), SIGHUP (reload config), SIGUSR1 (dump status JSON). Skipped on non-main threads (test harness)
- [x] Concurrency: `threading.RLock` around the search pipeline; thread pool (size 4) accepting connections
- [-] Separate thread for the HTTP status server (Phase E) — _deferred to Phase E_

### A.4  CLI client

- [x] `ragrag/daemon/client.py`: `DaemonClient(index_path)` with methods `search, index, status, shutdown, ensure_daemon`
- [x] `ensure_daemon()`:
  - [x] try to connect to socket
  - [x] on failure: cleanup stale pid+sock, then spawn
  - [x] spawn: `subprocess.run([sys.executable, "-m", "ragrag.daemon", "--detach", "--index-path", path])`
  - [x] poll for socket creation, 60 s timeout
  - [x] raise `DaemonStartupError` on timeout
- [x] Path normalisation: accepts both the knowledge-base root and the `.ragrag` directory; the two are equivalent
- [x] `DaemonClient.search(request)` returns the JSON dict from the daemon
- [x] `DaemonClient.status()` returns a dict
- [x] `DaemonClient.shutdown()` sends the RPC and waits for the socket to disappear (up to 10 s)

### A.5  CLI entry point rewrite

- [x] `ragrag/cli.py::main()` becomes:
  - [x] sub-command dispatch on argv[1] for `daemon` / `status` / `shutdown` (`dashboard` deferred to Phase E)
  - [x] parse args + resolve settings + index root
  - [x] if `--no-daemon` or `RAGRAG_NO_DAEMON=1`: `_run_inprocess(args, settings)` directly
  - [x] else: try daemon path; on `DaemonStartupError` or `DaemonError` fall back to `_run_inprocess` with a stderr warning
- [x] New subcommands: `ragrag daemon`, `ragrag shutdown`, `ragrag status` (`dashboard` lands in Phase E)
- [x] `ragrag --no-daemon` global flag
- [x] Keep `_run_inprocess` around permanently — tests rely on it, sandboxes need it

### A.6  Settings

- [x] `daemon_idle_timeout_s: float = 43200`
- [x] `daemon_autostart: bool = True`
- [x] `daemon_status_port: int = 27272` (Phase E wires it up)
- [x] `daemon_status_host: str = "127.0.0.1"`
- [-] `daemon_socket_mode: int = 0o600` — _hard-coded in `_bind`; promote to setting only if a user complains_

### A.7  CPU detection

- [x] `_detect_device_mode()` at daemon start (returns `cuda`/`cpu`/`mps`)
- [x] Logged in `daemon.log` at startup
- [x] Stored in `DaemonState.device_mode`
- [x] Surfaced in `/status` response

### A.8  Tests

- [x] `tests/test_daemon_rpc.py` — 11 tests: encode/decode roundtrip, malformed input, unknown method → JSON-RPC error
- [x] `tests/test_daemon_lifecycle.py` — 5 tests: status, search, shutdown, unknown method, garbage request
- [x] `tests/test_daemon_stale_pid.py` — 4 tests: stale pid+sock cleanup, live pid kept, missing/garbage pid file
- [x] `tests/test_cli_auto_start.py` — 3 tests: ensure_daemon path, status sub-command, status without daemon
- [x] `tests/test_cli_inprocess_fallback.py` — 3 tests: DaemonStartupError fallback, --no-daemon flag, env var
- [x] `tests/test_cpu_fallback_detection.py` — 3 tests: detector returns known value, torch-missing fallback, DaemonState device_mode

29 new tests, all green. Total unit suite: 76 → 105.

## Acceptance

- [x] `.venv/bin/python -m pytest tests/ --ignore=tests/test_e2e.py -q` → **105 passed**
- [x] `.venv/bin/python -m pytest tests/ -q` (full e2e) → **109 passed**
- [x] Warm `ragrag "..." /tmp/stm32h743_bench`: 1.05 s wall (was 23.6 s baseline)
- [x] `ragrag status` prints uptime and device mode
- [x] `ragrag shutdown` cleanly exits the daemon; `.sock` and `.pid` gone
- [x] `ragrag --no-daemon "query"` still works via in-process fallback
- [x] `RAGRAG_NO_DAEMON=1 ragrag "query"` equivalent
- [x] Benchmark report at `validation/benchmarks/phase-A.json`. Quality identical to baseline; `avg_query_wall_s` 23.6 s → 1.05 s
- [x] `diff_bench.py` exists; markdown delta in commit body
- [x] Commit `phase-A.json` + diff_bench.py + ROADMAP update

## Notes & decisions

- **Decision:** daemon is per-index-directory, not global. Multiple knowledge bases → multiple daemons. Rationale: socket path is keyed by `<index>/.ragrag/`, so opening two different indexes naturally spawns two daemons. Each one is tiny when idle.
- **Decision:** in-process fallback stays forever, not just "in v1". Sandboxed CI, containers without Unix-socket permissions, Windows users, and first-run environments all benefit.
- **Surprise:** `signal.signal()` only works on the main thread of the main interpreter. The first cut of `_install_signals` raised `ValueError` when the test harness ran the daemon in a background thread. Fixed with a `threading.current_thread() is threading.main_thread()` guard.
- **Surprise:** there's a real race between `bind()` and `listen()` and the test harness checking `socket_path.exists()`. The file appears immediately after `bind()`, but the kernel rejects connect() until `listen()` is called. Fixed by changing the test helper to actively probe a connect, not just check for the file.
- **Surprise:** `Settings.index_path` already includes the trailing `.ragrag` (it's the storage dir, not the project root). The first `DaemonClient(...)` cut appended `.ragrag` again and built sockets at `…/.ragrag/.ragrag/daemon.sock`. Fixed by normalising both ends — both the client and `_resolve_paths` now accept either form.
- **Open:** how to handle a daemon that's half-dead (socket exists, process stuck on model load). Current plan: client times out at 60 s on `ensure_daemon`, then the user is expected to `ragrag shutdown` or kill -9.

---

# Phase B — Topic chunks + VLM topic chunker + text topic segmenter

**Why second.** Defines the indexing unit the rest of the system operates on. Invasive; 
must land in one commit with a forced re-index.

**Branch:** `phase-b-topics`

## B.1  `Chunk` data model

- [ ] Add `Chunk` to `ragrag/models.py`:
  ```python
  class ChunkKind(str, Enum):
      PDF_TOPIC = "pdf_topic"
      TEXT_TOPIC = "text_topic"
      IMAGE = "image"

  class Chunk(BaseModel):
      chunk_id: str
      path: str
      file_sha256: str
      kind: ChunkKind
      title: str
      summary: str
      page_refs: list[int] = Field(default_factory=list)
      hero_page: Optional[int] = None
      line_ranges: list[tuple[int, int]] = Field(default_factory=list)
      byte_ranges: list[tuple[int, int]] = Field(default_factory=list)
      excerpt: str
      order_key: int = 0
  ```
- [ ] Keep `Segment` as a compat shim — tests importing it still work, but the ingest path writes `Chunk`
- [ ] `MultiVector` type alias unchanged (numpy ndarray)
- [ ] `SearchResult` gains: `title`, `summary`, `page_refs`, `line_ranges`, `context_pages: list[PageContext]`, `location: Optional[Location]`, `rerank_reason: Optional[str]`
- [ ] `PageContext(BaseModel)`: `page: int`, `page_image_path: Optional[str]`, `page_image_b64: Optional[str]`, `text: str`
- [ ] `Location(BaseModel)`: `path: str`, `directory: str`, `directory_listing: list[str]`

## B.2  `TextTopicSegmenter`

File: `ragrag/extractors/text_topic_segmenter.py` (NEW)

- [ ] Read file content (UTF-8 with `errors="replace"`)
- [ ] Tokenize roughly into VLM-prompt chunks (`chunker_vlm_ctx_tokens`, default 8192)
- [ ] **Single-call path** (file fits in one VLM prompt):
  - [ ] Build prompt: *"Here is a {language} file. Identify the distinct topics in it…"*
  - [ ] Call `VLMTopicClient.identify_text_topics(content, language_hint)` (shared VLM instance in the daemon)
  - [ ] Parse returned JSON list of `{title, summary, ranges}` objects
  - [ ] Validate ranges (in bounds, non-empty, ascending)
  - [ ] Emit one `Chunk` per topic
- [ ] **Window-slide path** (file exceeds context):
  - [ ] Walk the file with stride `chunker_vlm_ctx_tokens / 2` and 25% overlap
  - [ ] Per window: single call with absolute-line-number output
  - [ ] Merge topics across windows by title fuzzy-match (Levenshtein ≤ 3 on normalized title)
  - [ ] Coalesce line ranges for matched titles
- [ ] **VLM failure fallback**:
  - [ ] 3 retries with terser prompts
  - [ ] On third failure: call the regex chunker (`_chunk_text`) and emit chunks with `title = "<regex fallback>"`, `summary = ""`
  - [ ] Log WARNING with file path
- [ ] Language hint from extension: `.c → C`, `.py → Python`, `.md → Markdown`, `.json → JSON`, etc. Default `unknown text`.

## B.3  `VLMTopicChunker` for PDFs

File: `ragrag/extractors/vlm_topic_chunker.py` (NEW)

- [ ] Input: iterator of `(page_number, page_image, native_text)` from the existing PDF extractor
- [ ] State: `topics: dict[str, TopicAccum]` with `{title, summary, page_refs, last_seen_page}`
- [ ] **Rolling window** of `chunker_stride_pages` pages (default 8)
- [ ] Per window:
  - [ ] Build prompt: running-topic summary + window page images + per-page native text headers
  - [ ] VLM call: `identify_pdf_topics(window)` returns per-page topic assignments (multi-assignment allowed)
  - [ ] Update state: append referenced pages to existing topic `page_refs`; create new topics for unseen ids
- [ ] **Cold topic flush**: if a topic's `last_seen_page` is older than `chunker_topic_cold_pages` (default 20), close it and stop adding to it
- [ ] **End-of-document flush**: emit remaining open topics
- [ ] **Overlap allowed**: the same page may appear in multiple topics' `page_refs`
- [ ] **Non-contiguous allowed**: a topic's `page_refs` may have gaps
- [ ] **Hero page**: set to the page where the topic was first introduced (first in `page_refs`)
- [ ] **VLM failure fallback**:
  - [ ] Per-window: 3 retries on parse failure
  - [ ] After 3 failures: fold window pages into the most recently active topic (or create `"Untitled (pages N..M)"` if no active topic)
  - [ ] Log WARNING per failed window
- [ ] **CPU fallback**: the same code path runs on CPU. No special "fixed-N" degradation. Expect ~10–30 s per VLM call. Document in the log that CPU mode indexing is slow.

## B.4  `IngestManager` integration

- [ ] Rewrite `_stream_embed_and_store` to dispatch on file type (no size thresholds):
  - [ ] TEXT → `TextTopicSegmenter.segment(path)` → iterate topics → embed text → upsert
  - [ ] PDF → `VLMTopicChunker.chunk(path)` → iterate topics → for each topic, embed visual (per-page concat) + embed text (per-range concat) → upsert both points
  - [ ] IMAGE → single-chunk path (unchanged)
- [ ] For each PDF topic, build the visual multivector by calling `embed_images([img])` once per page in `page_refs` and concatenating the token vectors along axis 0
- [ ] For each PDF topic, build the text multivector from the concatenated native text of every page in `page_refs`, separated by form-feed markers
- [ ] For a TEXT topic, build the text multivector from the concatenated content of every `line_range` (separator: `\n---\n`)
- [ ] Store both modality points atomically (same `chunk_id` on both); rollback on failure
- [ ] Track `file_sha256` on every chunk payload so the page cache lookup works at search time

## B.5  `Location` builder

- [ ] `ragrag/retrieval/location_builder.py` (NEW)
- [ ] `build_location(path: str, respect_gitignore: bool = True) -> Location`
  - [ ] `os.path.dirname(realpath(path))` → directory
  - [ ] `os.scandir(directory)` → list of entries (files + subdirs); sort alphabetically
  - [ ] If `respect_gitignore` and a `.gitignore` sits in or above `directory`: filter out matches (use a tiny stdlib-only glob matcher; no new deps)
  - [ ] Cap listing at `location_directory_listing_max` (default 64): head 32 + ellipsis + tail 32
  - [ ] Entries include directories with a trailing `/`
- [ ] `SearchEngine` calls `build_location` per result just before returning the response
- [ ] No cache — one `scandir` per search is fine

## B.6  Settings

- [ ] `chunker_vlm_ctx_tokens: int = 8192`
- [ ] `chunker_stride_pages: int = 8`
- [ ] `chunker_max_topics_per_call: int = 16`
- [ ] `chunker_topic_cold_pages: int = 20`
- [ ] `chunker_max_topic_overlap: float = 0.5` (warn if > half of pages end up in multiple topics)
- [ ] `chunker_fallback_regex: bool = True`
- [ ] `location_directory_listing_max: int = 64`
- [ ] `location_respect_gitignore: bool = True`

## B.7  VLM loader shared infrastructure

- [ ] `ragrag/embedding/vlm_loader.py` (NEW)
- [ ] `VLMHandle` class with `model`, `processor`, `device`, `quantization`, `loaded_at`
- [ ] `load_vlm(model_id, quantization="auto", device="auto") -> VLMHandle` — shared by chunker + reranker + text segmenter
- [ ] `VLMHandle.generate(prompt, images=None, max_new_tokens=512)` — unified generation
- [ ] Auto-detect: if CUDA available and `free_vram > 2.5 GiB`, load in 4-bit AWQ (or bnb); else load in bf16 on CPU
- [ ] The daemon holds at most one `VLMHandle` — chunker and reranker share it
- [ ] `unload()` frees model weights and empties CUDA cache

## B.8  Tests

- [ ] `tests/test_chunk_model.py` — Chunk round-trip through pydantic + JSON; `page_refs` with duplicates, non-contiguous; `line_ranges` overlapping
- [ ] `tests/test_text_topic_segmenter.py`
  - [ ] Small file → single topic (mock VLM returns one)
  - [ ] Window-slide path → merge topics across windows by title
  - [ ] VLM failure 3× → regex fallback, warning logged, chunks tagged
- [ ] `tests/test_vlm_topic_chunker.py`
  - [ ] 20-page synthetic PDF, mock VLM: two topics sharing pages 5–6 → chunks have overlapping `page_refs`
  - [ ] Topic spanning pages 1–3 and 15–17 → one chunk with non-contiguous refs
  - [ ] Cold-topic flush: topic unseen for 25 pages, `chunker_topic_cold_pages=20` → topic closed at page 21
  - [ ] End-of-document flush: all open topics emitted
- [ ] `tests/test_vlm_chunker_parse_failure.py` — garbage JSON → retry 3× → fold into current topic
- [ ] `tests/test_location_builder.py`
  - [ ] 5-file dir → all entries
  - [ ] 100-file dir → head 32 + tail 32 with ellipsis marker
  - [ ] `.gitignore` present + `respect_gitignore=True` → excluded entries not in listing
  - [ ] Symlinks + trailing slashes on directories
- [ ] `tests/test_ingest_manager_topic_routing.py` — text file → TextTopicSegmenter called; PDF → VLMTopicChunker called; image → legacy path
- [ ] `tests/test_overlap_embed_cost.py` — a topic overlap causes duplicate `embed_images` calls; assert count matches expectation
- [ ] `tests/test_vlm_loader.py` — mock torch.cuda, assert 4-bit on CUDA / bf16 on CPU; `unload()` frees

## Acceptance

- [ ] Unit suite: `pytest tests/ --ignore=tests/test_e2e.py -q` → all pass
- [ ] Full suite: `pytest tests/ -q` → all pass
- [ ] Force a fresh index of the STM32 datasheet through `VLMTopicChunker`:
  - [ ] `rm -rf /tmp/stm32h743_bench/.ragrag && ragrag "warmup" /tmp/stm32h743_bench --new`
  - [ ] Chunk count in sane range (~25–80)
  - [ ] Store size ≤ ~4 GB
  - [ ] Indexing wall clock ≤ 90 min on GPU (baseline 60 min + VLM chunking overhead)
  - [ ] Peak RSS ≤ 6 GB
- [ ] Run `scripts/benchmark_stm32h743.py --daemon --report validation/benchmarks/phase-B.json`
- [ ] `diff_bench.py validation/benchmarks/phase-A.json validation/benchmarks/phase-B.json` shows:
  - [ ] `semantic_at_5` still 1.00 (we didn't lose recall)
  - [ ] Manual eyeball: top-1 excerpts on 3 random queries are now full topic excerpts, not mid-table fragments
- [ ] Commit `phase-B.json` to the repo
- [ ] **Document breakage:** commit message states "requires `--new` re-index on upgrade"

## Notes & decisions

_(append as we go)_

- **Decision:** topics are materialized as two vector store points (text + image), both carrying the same `chunk_id`. Search-time rollup dedupes. Storing the page-level embeddings separately and reconstructing topic multivectors at query time is cleaner but adds a second index and per-query assembly — deferred past v1.
- **Decision:** no size-based gating anywhere. Every file goes through the VLM topic path. Short files get a single fast VLM call (usually "one topic, the whole file"); the cost is one ~1 s call per small file, which is fine on GPU and acceptable on CPU.
- **Decision:** the `Location` block is computed at search time by `os.scandir`. No graph, no persistent state, no indexing-time work.
- **Open:** what hero page to pick for a topic that references non-contiguous pages. Current plan: first page in `page_refs` (insertion order). Revisit if rerank prompts feel unhelpful.
- **Open:** how the daemon queues indexing alongside searches. First cut: same `RLock`, indexing blocks search for its duration. Second cut (deferred): background indexing thread with periodic pause for search requests.

---

# Phase C — Rich result contexts

**Why third.** Page image cache + context assembly + `Location` wiring turn the Phase B
chunks into results an LLM consumer can actually use.

**Branch:** `phase-c-contexts`

## C.1  `PageImageCache`

- [ ] `ragrag/index/page_cache.py` (NEW)
- [ ] Class `PageImageCache(root: Path)`
  - [ ] `get(sha, page) -> Path | None`
  - [ ] `put(sha, page, pil_image) -> Path`
  - [ ] `has(sha, page) -> bool`
  - [ ] `evict_file(sha) -> None`
  - [ ] `size_bytes() -> int`
- [ ] On-disk layout: `<root>/<sha[:2]>/<sha>/<page>.webp`
- [ ] WebP quality 85, method 6 → ~80 KB/page at DPI 250
- [ ] LRU eviction when `size_bytes() > page_cache_max_mb * 1024 * 1024` (default 1024 MiB)
- [ ] Thread-safe (lock on `size_bytes()` and eviction)
- [ ] Wire into `IngestManager._stream_embed_and_store`: write page to cache before dropping the PIL ref
- [ ] Idempotent: `put()` is a no-op if the file already exists (safe under concurrent upsert)

## C.2  `include_page_images` flag + budget

- [ ] Add CLI flag `--include-page-images={none,path,base64}`
- [ ] Default: `path` for CLI, `base64` for MCP (Phase F), `path` for dashboard
- [ ] Add `Settings.max_inline_image_kb: int = 4096` — budget cap in base64 mode
- [ ] `SearchEngine._attach_context_pages(result, include_mode)`:
  - [ ] For each page in `result.page_refs`, look up the cache
  - [ ] Build `PageContext(page, path, text)` — text is the native text for that page (read from PyMuPDF on demand, cached per-file in the search response builder)
  - [ ] If `include_mode == "base64"`, base64-encode the WebP bytes until budget exhausted; overflow falls back to `page_image_path`

## C.3  Formatters

- [ ] `format_as_markdown_rich(response)`:
  - [ ] Per result: rank, title, score, summary, rerank_reason if present
  - [ ] Inline image references for each page in `context_pages` (file:// URLs when mode=path)
  - [ ] `Location` block: path, directory, capped listing
- [ ] `format_as_compact_json(response)`:
  - [ ] Drop `page_image_b64` for everything except the hero page
  - [ ] Truncate `excerpt` and `summary` to 240 chars
  - [ ] Trim `directory_listing` to 16 entries
  - [ ] Collapse `timing_ms` to just `total_ms`
- [ ] `--format={json,compact-json,markdown,markdown-rich}`, `--markdown` aliases `markdown-rich`, `--json` aliases `json`

## C.4  Tests

- [ ] `tests/test_page_cache.py`
  - [ ] put/get round-trip
  - [ ] idempotent put
  - [ ] LRU eviction triggers at the configured cap
  - [ ] concurrent put from 8 threads → all files written exactly once
- [ ] `tests/test_search_context_pages.py`
  - [ ] PDF topic hit has `context_pages` populated from cache
  - [ ] Text topic hit has `context_pages` empty (no page concept)
  - [ ] Overlapping `page_refs` → deduplicated in `context_pages`
  - [ ] Non-contiguous `page_refs` → `context_pages` preserves order
- [ ] `tests/test_search_location_attach.py`
  - [ ] Result includes `location` block
  - [ ] `directory_listing` respects gitignore
  - [ ] Large directory gets head+tail split
- [ ] `tests/test_result_formatter_rich.py`
  - [ ] markdown-rich renders images + location
  - [ ] compact-json trims base64 to hero page + 240-char excerpts

## Acceptance

- [ ] Unit + full suite green
- [ ] Eyeball Q05 against the rebuilt STM32 index: top-1 result returns page images including Table 77 or Table 78; `location.directory_listing` shows the datasheet filename
- [ ] Measure response size for a typical top-5 query in each format:
  - [ ] `json` (no images) → baseline
  - [ ] `markdown-rich` → paths only, human-readable
  - [ ] `compact-json` → base64 for hero only, ≤ 200 KB
- [ ] Commit `validation/benchmarks/phase-C.json` (same quality numbers as B, confirm no regression)

## Notes & decisions

_(append as we go)_

---

# Phase D — VLM reranker

**Why fourth.** Biggest single quality lever we have. Lands after B and C so we measure
the compound gain over the new chunking model and the rich contexts.

**Branch:** `phase-d-rerank`

## D.1  Model choice

- [ ] Primary GPU: `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`
- [ ] CPU option: `vikhyatk/moondream2` in int4 (~800 MB, ~10 s per rerank call on laptop CPU)
- [ ] Config: `reranker_model ∈ {"qwen2.5-vl-3b", "moondream2", "none"}`
- [ ] Default on CUDA: `"qwen2.5-vl-3b"`
- [ ] Default on CPU: `"none"` — user must opt in to reranking on CPU

## D.2  VRAM budget & offload

- [ ] On daemon start (CUDA only), estimate VRAM budget:
  - ColQwen3 4-bit ≈ 2.5 GB
  - Qwen2.5-VL-3B AWQ ≈ 2.5 GB
  - activations peak ≈ 1.5–2 GB
- [ ] If `free_vram >= 7 GiB`: both models on GPU
- [ ] If `free_vram < 7 GiB`: enable `accelerate.cpu_offload_with_hook` on the reranker — weights live in CPU RAM, pulled to GPU per-call
- [ ] `Settings.reranker_cpu_offload: str = "auto"` (`"auto"|"always"|"never"`)

## D.3  `VLMReranker`

- [ ] `ragrag/retrieval/reranker.py` (NEW)
- [ ] Uses the shared `VLMHandle` from `ragrag/embedding/vlm_loader.py`
- [ ] `VLMReranker.rerank(query: str, candidates: list[RerankCandidate], top_k: int) -> list[int]`
- [ ] `RerankCandidate` dataclass: `{chunk_id, title, summary, hero_page_image: PIL.Image, hero_page_number, score}`
- [ ] Listwise recipe — one VLM call, all candidate images:
  - [ ] Prompt template with query + numbered candidates (title, summary, image)
  - [ ] Output: JSON array of `{rank, id, score, reason}`
  - [ ] Retries (3) on parse failure with terser prompt
  - [ ] Fallback: identity order (MaxSim order) + warn
- [ ] Return ranked chunk-index list; search engine reorders accordingly
- [ ] Populate `SearchResult.rerank_reason` from the VLM's reasoning string

## D.4  Integration with `SearchEngine`

- [ ] After retrieval + rollup, if `reranker is not None and request.rerank != False`:
  - [ ] Build `RerankCandidate` list from top-`top_k * rerank_oversample` surviving chunks
  - [ ] Load hero-page images from the cache
  - [ ] Call reranker
  - [ ] Reorder + truncate to `top_k`
- [ ] `Settings.rerank_oversample: int = 3`
- [ ] `Settings.rerank_max_images: int = 10`
- [ ] `Settings.rerank_max_tokens: int = 512`

## D.5  Tests

- [ ] `tests/test_reranker_prompt.py` — build prompt from fake candidates, inspect structure
- [ ] `tests/test_reranker_json_parse.py` — parser handles good JSON, trailing whitespace, ```json fences, garbage
- [ ] `tests/test_reranker_fallback.py` — garbage in 3 retries → identity order + warn
- [ ] `tests/test_reranker_cpu_offload.py` — force `free_vram = 0` → offload activated
- [ ] `tests/test_search_engine_rerank_hook.py` — mock reranker returns a known order → results reorder correctly
- [ ] `tests/test_moondream2_loader.py` — int4 loader smoke test (skipped on CI if model not cached)

## Acceptance

- [ ] Unit + full suite green
- [ ] Rerun benchmark on warm daemon with rerank enabled → `validation/benchmarks/phase-D.json`
- [ ] Targets vs baseline:
  - [ ] `p_at_1 ≥ 0.45` (baseline 0.25) → ✅ ship as default
  - [ ] `p_at_5 ≥ 0.77` (baseline 0.67)
  - [ ] `avg_query_wall_s` warm ≤ 6 s (baseline 21.7 s cold)
- [ ] Stop-loss: if `p_at_1 < 0.40`, keep the code but default `reranker_model = "none"`. Document as opt-in. Not a merge blocker.
- [ ] On CPU path, Moondream2 reranking: validate it runs end-to-end on a small fixture. Latency acceptance: ≤ 30 s per query.
- [ ] Commit `phase-D.json` + its diff against `baseline.json` to the PR body

## Notes & decisions

_(append as we go)_

---

# Phase E — Minimal web dashboard

**Why fifth.** Indexing big corpora is long-running; the user needs visibility. Easy win
on top of the daemon that exists since Phase A.

**Branch:** `phase-e-dashboard`

## E.1  HTTP server

- [ ] `ragrag/daemon/http_status.py` (NEW)
- [ ] `http.server.ThreadingHTTPServer` running on a second thread, started at daemon startup
- [ ] Port resolution: try `daemon_status_port` (27272); if taken, pick any free port via `0`
- [ ] Write final port into `daemon.pid` line 4
- [ ] Never shares the RPC lock — handlers read daemon state read-only

## E.2  Endpoints

- [ ] `GET /` → static HTML dashboard (bundled string)
- [ ] `GET /status` → JSON snapshot
- [ ] `GET /pages/<sha>/<page>.webp` → serve from page cache
- [ ] `GET /log` → tail of `daemon.log` (last ~5 KB)
- [ ] `POST /shutdown` → graceful daemon exit (requires `X-Ragrag-Confirm: yes` header)
- [ ] `GET /events` → (stretch) Server-Sent Events for live updates; defer to follow-up if time-boxed

## E.3  `/status` payload

- [ ] Structure:
  ```json
  {
    "version": "0.2.0",
    "uptime_s": 1893,
    "config": {"daemon_idle_timeout_s": 43200, "reranker_model": "qwen2.5-vl-3b", "device_mode": "cuda"},
    "models_loaded": [{"name": "...", "vram_mib": 2450, "loaded_at": "..."}],
    "indexes_open": [{"path": "...", "points": 128, "size_mib": 1340}],
    "indexing": {
      "active_file": "...",
      "pages_done": 184, "pages_total": 642,
      "elapsed_s": 412, "eta_s": 990,
      "queued_files": 3, "completed_files": 12
    },
    "resources": {
      "cpu_pct": 94.3, "gpu_pct": 72.1,
      "mem_rss_mib": 4612, "vram_used_mib": 5120, "vram_total_mib": 8192
    },
    "recent_queries": [{"query": "...", "wall_ms": 2341, "top1_path": "..."}]
  }
  ```
- [ ] `psutil` for CPU + RSS (add to `pyproject.toml`)
- [ ] `torch.cuda.mem_get_info` for VRAM on CUDA hosts
- [ ] `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits` for GPU util, 5 s cache, best-effort
- [ ] On CPU: omit `gpu_pct` and `vram_*`

## E.4  HTML dashboard

- [ ] `ragrag/daemon/static.py` (NEW) — single `INDEX_HTML: str` constant
- [ ] Inline CSS + vanilla JS, no external deps
- [ ] Cards:
  - [ ] Header with version + device mode badge
  - [ ] Daemon card (uptime, idle timeout, Shut down button)
  - [ ] Models card (list with VRAM bars)
  - [ ] Indexing card (file, progress bar, ETA, queue length)
  - [ ] Resources card (CPU/GPU/RAM/VRAM gauges)
  - [ ] Indexes card (path, size, points, last touched)
  - [ ] Recent queries card (table: time, query, top-1 path, wall ms, status)
- [ ] JS poll `/status` every 2 s; flash card border on `indexing.active_file` change
- [ ] Inline image refs served from `/pages/<sha>/<page>.webp`

## E.5  CLI subcommands

- [ ] `ragrag dashboard` — print URL, optionally launch browser (`--open`)
- [ ] `ragrag status` — plain-text pretty-print of `/status`
- [ ] `ragrag shutdown` — already in Phase A, just confirm it uses HTTP if the socket is stuck

## E.6  Tests

- [ ] `tests/test_http_status.py` — start daemon without loaded models, hit `/status`, assert required keys
- [ ] `tests/test_http_pages.py` — populate cache, hit `/pages/<sha>/<page>.webp`, assert 200 + WebP magic bytes
- [ ] `tests/test_http_shutdown.py` — POST /shutdown with correct header → daemon exits; without header → 400
- [ ] `tests/test_cli_dashboard.py` — `ragrag dashboard` prints a URL matching `127.0.0.1:<int>/`

## Acceptance

- [ ] Start daemon, open dashboard, index a 20-page fixture PDF, watch progress bar move
- [ ] Fire two searches from another terminal, see them in Recent queries card
- [ ] `ragrag shutdown` from dashboard button works
- [ ] Commit `phase-E.json` (no retrieval delta expected, just sanity)

## Notes & decisions

_(append as we go)_

---

# Phase F — MCP server

**Why sixth.** Makes ragrag the natural RAG tool for any MCP-compatible LLM agent. Trivial
on top of A + C + D.

**Branch:** `phase-f-mcp`

## F.1  Server

- [ ] `ragrag/mcp_server.py` (NEW)
- [ ] `ragrag mcp` CLI subcommand → spawns MCP server bound to stdio (MCP standard)
- [ ] One tool: `search_documentation(query, top_k=5, paths=["."], rerank=True, include_page_images="base64")`
- [ ] Internally: `DaemonClient.ensure_daemon()` + `search()` + return
- [ ] Default `include_page_images="base64"` so MCP clients don't need filesystem access
- [ ] Compact JSON mode by default to stay under ~200 KB per response
- [ ] Base64 images for top-3 hero pages, paths for everything else

## F.2  pyproject

- [ ] `[project.optional-dependencies]` — `mcp = ["mcp>=0.9.0"]`
- [ ] Installation: `pip install ragrag[mcp]`

## F.3  Tests

- [ ] `tests/test_mcp_server.py` — subprocess the server, send `tools/list` + `tools/call`, assert JSON-RPC response shape + `SearchResponse` payload

## Acceptance

- [ ] `mcp inspect ragrag mcp` roundtrip works
- [ ] Response includes hero-page base64 for top-3
- [ ] Total response ≤ 250 KB for a top-5 query on the STM32 fixture

## Notes & decisions

_(append as we go)_

---

# Phase G — Benchmark harness polish

**Interleaved with every phase.** Not a separate milestone; each phase above pushes its
report to `validation/benchmarks/phase-X.json`.

## G.1  Harness upgrades

- [ ] `scripts/benchmark_stm32h743.py --daemon` flag — assume daemon is running, time RPC only
- [ ] `scripts/benchmark_stm32h743.py --report validation/benchmarks/<name>.json` convention
- [ ] `scripts/diff_bench.py <a.json> <b.json>` — per-question delta + aggregate delta
- [ ] Markdown output mode for `diff_bench.py` → paste into PR bodies

## G.2  Version-controlled benchmarks

- [ ] Create `validation/benchmarks/` directory
- [ ] Move `baseline.json` there (update script default path)
- [ ] Phase-A/B/C/D/E/F each commit their own report

## G.3  CI hooks (deferred)

- [ ] Future: nox session that runs the bench on a small fixture on every PR
- [ ] Future: publish bench reports as CI artifacts

## Notes & decisions

_(append as we go)_

---

## Critical files reference

| Path | Touched by | Purpose |
|---|---|---|
| `ragrag/daemon/server.py` | A, E | Accept loop, RPC dispatch, lifecycle, signals, idle timeout |
| `ragrag/daemon/client.py` | A, F | `DaemonClient`, auto-start, in-process fallback |
| `ragrag/daemon/rpc.py` | A | JSON-RPC 2.0 framing, errors |
| `ragrag/daemon/http_status.py` | E | Dashboard HTTP server, `/status`, `/pages`, `/shutdown` |
| `ragrag/daemon/static.py` | E | Bundled HTML dashboard |
| `ragrag/cli.py` | A, E, F | Thin client; `daemon`, `status`, `shutdown`, `dashboard`, `mcp` subcommands |
| `ragrag/config.py` | A–F | All new settings (daemon, reranker, chunker, location, page cache) |
| `ragrag/models.py` | B | `Chunk`, `PageContext`, `Location`, retire `Segment` (shim) |
| `ragrag/index/page_cache.py` | C | WebP page image cache with LRU eviction |
| `ragrag/index/ingest_manager.py` | B | Dispatch to topic chunker/segmenter; stream embed + upsert |
| `ragrag/index/qdrant_store.py` | B | Payload extended with chunk fields; store format unchanged |
| `ragrag/extractors/pdf_extractor.py` | B | Still yields page images; downstream is VLM chunker |
| `ragrag/extractors/text_topic_segmenter.py` | B | NEW — VLM topic segmenter for text files |
| `ragrag/extractors/vlm_topic_chunker.py` | B | NEW — VLM rolling-window topic discovery for PDFs |
| `ragrag/retrieval/search_engine.py` | B, C, D | Rollup, rerank hook, context + location attach |
| `ragrag/retrieval/location_builder.py` | B | NEW — `build_location()` |
| `ragrag/retrieval/reranker.py` | D | NEW — `VLMReranker` |
| `ragrag/retrieval/result_formatter.py` | C | markdown-rich + compact-json formatters |
| `ragrag/embedding/vlm_loader.py` | B, D | NEW — shared VLM loader + offload logic |
| `ragrag/embedding/colqwen_embedder.py` | — | No major change; loader extracted to vlm_loader |
| `ragrag/mcp_server.py` | F | NEW — MCP adapter |
| `scripts/benchmark_stm32h743.py` | G | `--daemon` flag, report path |
| `scripts/diff_bench.py` | G | NEW — delta reporter |
| `validation/benchmarks/*.json` | G | Per-phase reports, committed |
| `pyproject.toml` | A, E, F | `psutil`, `mcp` extra, optional `autoawq` extra |

---

## Out of scope (deliberate)

- Cross-encoder text-only reranker (BGE-reranker-v2-m3). Possibly later if VLM rerank has gaps.
- Query rewriting / synonym expansion. Reranker covers the gap.
- Multi-user daemon with authentication. Single-developer tool.
- Cloud / remote indexing. Local only.
- Fine-tuning any model. Prompt engineering + off-the-shelf weights only.
- HTTP REST API beyond the dashboard (status server is not a general-purpose API).
- Streaming result pages to the CLI.
- Windows-native daemon (Unix socket only in v1; Windows gets the in-process fallback).
- Related-document graph (heuristic sibling/header/import detection). Replaced by `Location`.
- Page-level shared-embedding store (avoids duplicate embeds on topic overlap). Deferred; v1 duplicates embedding work for overlap at ~30% indexing cost.

---

## Risks & mitigations

| Risk | Mitigation | Tracked in |
|---|---|---|
| CPU fallback latency is painful (VLM chunking ~20 min on Iris iGPU for 100 pages) | Document in `/status` as "CPU mode"; run small fixtures for CI; indexing runs in background so daemon stays responsive | Phase B |
| VLM chunker returns garbage JSON | 3 retries with terser prompts; fold into running topic on failure; loud WARNING | Phase B |
| Topic overlap blows up index size | `chunker_max_topic_overlap` warning if > 50% of pages in multiple topics | Phase B |
| Moondream2 rerank quality on dense datasheets | Opt-in on CPU; default `reranker_model="none"` on CPU | Phase D |
| 12-hour daemon holds ~5 GB RAM all day | Acceptable on 32 GB workstation; `ragrag shutdown` exists; `--no-daemon` short-circuits | Phase A |
| `Location.directory_listing` leaks vendor/private paths | `location_respect_gitignore=True` by default; exclude entries matching local `.gitignore` | Phase B |
| Windows support gap | Document; Windows users get in-process fallback; TCP-localhost daemon is a follow-up | Phase A |
| `autoawq` install friction | Optional extra; fall back to bnb 4-bit if not installed | Phase D |
| Chunk refactor invalidates existing indexes | One-time `--new` requirement on upgrade, documented in commit | Phase B |
| Multiple topics sharing a page → duplicate embedding cost | Accept ~30% slower indexing on overlapping corpora in v1; memoized page store is a future optimisation | Phase B |

---

## Change log

_(append at the end of each phase)_

- `2026-04-13` — Baseline captured on STM32H743VI at commit `47e90ff`. See `baseline.json`. P@1 = 0.25, P@5 = 0.67, avg query 21.7 s, store 3.0 GB. Memory fix (streaming + mmap store) complete.
- `2026-04-13` — Phase B architecture revised: topics over sources, no size gates, no related graph. Replaced section-based chunking with VLM topic discovery that allows overlap and non-contiguity. `Location` block replaces the related-document graph.
- `2026-04-14` — **Phase A landed.** Daemon + thin CLI + 12 h idle timeout + auto-spawn + in-process fallback + `ragrag {daemon,status,shutdown}` subcommands. 29 new tests (105 unit total, 109 with e2e). Phase-A benchmark on the existing STM32 index: quality unchanged; **avg query wall 23.6 s → 1.05 s**. See `validation/benchmarks/phase-A.json`.

---

## Links

- Latest bench: `validation/benchmarks/baseline.json` (will be moved there in Phase G).
- Baseline commit: `47e90ff` (memory fix + streaming ingest + custom mmap store).
- Internal design doc: `/home/pavel/.claude/plans/fluffy-knitting-cupcake.md` (private working notes; duplicates most of this file but with more scratch).
