# Local CPU-Only Multimodal MCP Search Engine Design

## 1. Use Case

### 1.1 Problem statement
LLM coding agents working on embedded projects need fast semantic lookup over large local documentation sets:
- source trees
- PDF datasheets and reference manuals
- image files with text/diagrams

Traditional text-only RAG misses diagram-heavy content (clock trees, pin mux diagrams, timing plots, block diagrams). We need a single search system that can retrieve both textual and visual evidence.

### 1.2 Primary workflow
1. User starts a local Python MCP server in background.
2. Coding agent connects to MCP.
3. Agent calls one search tool with:
   - `paths`: files and/or directories
   - `query`: semantic natural-language question
4. Server expands directories into files, indexing supported files (PDF, text files (Markdown, source code, plain text, etc) and skipping non-data files (binaries, temporary files, etc). In a later stage of the project, support for EDA files will be added (schematics, PCB layout, etc).
5. For files not yet indexed or changed since last query, server indexes them immediately.
6. Server runs semantic retrieval and returns best matches with location hints and excerpts.

### 1.3 Why multimodal is mandatory
Electronics documents contain content that is not faithfully representable as plain text:
- schematics
- signal timing diagrams
- clock trees
- annotated block diagrams

Therefore, the MVP must index:
- text
- PDF pages as images
- standalone images

### 1.4 MVP goals
- Single local process, CPU-only.
- No dependency on cloud inference providers.
- Best feasible retrieval quality for visually rich technical docs.
- Lazy indexing on first query and on change (index-on-demand).
- No mandatory persistent storage (unless it simplifies implementation).

### 1.5 MVP non-goals
- distributed indexing
- authentication/authorization/security
- guaranteed real-time indexing of all filesystem changes
- cloud fallback

## 2. Chosen Technical Direction

## 2.1 Single-model decision
Use one embedding model for all modalities:
- `TomoroAI/tomoro-colqwen3-embed-4b`

Rationale:
- handles text + image embeddings
- late-interaction style retrieval quality for visual documents
- open weights, fully local execution

### 2.2 Storage / retrieval backend
Use Qdrant multivector search with MaxSim:
- stores token/patch-level embeddings
- supports late-interaction retrieval pattern

### 2.3 Extraction stack
- PDFs: PyMuPDF (`fitz`) for page rendering and text extraction
- OCR for image-only regions/pages: Tesseract (`pytesseract`)
- Images: Pillow/OpenCV + OCR

OCR is auxiliary for snippets and metadata; visual embeddings remain the primary retrieval signal for diagrams.

## 3. High-Level Architecture

```text
MCP Client (LLM agent)
        |
        v
   MCP Server (Python)
        |
        +--> Path Expander + File Enumerator
        |
        +--> Index Manager (lazy indexing, staleness checks)
        |         |
        |         +--> Content Extractor
        |         |       - text files
        |         |       - PDFs (text + rendered pages)
        |         |       - images
        |         |       - eventually EDA files
        |         |
        |         +--> Embedding Engine (tomoro-colqwen3, CPU)
        |         |
        |         +--> Qdrant Multivector Collection
        |
        +--> Search Engine (query embedding + filtered retrieval)
        |
        +--> Result Assembler (JSON + Markdown)
```

## 4. Core Components

### 4.1 MCP server layer
Responsibilities:
- expose one primary tool `semantic_search`
- validate inputs
- orchestrate index-then-search flow
- return structured results consumable by agents

Suggested stack:
- `mcp` Python SDK (`FastMCP`)
- async handlers with bounded worker pools for CPU-heavy operations

### 4.2 Path expansion and discovery
Input can include mixed files and directories. Expansion rules:
- file path: include if supported type
- directory path: recursive file walk
- symlink handling: disabled by default (avoid cycles)
- hidden files: optional, default false
- max files per request: configurable safety limit

Supported MVP file types:
- text/code: `.txt .md .rst .c .h .cpp .hpp .py .json .yaml .yml .toml` (expandable)
- PDF: `.pdf`
- images: `.png .jpg .jpeg .bmp .tiff .webp`

### 4.3 File staleness + identity
For each file maintain:
- absolute path
- size
- mtime_ns
- optional content hash (lazy, only if needed for collision-proofing)

Staleness check:
- if `(size, mtime_ns)` unchanged and indexed -> skip reindex
- else remove old points for path, re-ingest file

### 4.4 Content extraction

#### 4.4.1 Text files
- read UTF-8 with replacement fallback
- split into line-preserving chunks
- keep `start_line`, `end_line`, raw text snippet

Chunking strategy:
- target 600-1200 characters
- overlap 80-120 characters
- prefer boundary-aware splits (blank lines/headings)

#### 4.4.2 PDF files
For each page:
1. extract native text using PyMuPDF
2. render page image (for visual embedding), default 200 DPI
3. if native text too short, run OCR on rendered image
4. create:
   - page-image segment (always)
   - text chunks from native/OCR text when available

This ensures diagrams are searchable even when text extraction fails.

#### 4.4.3 Image files
- load image
- create visual segment embedding
- run OCR for auxiliary snippet/keywords
- store OCR text in payload for explainability

### 4.5 Embedding engine (local CPU)

Model:
- `TomoroAI/tomoro-colqwen3-embed-4b`

Runtime:
- `torch` CPU
- `torch.inference_mode()`
- `attn_implementation="sdpa"` fallback path

Engine interface:
- `embed_query_text(query: str) -> MultiVector`
- `embed_text_chunk(text: str) -> MultiVector`
- `embed_image(image: PIL.Image) -> MultiVector`

Where `MultiVector` is shaped as `N x D`:
- `N`: token/patch vectors per unit
- `D`: embedding dimension (resolved from model output)

Operational controls:
- max text tokens per chunk
- max image resolution before resize
- batch size (small, CPU-safe defaults)

### 4.6 Vector index (Qdrant)

Collection design:
- one collection for all segments
- multivector enabled with MaxSim comparator
- payload contains metadata for filtering and presentation

Payload fields:
- `segment_id`
- `path`
- `file_type` (`text|pdf|image`)
- `modality` (`text|image`)
- `page` (nullable)
- `start_line` (nullable)
- `end_line` (nullable)
- `bbox` (nullable; for image region hints)
- `excerpt` (short text)
- `mtime_ns`
- `fingerprint`

Indices:
- payload index on `path`
- payload index on `file_type`
- payload index on `page`

Deletion strategy:
- before reindexing a changed file: delete all points where `path == file_path`

### 4.7 Retrieval engine

Search flow:
1. embed query text as multivector
2. search Qdrant with payload filter limited to requested paths
3. retrieve top-K segments
4. group/normalize scores by file and location
5. construct excerpts and location hints

Location hint strategy:
- text segments: line range from chunk metadata
- PDF/image visual segments:
   - page number
   - optional approximate region from similarity heatmap-to-grid mapping
   - OCR snippet closest to top-scoring region (if available)

### 4.8 Result formatter

Primary tool output should be structured JSON (best machine-parsable format for agents).  
Optional companion Markdown summary can be included for readability.

## 5. MCP Tool Contract

## 5.1 Tool name
`semantic_search`

### 5.2 Input schema
```json
{
  "paths": ["./docs", "./datasheets/stm32.pdf", "./images/clock_tree.png"],
  "query": "clock tree diagram and APB domain clocks",
  "top_k": 8,
  "include_markdown": true
}
```

Field definitions:
- `paths` (required): list of filesystem paths (files and/or directories)
- `query` (required): semantic query string
- `top_k` (optional): number of results, default `10`, max `50`
- `include_markdown` (optional): include human-readable markdown block in the output

### 5.3 Output schema
```json
{
  "query": "clock tree diagram and APB domain clocks",
  "indexed_now": {
    "files_added": 3,
    "files_updated": 1,
    "files_skipped_unchanged": 27
  },
  "results": [
    {
      "rank": 1,
      "score": 0.8123,
      "path": "/abs/path/datasheets/stm32.pdf",
      "file_type": "pdf",
      "modality": "image",
      "page": 114,
      "start_line": null,
      "end_line": null,
      "bbox": [0.22, 0.18, 0.91, 0.74],
      "excerpt": "RCC clock distribution and PLL branches..."
    }
  ],
  "markdown": "## Search results ...",
  "timing_ms": {
    "discovery": 45,
    "indexing": 22340,
    "query_embedding": 980,
    "retrieval": 210,
    "formatting": 8,
    "total": 23583
  }
}
```

## 6. Detailed Runtime Flow

### 6.1 End-to-end request handling
1. Validate request parameters.
2. Expand input paths to eligible file list.
3. For each file:
   - compare metadata with index cache
   - if new/changed -> ingest + embed + upsert points
4. Embed query.
5. Execute filtered multivector search.
6. Build ranked response with excerpts and locations.
7. Return JSON (+ optional Markdown).

### 6.2 Path filter correctness
To avoid cross-project contamination, retrieval is always filtered to resolved files from the current request.  
Even if index contains previously indexed files from other paths, they are excluded unless requested.

### 6.3 Incremental behavior
The system is lazy and incremental:
- first query on a large directory may be slow (cold indexing)
- subsequent queries are faster unless files changed

## 7. Data Structures

### 7.1 In-memory file registry
```python
FileState = {
  "path": str,
  "size": int,
  "mtime_ns": int,
  "fingerprint": str,
  "last_indexed_at": float,
  "point_ids": list[str]
}
```

### 7.2 Segment metadata
```python
Segment = {
  "segment_id": str,
  "path": str,
  "file_type": str,
  "modality": str,
  "page": int | None,
  "start_line": int | None,
  "end_line": int | None,
  "bbox": tuple[float, float, float, float] | None,
  "excerpt": str
}
```

## 8. Quality Strategy

### 8.1 Why this should work for diagram queries
For a query like `"clock tree diagram"`:
- query text is embedded into late-interaction query vectors
- PDF page images and standalone images are embedded into patch/token vectors
- MaxSim matching can score relevant visual regions even with sparse/no text

### 8.2 Excerpt generation policy
- if text chunk hit: excerpt from chunk text around strongest local match
- if image hit with OCR text: excerpt from OCR near matched region
- if image hit without OCR: excerpt fallback like `"Visual match on page 114 (clock diagram region)."`

### 8.3 Ranking normalization
Because mixed modalities can have score-scale differences:
- perform per-modality score normalization before final ranking
- keep raw score and normalized score in debug mode

## 9. Performance Expectations (CPU-Only)

### 9.1 Practical expectation
- cold start indexing of large PDFs will be the bottleneck
- warm query latency should be substantially lower than cold indexing latency

### 9.2 MVP optimization knobs
- lower PDF render DPI (e.g., 150 instead of 200) for faster indexing
- cap max page image size
- reduce chunk overlap
- bound top_k
- cache rendered page images during current process lifetime

### 9.3 Operational guidance
- run initial warm-up query over main doc directories before active coding session
- keep process alive to reuse in-memory index

## 10. Error Handling and Resilience

### 10.1 File-level fault tolerance
If one file fails to ingest:
- log error entry
- continue processing remaining files
- return partial results with `errors[]` in response

### 10.2 Corrupt/unsupported files
- skip with explicit reason
- include in `skipped_files[]`

### 10.3 Timeouts
- configurable soft timeout for indexing phase
- if exceeded, return best-effort results from indexed subset

## 11. Security and Safety

### 11.1 Filesystem boundaries
By default allow absolute paths supplied by caller, but expose optional allowlist root configuration:
- `ALLOWED_ROOTS=/home/pavel/projects,/home/pavel/docs`

### 11.2 Prompt/data safety
No outbound network calls in MVP.  
All indexing and retrieval happen locally.

## 12. Dependencies

Core:
- `mcp` (Python SDK)
- `qdrant-client`
- `torch`
- `transformers`
- `Pillow`
- `PyMuPDF`
- `pytesseract`
- `numpy`
- `pydantic`

System packages:
- `tesseract-ocr`

## 13. Proposed Project Layout

```text
ragrag/
  pyproject.toml
  README.md
  docs/
    local-cpu-multimodal-mcp-search-design.md
  src/
    mcp_server.py
    config.py
    models.py
    path_discovery.py
    file_state.py
    extractors/
      text_extractor.py
      pdf_extractor.py
      image_extractor.py
      ocr.py
    embedding/
      colqwen_embedder.py
    index/
      qdrant_store.py
      ingest_manager.py
    retrieval/
      search_engine.py
      result_formatter.py
```

## 14. Implementation Plan

### Phase 1: server skeleton + core tool
- create MCP server with `semantic_search`
- implement path expansion and response scaffolding

### Phase 2: indexing pipeline
- text extractor + chunking
- PDF rendering + text extraction + OCR fallback
- image ingestion + OCR metadata

### Phase 3: embeddings + vector store
- integrate tomoro-colqwen3 local CPU embeddings
- Qdrant multivector collection and upsert/delete

### Phase 4: retrieval + formatting
- filtered multivector search
- ranking and excerpts
- JSON + Markdown output modes

### Phase 5: hardening
- timeouts
- partial-failure reporting
- benchmark script and tuning defaults

## 15. MVP Acceptance Criteria

1. Query over a directory containing mixed text/PDF/images returns ranked semantic matches.
2. First query lazily indexes files; repeated query avoids unnecessary reindex when unchanged.
3. Diagram-centric query (e.g. `"clock tree diagram"`) can hit relevant PDF page/image even when text extraction is weak.
4. Results include file path and approximate location (`page` and/or `line range` and optional `bbox`).
5. Entire pipeline runs locally on CPU with no network inference.

## 16. Open Decisions (keep minimal)

These are implementation details, not architecture changes:
- default PDF DPI (150 vs 200)
- default chunk size
- OCR trigger threshold for "too little native text"
- max files per request safeguard

Architecture remains fixed: local CPU-only, single-model multimodal late-interaction retrieval.

## 17. Manual Testing Tool

Add a simple Python CLI helper for manual testing without an LLM agent:
- file: `scripts/search_cli.py`
- purpose: invoke the same `semantic_search` tool path used by MCP and print results
- output modes:
  - raw JSON (default, easy for debugging)
  - Markdown (`--markdown`) for human inspection

Suggested CLI usage:
```bash
python scripts/search_cli.py \
  --paths ./docs ./datasheets/stm32.pdf \
  --query "clock tree diagram and APB clocks" \
  --top-k 8 \
  --markdown
```

Recommended behavior:
- accepts repeated `--paths`
- validates path existence before request
- calls the same internal search service entrypoint as MCP handler (no duplicated logic)
- prints `indexed_now`, `timing_ms`, and ranked results
- exits non-zero on validation/runtime errors
