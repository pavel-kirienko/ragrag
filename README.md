<div align="center">

# Ragrag — local multimodal RAG for documents

_RTFM for AI_

</div>

Local *multimodal* semantic search using ColQwen3 late-interaction embeddings + Qdrant MaxSim retrieval.
Indexes text files, PDFs with images/diagrams, and standalone images.

Ragrag is originally designed to allow AI agents to read complex technical documentation when doing embedded development, where simple text-based indexing won't work due to abundance of diragrams, schematics, and complex tabular data.

## Usage

Install:

```bash
# Python 3.10-3.13 is currently supported. Python 3.14 is not yet supported by PyTorch.
pip install -e .  # TODO: upgrade to `pip install ragrag` when published.
```

On Apple Silicon, Ragrag automatically uses Metal (MPS) when available.

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

### Configuration files

It is possible to set the defaults per directory via the config file. Ragrag will look for `ragrag.json` or `.ragrag.json` in the current working directory in that order; if not found, it will climb directory tree until one is found. All fields are optional.

```json
{
  "index_path": ".ragrag",
  "model_id": "TomoroAI/tomoro-colqwen3-embed-4b",
  "max_visual_tokens": 1280,
  "top_k": 10,
  "max_top_k": 50,
  "pdf_dpi": 200,
  "ocr_threshold": 50,
  "chunk_size": 900,
  "chunk_overlap": 100,
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
