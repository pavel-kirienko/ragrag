# ragrag — Local Multimodal Search for Embedded Docs

CPU-only semantic search using ColQwen3 late-interaction embeddings + Qdrant MaxSim retrieval.
Indexes text files, PDFs (page images), and standalone images. Results via JSON or Markdown.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Search current directory
ragrag "clock tree configuration"

# Search specific directories
ragrag "SPI timing diagram" ./docs ./datasheets

# More results, Markdown output
ragrag "GPIO initialization" --top-k 20 --markdown

# Override model
ragrag "motor controller specs" ./pdfs --model TomoroAI/tomoro-colqwen3-embed-4b
```

## Configuration

Create `ragrag.json` (or `.ragrag.json`) in your working directory. All fields are optional:

```json
{
  "model_id": "TomoroAI/tomoro-colqwen3-embed-4b",
  "index_path": ".ragrag",
  "top_k": 10,
  "pdf_dpi": 200,
  "chunk_size": 900,
  "chunk_overlap": 100,
  "max_files": 10000,
  "include_hidden": false,
  "follow_symlinks": false,
  "indexing_timeout": 600.0
}
```

## How It Works

1. **First run**: downloads the model (~8GB, one-time, ~10-30 min depending on connection)
2. **Extraction**: text chunking, PDF page rendering, image OCR
3. **Embedding**: ColQwen3 4B (CPU, BF16, ~8GB RAM)
4. **Index**: stored in `.ragrag/` directory (Qdrant local mode)
5. **Subsequent runs**: skip unchanged files (content hash detection)

Progress and logs go to **stderr**. Results go to **stdout** (pipe-friendly).

## Supported File Types

| Type | Extensions |
|------|-----------|
| Text | `.txt`, `.md`, `.rst`, `.c`, `.h`, `.cpp`, `.hpp`, `.py`, `.json`, `.yaml`, `.toml` |
| PDF  | `.pdf` (each page rendered as image + text extracted) |
| Image | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp` |

## Tests

```bash
# Download validation corpus (PDFs, fixtures)
python scripts/fetch_validation_data.py

# Run tests (no model required)
pytest tests/test_validation.py -v
```

## Architecture

- **Extractors**: text chunking, PDF page rendering (PyMuPDF), image OCR (Tesseract)
- **Embedder**: ColQwen3 4B — 320-dim multivector, L2-normalized token embeddings
- **Store**: Qdrant local mode, MaxSim late-interaction retrieval
- **CLI**: `src/cli.py` — argparse, JSON stdout, progress stderr
