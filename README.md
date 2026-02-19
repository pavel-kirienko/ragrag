# ragrag — Local Multimodal MCP Search Engine

CPU-only semantic search for embedded development documentation.
Uses ColQwen3 late-interaction embeddings + Qdrant MaxSim retrieval.

## Quick Start

### Install
pip install -e ".[dev]"

### Download validation data
python scripts/fetch_validation_data.py

### CLI Search (loads model on first run, ~2-5 min)
python scripts/search_cli.py --query "clock tree configuration" --paths validation/fixtures/

### MCP Server
python src/mcp_server.py

### Run Tests (no model required)
pytest tests/test_validation.py -v

## Configuration (environment variables)
- MODEL_ID: HuggingFace model ID (default: TomoroAI/tomoro-colqwen3-embed-4b)
- QDRANT_PATH: Local Qdrant storage path (default: ./qdrant_data)
- PDF_RENDER_DPI: PDF render resolution (default: 200)
- TOP_K_DEFAULT: Default search results count (default: 10)
- INDEXING_TIMEOUT_SECONDS: Max indexing time per request (default: 600)

## Architecture
- Extractors: text chunking, PDF page rendering, image OCR
- Embedder: ColQwen3 4B (CPU, BF16, ~8GB RAM)
- Store: Qdrant local mode, MaxSim multivector
- MCP: FastMCP server with async tool handler
