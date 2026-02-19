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
pip install -e .  # TODO: upgrade to `pip install ragrag` when published.
```

⚠️ The very first run will take a **very long time** because the tool will download the model from Huggingface. Despite downloading from the internet, indexing and search run 100% locally.

Search all supported documents in the current directory and subdirectories that are related to clock tree configuration:

```bash
ragrag "clock tree configuration"
```

When a new file is found or an existing file is changed, the model will automatically re-index it (no need to tell it to index manually), which may take anywhere from a few seconds to who knows how long depending on the documents and the performance of your computer (everything is done locally).
The index is stored in `./.ragrag`; it therefore matters which directory the tool is run from.
While reindexing is in progress, the tool may log in stderr, while the search results go to stdout.

Search specific directories with more results and with Markdown output:

```bash
ragrag "GPIO initialization" --top-k 20 --markdown
```

For more options see `ragrag --help`.

### Configuration files

It is possible to set the defaults per directory via the config file. Ragrag will look for `ragrag.json` or `.ragrag.json` in the current working directory in that order; if not found, it will climb directory tree until one is found. All fields are optional.

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

## Development

```bash
# Download validation corpus (PDFs, fixtures)
python scripts/fetch_validation_data.py

# Run tests (no model required)
pytest tests/test_validation.py -v
```
