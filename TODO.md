# TODO

## First order

### Search for index before creating a new one

When the program is invoked and there is no `.ragrag/` directory, instead of creating one immediately in the current directory, the tool must check the existence of `.ragrag/` or config files in this directory (`ragrag.json` or `.ragrag.json`) or in parent directories (similar to git etc):

1. Check if the current directory has `.ragrag/`; if yes, use that.
2. Check if the current directory has `ragrag.json` or `.ragrag.json`; if yes, create index here.
3. Climb one level up and go to 1. If no levels left to climb or directory is not writable, error out.

The error must instruct the user to either create a config file somewhere or to use `--new` CLI key to force the tool to create a new index here.

### Search result filtering

Search results include results from documents unrelated to the query. If the tool is invoked as `ragrag "query" foo.pdf`, only results from `foo.pdf` must be returned, all other matches must be suppressed. If the caller wanted to search other documents, they would have been included in the invocation.

### Change detection is broken

Files are reindexed on every run even when they are already in the index and haven't changed since. Easy to reproduce on a text file.

### Better logging

The tool should emit very detailed stderr info log messages if it has to update the index, preferably with progress reports (e.g., indexing document such and such, document A out of N, page X out of Y). This is because it may potentially take a (very) long time so we need to keep detailed liveness indication going. When searching an existing index it is better to stay quiet. The tool must use the Python logging module with stderr sink and the verbosity set to INFO. Paths that do not update the index should emit at most DEBUG messages.

### Slow start

The program takes a few seconds to start even before downloading the model. Even running `ragrag --version` takes a very long time. This must be fixed.

When starting, the tool must not attempt to contact Huggingface if the model is already downloaded, and it must not emit `Loading model (this may take a few minutes on first run)...` and `Model loaded.`.

### Avoid extension hardcoding

Replace pure extension-based file filtering with something more clever like MIME detector, such that we don't have to hardcode every known programming laguage and image format. One idea is to use mimetypes or libmagic.

### Better performance

The indexing currently is impractically slow, taking 3 minutes to index a small text document. The model must not be forced to run on CPU; instead, it should make use of GPU when one is available (automatically without user configuration) and switch to CPU only as a last resort. The user must not be required to configure this for the tool to work.

## Second order

The test suite must include:

- Unit tests for basic functions inside the library.
- End-to-end test that install the library into a fresh venv and ensure it runs. When running on CI, it is probably best to cache the model somewhere.
- Both unit and e2e tests must be automated with Nox (noxfile.py).
- Nox should track and report code coverage. At least 80% branch coverage is required, bonus points for 90%.
- There must be a solid CI pipeline that exercises the Nox automation fully. We don't want to download the large model files excessively though so maybe some caching is needed.

The config documentation requires a detailed explanation for every field. Right now the purpose of `chunk_overlap` is unclear, for example.

`max_files` is not needed, we always index everything (assume infinity). `follow_symlinks` should be enabled by default. `indexing_timeout` should be set to 100000 by default to allow long unattended runs.

## Third order

When all of the above are done and reviewed, the project needs to be published on PyPI as `ragrag`. Use env var `PYPI_API_TOKEN`.
