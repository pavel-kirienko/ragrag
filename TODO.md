# TODO

The program takes a few seconds to start even before downloading the model. Even running `ragrag --version` takes a very long time. This must be fixed.

Replace pure extension-based file filtering with something more clever like MIME detector, such that we don't have to hardcode every known programming laguage and image format. One idea is to use mimetypes or libmagic.

The tool should only emit stderr info log messages if it has to update the index, in which case it is allowed to be verbose. This is because it may potentially take a (very) long time so we need to keep some liveness indication going. When searching an existing index it is better to stay quiet. These are not 100% hard rules but more of a guiding principle.

The test suite must include:

- Unit tests for basic functions inside the library.
- End-to-end test that install the library into a fresh venv and ensure it runs. When running on CI, it is probably best to cache the model somewhere.
- Both unit and e2e tests must be automated with Nox (noxfile.py).
- Nox should track and report code coverage. At least 80% branch coverage is required, bonus points for 90%.
- There must be a solid CI pipeline that exercises the Nox automation fully. We don't want to download the large model files excessively though so maybe some caching is needed.

The config documentation requires a detailed explanation for every field. Right now the purpose of `chunk_overlap` is unclear, for example.

`max_files` is not needed, we always index everything (assume infinity). `follow_symlinks` should be enabled by default. `indexing_timeout` should be set to 100000 by default to allow long unattended runs.

---

When all of the above are done, the project needs to be published on PyPI as `ragrag`. Use env var `PYPI_API_TOKEN`.
