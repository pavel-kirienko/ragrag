from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ragrag.config import find_index_root


def test_find_index_root_prefers_local_index_dir_over_config(tmp_path: Path) -> None:
    index_dir = tmp_path / ".ragrag"
    index_dir.mkdir()
    (tmp_path / "ragrag.json").write_text(
        json.dumps({"index_path": "custom-index", "top_k": 42}),
        encoding="utf-8",
    )

    root, settings = find_index_root(str(tmp_path))

    assert root == str(tmp_path.resolve())
    assert settings.index_path == str(index_dir.resolve())


def test_find_index_root_supports_hidden_config_while_climbing(tmp_path: Path) -> None:
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    nested_dir = root_dir / "src"
    nested_dir.mkdir()
    (root_dir / ".ragrag.json").write_text(
        json.dumps({"top_k": 17, "index_path": "index-store"}),
        encoding="utf-8",
    )

    root, settings = find_index_root(str(nested_dir))

    assert root == str(root_dir.resolve())
    assert settings.top_k == 17
    assert settings.index_path == str((root_dir / "index-store").resolve())


def test_find_index_root_error_contains_create_or_new_guidance(tmp_path: Path) -> None:
    isolated_dir = tmp_path / "isolated"
    isolated_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        find_index_root(str(isolated_dir))

    message = str(exc_info.value)
    assert "ragrag.json" in message
    assert "--new" in message


def test_embedder_source_uses_cache_first_loading_contract() -> None:
    source = Path("ragrag/embedding/colqwen_embedder.py").read_text(encoding="utf-8")
    assert "try_to_load_from_cache" in source
    assert "local_files_only=local_only" in source


def test_noxfile_defines_unit_e2e_and_coverage_with_threshold() -> None:
    source = Path("noxfile.py").read_text(encoding="utf-8")

    assert "def unit(" in source
    assert "def e2e(" in source
    assert "def coverage(" in source
    assert "--cov-branch" in source
    assert re.search(r'coverage",\s*"report",\s*"--fail-under=80"', source)


def test_ci_workflow_runs_nox_sessions_and_model_cache() -> None:
    source = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "actions/cache@v4" in source
    assert "nox -s unit" in source
    assert "nox -s e2e" in source
    assert "nox -s coverage" in source
