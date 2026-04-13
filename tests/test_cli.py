from __future__ import annotations

import builtins
import sys
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from src.cli import _build_parser, main
from src.config import Settings
from src.models import IndexingStats, SearchResponse, TimingInfo


class MockEmbedder:
    embedding_dim = 4

    def embed_text_chunk(self, text: str):
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_text_chunks(self, texts):
        return [self.embed_text_chunk(t) for t in texts]

    def embed_query_text(self, query: str):
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_image(self, image):
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_images(self, images):
        return [self.embed_image(img) for img in images]


def _make_mock_response(status: str = "complete") -> SearchResponse:
    return SearchResponse(
        query="test",
        status=status,
        indexed_now=IndexingStats(),
        timing_ms=TimingInfo(),
    )


@contextmanager
def _patched_main_dependencies(
    tmp_path,
    response: SearchResponse | None = None,
    side_effect: BaseException | None = None,
) -> Iterator[tuple[Any, Any, Any, Any]]:
    settings = Settings(index_path=str(tmp_path / ".ragrag"), top_k=13)
    with (
        patch("src.embedding.colqwen_embedder.ColQwenEmbedder") as mock_embedder_cls,
        patch("src.index.qdrant_store.QdrantStore") as mock_store_cls,
        patch("src.index.ingest_manager.IngestManager") as mock_ingest_cls,
        patch("src.retrieval.search_engine.SearchEngine") as mock_engine_cls,
        patch("src.config.find_index_root", return_value=(str(tmp_path), settings)),
        patch("src.cli.find_index_root", return_value=(str(tmp_path), settings)),
        patch("src.cli.get_settings", return_value=settings),
    ):
        mock_embedder_cls.return_value = MockEmbedder()
        if side_effect is not None:
            mock_engine_cls.return_value.search.side_effect = side_effect
        else:
            mock_engine_cls.return_value.search.return_value = response or _make_mock_response()
        yield mock_embedder_cls, mock_store_cls, mock_ingest_cls, mock_engine_cls


def test_parser_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["query"])

    assert args.query == "query"
    assert args.paths == ["."]
    assert args.top_k is None
    assert args.output_json is True
    assert args.output_markdown is False
    assert args.new is False


def test_parser_top_k() -> None:
    parser = _build_parser()
    args = parser.parse_args(["query", "--top-k", "5"])

    assert args.top_k == 5


def test_parser_markdown() -> None:
    parser = _build_parser()
    args = parser.parse_args(["query", "--markdown"])

    assert args.output_markdown is True


def test_parser_version() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])

    assert exc_info.value.code == 0


def test_main_version_exits_before_runtime_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = {
        "src.embedding.colqwen_embedder",
        "src.index.ingest_manager",
        "src.index.qdrant_store",
        "src.models",
        "src.retrieval.result_formatter",
        "src.retrieval.search_engine",
    }
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):
        if name in blocked:
            raise AssertionError(f"unexpected runtime import during --version: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["ragrag", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0


def test_parser_new_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["query", "--new"])

    assert args.new is True


def test_parser_model_override() -> None:
    parser = _build_parser()
    args = parser.parse_args(["query", "--model", "my/model"])

    assert args.model == "my/model"


def test_main_success(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    with _patched_main_dependencies(tmp_path, response=_make_mock_response("complete")) as (_, _, _, mock_engine_cls):
        monkeypatch.setattr(sys, "argv", ["ragrag", "test query"])
        result = main()

    assert result == 0
    request = mock_engine_cls.return_value.search.call_args.args[0]
    assert request.top_k == 13
    assert request.include_markdown is False
    assert '"status": "complete"' in capsys.readouterr().out


def test_main_partial_status(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    with _patched_main_dependencies(tmp_path, response=_make_mock_response("partial")):
        monkeypatch.setattr(sys, "argv", ["ragrag", "test query"])
        result = main()

    assert result == 1


def test_main_exception(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    with _patched_main_dependencies(tmp_path, side_effect=RuntimeError("boom")):
        monkeypatch.setattr(sys, "argv", ["ragrag", "test query"])
        result = main()

    assert result == 1
    assert "Error: boom" in capsys.readouterr().err


def test_main_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    with _patched_main_dependencies(tmp_path, side_effect=KeyboardInterrupt()):
        monkeypatch.setattr(sys, "argv", ["ragrag", "test query"])
        result = main()

    assert result == 1
    assert "Interrupted." in capsys.readouterr().err


def test_main_new_flag(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    with _patched_main_dependencies(tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["ragrag", "test query", "--new"])
        result = main()

    assert result == 0
    assert (tmp_path / ".ragrag").is_dir()


def test_main_markdown_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _patched_main_dependencies(tmp_path, response=_make_mock_response("complete")):
        monkeypatch.setattr(sys, "argv", ["ragrag", "test query", "--markdown"])
        result = main()

    assert result == 0
    assert "# Search Results" in capsys.readouterr().out


def test_main_model_and_top_k_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    with _patched_main_dependencies(tmp_path) as (mock_embedder_cls, _, _, mock_engine_cls):
        monkeypatch.setattr(
            sys,
            "argv",
            ["ragrag", "test query", "--model", "my/model", "--top-k", "7"],
        )
        result = main()

    assert result == 0
    mock_embedder_cls.assert_called_once_with("my/model", 16384, quantization="auto")
    request = mock_engine_cls.return_value.search.call_args.args[0]
    assert request.top_k == 7
