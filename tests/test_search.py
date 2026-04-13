from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ragrag.config import Settings
from ragrag.index.ingest_manager import IngestManager
from ragrag.index.qdrant_store import QdrantStore
from ragrag.models import SearchRequest
from ragrag.retrieval.search_engine import SearchEngine


class MockEmbedder:
    embedding_dim = 4

    def embed_text_chunk(self, text: str):
        _ = text
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_text_chunks(self, texts):
        return [self.embed_text_chunk(t) for t in texts]

    def embed_query_text(self, query: str):
        _ = query
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_image(self, image):
        _ = image
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_images(self, images):
        return [self.embed_image(img) for img in images]


def _build_engine(tmp_path: Path, *, embedder: MockEmbedder | None = None) -> tuple[SearchEngine, IngestManager]:
    settings = Settings(index_path=str(tmp_path / ".ragrag"))
    (tmp_path / ".ragrag").mkdir(parents=True, exist_ok=True)

    active_embedder = embedder or MockEmbedder()
    store = QdrantStore(path=str(tmp_path / ".ragrag"), collection_name="test", embedding_dim=4)
    ingest_manager = IngestManager(cast(Any, active_embedder), store, settings)
    search_engine = SearchEngine(cast(Any, active_embedder), store, ingest_manager, settings)
    return search_engine, ingest_manager


def test_search_finds_indexed_content(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    _ = file_path.write_text("GPIO configuration and pin setup details\n", encoding="utf-8")

    search_engine, _ = _build_engine(tmp_path)
    response = search_engine.search(SearchRequest(paths=[str(file_path)], query="GPIO setup", top_k=5))

    assert response.status == "complete"
    assert response.indexed_now.files_added >= 1
    assert len(response.results) > 0


def test_search_empty_index(tmp_path: Path) -> None:
    search_engine, _ = _build_engine(tmp_path)
    response = search_engine.search(SearchRequest(paths=[str(tmp_path)], query="no files here", top_k=5))

    assert response.status == "complete"
    assert response.results == []


def test_search_query_embedding_error(tmp_path: Path) -> None:
    class FailingQueryEmbedder(MockEmbedder):
        def embed_query_text(self, query: str):
            _ = query
            raise RuntimeError("query embedding failed")

    _ = (tmp_path / "notes.txt").write_text("timer register reference\n", encoding="utf-8")
    search_engine, _ = _build_engine(tmp_path, embedder=FailingQueryEmbedder())
    response = search_engine.search(SearchRequest(paths=[str(tmp_path)], query="timer register", top_k=5))

    assert response.status == "partial"
    assert response.errors


def test_search_path_filter(tmp_path: Path) -> None:
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    path_a.mkdir()
    path_b.mkdir()

    file_a = path_a / "a.txt"
    file_b = path_b / "b.txt"
    _ = file_a.write_text("clock tree source A\n", encoding="utf-8")
    _ = file_b.write_text("clock tree source B\n", encoding="utf-8")

    search_engine, ingest_manager = _build_engine(tmp_path)
    _ = ingest_manager.ingest_paths([str(path_a), str(path_b)])
    response = search_engine.search(SearchRequest(paths=[str(file_a)], query="clock tree", top_k=10))

    assert response.status == "complete"
    assert response.results
    assert all(result.path == str(file_a.resolve()) for result in response.results)
