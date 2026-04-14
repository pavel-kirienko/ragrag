from __future__ import annotations

import itertools
import logging
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from ragrag.config import Settings
from ragrag.extractors.vlm_topic_client import PdfTopicAssignment, TextTopic, VLMTopicClient
from ragrag.index.ingest_manager import IngestManager
from ragrag.index.qdrant_store import QdrantStore
from ragrag.models import FileType, Modality


class MockEmbedder:
    embedding_dim: int = 4

    def embed_text_chunk(self, text: str):
        _ = text
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_text_chunks(self, texts):
        return [self.embed_text_chunk(t) for t in texts]

    def embed_query_text(self, query: str):
        _ = query
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_image(self, image: object):
        _ = image
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_images(self, images):
        return [self.embed_image(img) for img in images]


class _StubHandle:
    def generate(self, *a, **kw):
        raise NotImplementedError  # stub subclass never hits the base


class _AllInOneVLMClient(VLMTopicClient):
    """Deterministic stub: every file is one topic, covering whatever pages or
    lines are passed in. Used by the ingest tests to bypass the real VLM while
    keeping the rest of the pipeline real."""

    def __init__(self) -> None:
        super().__init__(_StubHandle())

    def identify_pdf_topics(self, window_pages, window_images, window_texts, running_topics, *, max_topics_per_call=16):
        is_new = "t1" not in running_topics
        return [
            PdfTopicAssignment(
                page=p,
                topic_id="t1",
                is_continuation=not is_new,
                title="" if not is_new else "Stub topic",
                summary="Single topic covering the whole file (stub VLM)",
            )
            for p in window_pages
        ]

    def identify_text_topics(self, content, *, language_hint="text", absolute_line_offset=0):
        n_lines = content.count("\n") + 1
        return [
            TextTopic(
                title=f"Stub {language_hint} topic",
                summary="Single topic covering the whole file (stub VLM)",
                ranges=[(absolute_line_offset + 1, absolute_line_offset + max(1, n_lines))],
            )
        ]


def _build_manager(tmp_path: Path, *, indexing_timeout: float = 100000.0) -> IngestManager:
    index_path = tmp_path / ".ragrag"
    index_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(index_path=str(index_path), indexing_timeout=indexing_timeout)
    store = QdrantStore(
        path=str(index_path),
        collection_name=f"ingest_test_{uuid.uuid4().hex}",
        embedding_dim=4,
    )
    return IngestManager(
        cast(Any, MockEmbedder()),
        store,
        settings,
        vlm_client=_AllInOneVLMClient(),
    )


def _long_text() -> str:
    return "The STM32 microcontroller GPIO configuration requires setting the MODER register. " * 10


def test_ingest_text_file(tmp_path: Path) -> None:
    text_file = tmp_path / "driver.md"
    _ = text_file.write_text(_long_text(), encoding="utf-8")

    manager = _build_manager(tmp_path)
    stats, skipped, _ = manager.ingest_paths([str(text_file)])

    assert stats.files_added == 1
    assert skipped == []


def test_ingest_unchanged_file(tmp_path: Path) -> None:
    text_file = tmp_path / "same.txt"
    _ = text_file.write_text(_long_text(), encoding="utf-8")

    manager = _build_manager(tmp_path)
    _ = manager.ingest_paths([str(text_file)])
    stats, skipped, _ = manager.ingest_paths([str(text_file)])

    assert stats.files_skipped_unchanged == 1
    assert skipped == []


def test_ingest_modified_file(tmp_path: Path) -> None:
    text_file = tmp_path / "change.txt"
    _ = text_file.write_text(_long_text(), encoding="utf-8")

    manager = _build_manager(tmp_path)
    _ = manager.ingest_paths([str(text_file)])
    _ = text_file.write_text(_long_text() + "updated", encoding="utf-8")
    stats, skipped, _ = manager.ingest_paths([str(text_file)])

    assert stats.files_updated == 1
    assert skipped == []


def test_ingest_logs_progress_when_updating_index(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    text_file = tmp_path / "logging.txt"
    _ = text_file.write_text(_long_text(), encoding="utf-8")

    manager = _build_manager(tmp_path)
    with caplog.at_level(logging.INFO, logger="ragrag.index.ingest_manager"):
        stats, skipped, _ = manager.ingest_paths([str(text_file)])

    assert stats.files_added == 1
    assert skipped == []
    messages = [record.getMessage() for record in caplog.records if record.name == "ragrag.index.ingest_manager"]
    assert any(
        ("Planning" in message or "Indexing" in message) and "(1/1)" in message
        for message in messages
    )
    assert any("Index up to date:" in message for message in messages)


def test_ingest_unchanged_is_quiet_at_info_level(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    text_file = tmp_path / "quiet.txt"
    _ = text_file.write_text(_long_text(), encoding="utf-8")

    manager = _build_manager(tmp_path)
    _ = manager.ingest_paths([str(text_file)])

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ragrag.index.ingest_manager"):
        stats, skipped, _ = manager.ingest_paths([str(text_file)])

    assert stats.files_skipped_unchanged == 1
    assert skipped == []
    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "ragrag.index.ingest_manager" and record.levelno >= logging.INFO
    ]
    assert info_messages == []


def test_ingest_unsupported_file(tmp_path: Path) -> None:
    binary_file = tmp_path / "firmware.elf"
    _ = binary_file.write_bytes(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00")

    manager = _build_manager(tmp_path)
    stats, skipped, _ = manager.ingest_paths([str(binary_file)])

    assert stats.files_added == 0
    assert any(item.path == str(binary_file.resolve()) for item in skipped)
    assert any(item.reason == "unsupported file type" for item in skipped)


def test_ingest_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text_file = tmp_path / "timeout.txt"
    _ = text_file.write_text(_long_text(), encoding="utf-8")

    manager = _build_manager(tmp_path, indexing_timeout=0)

    from ragrag.index import ingest_manager as ingest_module

    tick = itertools.count()
    monkeypatch.setattr(ingest_module.time, "time", lambda: float(next(tick)))
    _, skipped, _ = manager.ingest_paths([str(text_file)])

    assert any(item.path == str(text_file.resolve()) for item in skipped)
    assert any(item.reason == "indexing timeout" for item in skipped)


def test_ingest_multiple_files(tmp_path: Path) -> None:
    for idx in range(3):
        _ = (tmp_path / f"file_{idx}.txt").write_text(_long_text(), encoding="utf-8")

    manager = _build_manager(tmp_path)
    stats, skipped, _ = manager.ingest_paths([str(tmp_path)])

    assert stats.files_added == 3
    assert skipped == []


class CountingEmbedder(MockEmbedder):
    """Records every batch size it's asked to embed, for test observation."""

    def __init__(self) -> None:
        self.text_batches: list[int] = []
        self.single_calls: int = 0
        self.raise_next_batch: bool = False

    def embed_text_chunk(self, text: str):
        self.single_calls += 1
        return super().embed_text_chunk(text)

    def embed_text_chunks(self, texts):
        if self.raise_next_batch:
            self.raise_next_batch = False
            raise RuntimeError("simulated batch failure")
        self.text_batches.append(len(texts))
        return [super(CountingEmbedder, self).embed_text_chunk(t) for t in texts]


def _build_manager_with_embedder(
    tmp_path: Path, embedder: MockEmbedder, *, text_batch_size: int = 8,
) -> IngestManager:
    index_path = tmp_path / ".ragrag"
    index_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(index_path=str(index_path), text_batch_size=text_batch_size)
    store = QdrantStore(
        path=str(index_path),
        collection_name=f"ingest_test_{uuid.uuid4().hex}",
        embedding_dim=4,
    )
    return IngestManager(
        cast(Any, embedder),
        store,
        settings,
        vlm_client=_AllInOneVLMClient(),
    )


def test_ingest_without_vlm_client_rejects_text_files(tmp_path: Path) -> None:
    """Policy: no heuristic chunking without a VLM."""
    text_file = tmp_path / "rejected.txt"
    _ = text_file.write_text(_long_text(), encoding="utf-8")

    index_path = tmp_path / ".ragrag"
    index_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(index_path=str(index_path))
    store = QdrantStore(
        path=str(index_path),
        collection_name=f"ingest_test_{uuid.uuid4().hex}",
        embedding_dim=4,
    )
    mgr = IngestManager(cast(Any, MockEmbedder()), store, settings, vlm_client=None)
    stats, skipped, _ = mgr.ingest_paths([str(text_file)])
    assert stats.files_added == 0
    assert any("VLM client" in item.reason for item in skipped)


def test_ingest_embeds_text_topic_chunk(tmp_path: Path) -> None:
    """The VLM stub returns one topic covering the file; exactly one text
    embed call lands and the file state is marked indexed."""
    content = ("paragraph with several reasonable-length words in it. " * 2 + "\n\n") * 30
    text_file = tmp_path / "batched.md"
    _ = text_file.write_text(content, encoding="utf-8")

    embedder = CountingEmbedder()
    manager = _build_manager_with_embedder(tmp_path, embedder, text_batch_size=8)

    stats, skipped, _ = manager.ingest_paths([str(text_file)])

    assert stats.files_added == 1
    assert skipped == []
    # One text chunk per file → exactly one call to embed_text_chunks with one item.
    assert embedder.text_batches == [1]
