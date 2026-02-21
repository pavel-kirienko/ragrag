from __future__ import annotations

import itertools
import logging
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from src.config import Settings
from src.index.ingest_manager import IngestManager
from src.index.qdrant_store import QdrantStore
from src.models import FileType, Modality


class MockEmbedder:
    embedding_dim: int = 4

    def embed_text_chunk(self, text: str):
        _ = text
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_query_text(self, query: str):
        _ = query
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_image(self, image: object):
        _ = image
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]


def _build_manager(tmp_path: Path, *, indexing_timeout: float = 100000.0) -> IngestManager:
    index_path = tmp_path / ".ragrag"
    index_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(index_path=str(index_path), indexing_timeout=indexing_timeout)
    store = QdrantStore(
        path=str(index_path),
        collection_name=f"ingest_test_{uuid.uuid4().hex}",
        embedding_dim=4,
    )
    return IngestManager(cast(Any, MockEmbedder()), store, settings)


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
    with caplog.at_level(logging.INFO, logger="src.index.ingest_manager"):
        stats, skipped, _ = manager.ingest_paths([str(text_file)])

    assert stats.files_added == 1
    assert skipped == []
    messages = [record.getMessage() for record in caplog.records if record.name == "src.index.ingest_manager"]
    assert any("Indexing" in message and "(1/1)" in message for message in messages)
    assert any("Index up to date:" in message for message in messages)


def test_ingest_unchanged_is_quiet_at_info_level(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    text_file = tmp_path / "quiet.txt"
    _ = text_file.write_text(_long_text(), encoding="utf-8")

    manager = _build_manager(tmp_path)
    _ = manager.ingest_paths([str(text_file)])

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="src.index.ingest_manager"):
        stats, skipped, _ = manager.ingest_paths([str(text_file)])

    assert stats.files_skipped_unchanged == 1
    assert skipped == []
    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "src.index.ingest_manager" and record.levelno >= logging.INFO
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

    from src.index import ingest_manager as ingest_module

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


def test_extract_segments_text(tmp_path: Path) -> None:
    text_file = tmp_path / "chunks.txt"
    _ = text_file.write_text(_long_text(), encoding="utf-8")

    manager = _build_manager(tmp_path)
    extract_segments = cast(Any, getattr(manager, "_extract_segments"))
    segments, images = extract_segments(str(text_file), FileType.TEXT)

    assert len(segments) > 0
    assert images == []
    assert all(segment.modality == Modality.TEXT for segment in segments)


def test_extract_segments_image(tmp_path: Path) -> None:
    image_file = tmp_path / "diagram.png"
    image = Image.new("RGB", (100, 50), "white")
    image.save(str(image_file))

    manager = _build_manager(tmp_path)
    extract_segments = cast(Any, getattr(manager, "_extract_segments"))
    segments, images = extract_segments(str(image_file), FileType.IMAGE)

    assert len(segments) == 1
    assert len(images) == 1
    assert segments[0].modality == Modality.IMAGE
