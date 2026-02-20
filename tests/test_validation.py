from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import cast

import yaml
from PIL import Image, ImageDraw

from src.config import Settings, get_settings
from src.extractors.image_extractor import extract_image_segments
from src.extractors.ocr import is_tesseract_available
from src.extractors.pdf_extractor import extract_pdf_segments
from src.extractors.text_extractor import extract_text_segments
from src.file_state import FileStateTracker, compute_file_hash
from src.index.qdrant_store import QdrantStore
from src.models import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
    FileState,
    FileType,
    IndexingStats,
    Modality,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Segment,
    SkippedFile,
    TimingInfo,
)
from src.path_discovery import discover_files
from src.retrieval.result_formatter import format_as_json, format_as_markdown


ROOT = Path(__file__).resolve().parent.parent


def test_config_defaults() -> None:
    """Config loads with correct defaults."""
    settings = Settings()

    assert settings.index_path == ".ragrag"
    assert settings.include_hidden is False
    assert settings.follow_symlinks is False
    assert settings.max_files == 10000
    assert settings.pdf_dpi == 200
    assert settings.ocr_threshold == 50
    assert settings.chunk_size == 900
    assert settings.chunk_overlap == 100
    assert settings.top_k == 10
    assert settings.max_top_k == 50
    assert settings.model_id == "TomoroAI/tomoro-colqwen3-embed-4b"
    assert settings.max_visual_tokens == 1280
    assert settings.indexing_timeout == 600.0

def test_models_instantiate() -> None:
    """All Pydantic models instantiate correctly."""
    file_state = FileState(
        path="/tmp/example.txt",
        size=42,
        mtime_ns=123,
        content_hash_sha256="abc123",
        last_indexed_at=1.0,
        point_ids=["p1"],
    )
    assert file_state.path.endswith("example.txt")

    segment = Segment(
        segment_id="seg-1",
        path="/tmp/example.txt",
        file_type=FileType.TEXT,
        modality=Modality.TEXT,
        start_line=1,
        end_line=2,
        excerpt="hello",
    )
    assert segment.file_type == FileType.TEXT
    assert segment.modality == Modality.TEXT

    request = SearchRequest(paths=["validation/fixtures/text"], query="gpio")
    assert request.top_k == 10

    response = SearchResponse(
        query="gpio",
        status="complete",
        indexed_now=IndexingStats(),
        skipped_files=[SkippedFile(path="/tmp/skip.elf", reason="unsupported extension")],
        results=[
            SearchResult(
                rank=1,
                score=0.9,
                path="/tmp/example.txt",
                file_type="text",
                modality="text",
                start_line=1,
                end_line=2,
                excerpt="hello",
            )
        ],
        timing_ms=TimingInfo(total_ms=10.0),
    )
    assert response.results[0].rank == 1

    assert ".c" in TEXT_EXTENSIONS
    assert ".pdf" in PDF_EXTENSIONS
    assert ".png" in IMAGE_EXTENSIONS


def test_path_discovery(tmp_path: Path) -> None:
    """discover_files finds supported files and skips unsupported."""
    root = tmp_path / "fixtures"
    root.mkdir()
    _ = (root / "code.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    _ = (root / "doc.pdf").write_bytes(b"%PDF-1.7\n")
    _ = (root / "notes.md").write_text("# Notes\n", encoding="utf-8")
    _ = (root / "firmware.elf").write_bytes(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    _ = (root / ".hidden.txt").write_text("hidden", encoding="utf-8")

    settings = get_settings().model_copy(
        update={
            "include_hidden": False,
            "follow_symlinks": False,
            "max_files": 10000,
        }
    )

    discovered, skipped = discover_files([str(root)], settings)

    discovered_names = {Path(p).name for p in discovered}
    skipped_names = {Path(s.path).name for s in skipped}

    assert "code.c" in discovered_names
    assert "doc.pdf" in discovered_names
    assert "notes.md" in discovered_names
    assert "firmware.elf" in skipped_names
    assert ".hidden.txt" not in discovered_names


def test_file_state_tracker(tmp_path: Path) -> None:
    """FileStateTracker detects new/changed/unchanged files."""
    file_path = tmp_path / "state.txt"
    _ = file_path.write_text("alpha", encoding="utf-8")

    tracker = FileStateTracker()
    first_hash = compute_file_hash(str(file_path))
    assert isinstance(first_hash, str) and len(first_hash) == 64

    needs_reindex, _ = tracker.check_staleness(str(file_path))
    assert needs_reindex is True

    tracker.mark_indexed(str(file_path), ["point-1", "point-2"])
    needs_reindex, _ = tracker.check_staleness(str(file_path))
    assert needs_reindex is False
    assert tracker.get_point_ids(str(file_path)) == ["point-1", "point-2"]

    _ = file_path.write_text("beta", encoding="utf-8")
    needs_reindex, _ = tracker.check_staleness(str(file_path))
    assert needs_reindex is True


def test_text_extractor(tmp_path: Path) -> None:
    """Text extractor produces valid chunks."""
    content = ("Line " + ("x" * 80) + "\n") * 60
    file_path = tmp_path / "large.md"
    _ = file_path.write_text(content, encoding="utf-8")

    settings = get_settings().model_copy(
        update={"chunk_size": 900, "chunk_overlap": 100}
    )
    segments = extract_text_segments(str(file_path), settings)

    assert len(segments) >= 2
    for segment in segments:
        assert segment.file_type == FileType.TEXT
        assert segment.modality == Modality.TEXT
        assert segment.start_line is not None
        assert segment.end_line is not None
        assert segment.start_line <= segment.end_line
        assert segment.excerpt.strip()


def test_pdf_extractor_corrupt(tmp_path: Path) -> None:
    """PDF extractor handles corrupt files gracefully."""
    bad_pdf = tmp_path / "corrupt.pdf"
    _ = bad_pdf.write_bytes(b"not a real pdf")

    segments, images = extract_pdf_segments(str(bad_pdf), get_settings())
    assert segments == []
    assert images == []


def test_image_extractor(tmp_path: Path) -> None:
    """Image extractor produces one segment per image."""
    image_path = tmp_path / "img.png"
    image = Image.new("RGB", (240, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 30), "Timing", fill="black")
    image.save(image_path)

    segments, images = extract_image_segments(str(image_path), get_settings())

    assert len(segments) == 1
    assert len(images) == 1
    assert segments[0].file_type == FileType.IMAGE
    assert segments[0].modality == Modality.IMAGE
    assert segments[0].excerpt


def test_ocr_available() -> None:
    """Tesseract is available."""
    assert is_tesseract_available()


def test_qdrant_store() -> None:
    """QdrantStore can upsert, search, and delete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collection = f"test_collection_{uuid.uuid4().hex}"
        store = QdrantStore(path=tmpdir, collection_name=collection, embedding_dim=4)

        segment = Segment(
            segment_id=str(uuid.uuid4()),
            path="/tmp/test.txt",
            file_type=FileType.TEXT,
            modality=Modality.TEXT,
            start_line=1,
            end_line=1,
            excerpt="hello qdrant",
        )
        vector = [[0.10, 0.20, 0.30, 0.40], [0.40, 0.30, 0.20, 0.10]]

        store.upsert(segment, vector)
        hits = store.search(vector, top_k=5)
        assert len(hits) >= 1
        assert any(hit.payload and hit.payload.get("path") == segment.path for hit in hits)

        store.delete_by_ids([segment.segment_id])
        hits_after_delete = store.search(vector, top_k=5)
        assert len(hits_after_delete) == 0


def test_result_formatter() -> None:
    """Result formatter produces valid JSON and Markdown."""
    long_excerpt = "x" * 260
    response = SearchResponse(
        query="gpio init",
        status="complete",
        indexed_now=IndexingStats(files_added=1, files_updated=0, files_skipped_unchanged=2),
        skipped_files=[SkippedFile(path="/tmp/firmware.elf", reason="unsupported extension")],
        errors=["minor warning"],
        results=[
            SearchResult(
                rank=1,
                score=0.77,
                path="/tmp/stm32_gpio_driver.c",
                file_type="text",
                modality="text",
                start_line=10,
                end_line=20,
                excerpt=long_excerpt,
            )
        ],
        timing_ms=TimingInfo(total_ms=12.5),
    )

    as_json = format_as_json(response)
    parsed = cast(dict[str, object], json.loads(as_json))
    assert parsed["query"] == "gpio init"
    assert parsed["status"] == "complete"
    assert len(cast(list[object], parsed["results"])) == 1

    as_markdown = format_as_markdown(response)
    assert "# Search Results: \"gpio init\"" in as_markdown
    assert "## Results" in as_markdown
    assert "> " in as_markdown
    assert "..." in as_markdown


def test_queries_yaml_valid() -> None:
    """validation/expected/queries.yaml is valid and has required fields."""
    queries_file = ROOT / "validation" / "expected" / "queries.yaml"
    data = cast(dict[str, object], yaml.safe_load(queries_file.read_text(encoding="utf-8")))

    assert isinstance(data, dict)
    assert "queries" in data
    queries = cast(list[dict[str, object]], data["queries"])
    assert len(queries) >= 1

    required_fields = {"id", "query", "paths", "min_results", "description", "expected_paths_contain"}
    for query in queries:
        assert required_fields.issubset(query.keys())
        assert isinstance(query["id"], str) and query["id"].strip()
        assert isinstance(query["query"], str) and query["query"].strip()
        assert isinstance(query["paths"], list) and query["paths"]


def test_validation_fixtures_exist() -> None:
    """All validation fixture files exist."""
    expected_files = [
        ROOT / "validation" / "fixtures" / "pdfs" / "FluxGrip_FG40_datasheet.pdf",
        ROOT / "validation" / "fixtures" / "pdfs" / "esp32_datasheet_en.pdf",
        ROOT / "validation" / "fixtures" / "pdfs" / "rp2040-datasheet.pdf",
        ROOT / "validation" / "fixtures" / "images" / "timing_diagram.png",
        ROOT / "validation" / "fixtures" / "text" / "embedded_notes.md",
        ROOT / "validation" / "fixtures" / "text" / "project_config.yaml",
        ROOT / "validation" / "fixtures" / "text" / "stm32_gpio_driver.c",
        ROOT / "validation" / "fixtures" / "text" / "stm32_gpio_driver.h",
        ROOT / "validation" / "fixtures" / "unsupported" / "firmware.elf",
        ROOT / "validation" / "expected" / "queries.yaml",
    ]

    for path in expected_files:
        assert path.exists(), f"Missing fixture: {path}"
        assert path.is_file(), f"Expected file but got non-file: {path}"
        assert path.stat().st_size > 0, f"Fixture is empty: {path}"
