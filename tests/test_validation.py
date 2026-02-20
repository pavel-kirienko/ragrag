from __future__ import annotations

import ast
import json
import tempfile
import uuid
from pathlib import Path
from typing import Callable, cast

import pytest
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


def test_config_climbing(tmp_path: Path) -> None:
    from src.config import find_index_root

    ragrag_dir = tmp_path / ".ragrag"
    ragrag_dir.mkdir()
    child_dir = tmp_path / "subdir"
    child_dir.mkdir()

    root, settings = find_index_root(start_dir=str(child_dir))
    assert str(root) == str(tmp_path)
    assert settings.index_path == str(ragrag_dir)


def test_config_climbing_no_index(tmp_path: Path) -> None:
    from src.config import find_index_root

    isolated = tmp_path / "isolated"
    isolated.mkdir()

    with pytest.raises(SystemExit):
        _ = find_index_root(start_dir=str(isolated))


def test_mime_detection_verilog(tmp_path: Path) -> None:
    from src import models as models_module
    from src.models import FileType, get_file_type

    verilog_file = tmp_path / "top.v"
    _ = verilog_file.write_text("module top; endmodule\n", encoding="utf-8")

    ft = get_file_type(str(verilog_file))
    has_magic = cast(bool, getattr(models_module, "_HAS_MAGIC"))
    expected = FileType.TEXT if has_magic else None
    assert ft == expected, f"Expected {expected} for .v file, got {ft}"


def test_mime_detection_binary(tmp_path: Path) -> None:
    from src.models import get_file_type

    elf_file = tmp_path / "firmware.elf"
    _ = elf_file.write_bytes(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00")

    ft = get_file_type(str(elf_file))
    assert ft is None, f"Expected None for ELF binary, got {ft}"


def test_mime_detection_empty(tmp_path: Path) -> None:
    from src import models as models_module
    from src.models import FileType, get_file_type

    empty_file = tmp_path / "empty.txt"
    _ = empty_file.write_bytes(b"")

    ft = get_file_type(str(empty_file))
    has_magic = cast(bool, getattr(models_module, "_HAS_MAGIC"))
    expected = None if has_magic else FileType.TEXT
    assert ft == expected, f"Expected {expected} for empty file, got {ft}"


def test_qdrant_matchany_import() -> None:
    source = Path("src/index/qdrant_store.py").read_text(encoding="utf-8")
    assert "MatchAny" in source, "MatchAny not found in qdrant_store.py"
    assert "MatchValue(any=" not in source, "Bug: MatchValue(any=...) still present"


def test_gpu_detection() -> None:
    from src.embedding import colqwen_embedder

    detect_device = cast(Callable[[], str], getattr(colqwen_embedder, "_detect_device"))
    device = detect_device()
    assert device in ("cuda", "mps", "cpu"), f"Unexpected device: {device}"


def test_cli_no_heavy_toplevel_imports() -> None:
    source = Path("src/cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    heavy_modules = {"src.embedding", "src.index", "src.retrieval"}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for heavy in heavy_modules:
                assert not node.module.startswith(heavy), (
                    f"Heavy top-level import found at line {node.lineno}: {node.module}"
                )


def test_default_log_level_is_info() -> None:
    source = Path("src/cli.py").read_text(encoding="utf-8")
    assert 'default="INFO"' in source or "default='INFO'" in source, (
        "Default log level is not INFO in cli.py"
    )


def test_file_state_tracker(tmp_path: Path) -> None:
    """FileStateTracker detects new/changed/unchanged files."""
    file_path = tmp_path / "state.txt"
    _ = file_path.write_text("alpha", encoding="utf-8")
    index_dir = tmp_path / ".ragrag"
    index_dir.mkdir()

    tracker = FileStateTracker(str(index_dir))
    first_hash = compute_file_hash(str(file_path))
    assert isinstance(first_hash, str) and len(first_hash) == 64

    needs_reindex, current_state = tracker.check_staleness(str(file_path))
    assert needs_reindex is True

    tracker.mark_indexed(
        str(file_path),
        ["point-1", "point-2"],
        file_state=current_state,
    )
    needs_reindex, _ = tracker.check_staleness(str(file_path))
    assert needs_reindex is False
    assert tracker.get_point_ids(str(file_path)) == ["point-1", "point-2"]

    tracker_reloaded = FileStateTracker(str(index_dir))
    needs_reindex, _ = tracker_reloaded.check_staleness(str(file_path))
    assert needs_reindex is False
    assert tracker_reloaded.get_point_ids(str(file_path)) == ["point-1", "point-2"]

    _ = file_path.write_text("beta", encoding="utf-8")
    needs_reindex, _ = tracker_reloaded.check_staleness(str(file_path))
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


@pytest.mark.timeout(10)
def test_search_performance(tmp_path: Path) -> None:
    import random
    import time

    collection = f"perf_test_{uuid.uuid4().hex}"
    store = QdrantStore(path=str(tmp_path), collection_name=collection, embedding_dim=128)

    rng = random.Random(42)
    doc_path = str(tmp_path / "test_doc.txt")

    for i in range(50):
        segment = Segment(
            segment_id=str(uuid.uuid4()),
            path=doc_path,
            file_type=FileType.TEXT,
            modality=Modality.TEXT,
            start_line=i * 10 + 1,
            end_line=i * 10 + 10,
            excerpt=f"Segment {i}: GPIO configuration register bank {i}",
        )
        vector = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(8)]
        store.upsert(segment, vector)

    query_vector = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(8)]

    t0 = time.perf_counter()
    results = store.search(query_vector, top_k=10)
    elapsed = time.perf_counter() - t0

    assert len(results) > 0, "Search returned no results"
    assert len(results) <= 10, "Search returned more than top_k results"
    assert elapsed <= 10.0, f"Search took {elapsed:.2f}s, expected <= 10s"


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
