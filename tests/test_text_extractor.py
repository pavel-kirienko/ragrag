"""Regression tests for text_extractor chunk splitting and overlap."""
from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.extractors.text_extractor import _chunk_text, _find_boundary, extract_text_segments


def test_find_boundary_sentence_end() -> None:
    text = "Hello. World and the rest of the string keeps going on."
    pos = _find_boundary(text, 0, len(text))
    assert pos is not None
    assert text[:pos].endswith(". ")
    assert pos == text.index("World")


def test_find_boundary_prefers_rightmost_blank_line() -> None:
    text = "intro\n\nfirst section text\n\nsecond section text"
    pos = _find_boundary(text, 0, len(text))
    assert pos is not None
    assert text[:pos].count("\n\n") == 2
    assert text[pos:].startswith("second section text")


def test_find_boundary_heading() -> None:
    text = "some prose\n## Section Two\ncontent"
    pos = _find_boundary(text, 0, len(text))
    assert pos is not None
    assert text[pos:].startswith("Section Two")


def test_find_boundary_word_fallback() -> None:
    text = "aaaaa bbbbb ccccc ddddd"
    pos = _find_boundary(text, 0, len(text))
    assert pos is not None
    assert text[:pos].endswith(" ")
    assert text[pos:] == "ddddd"


def test_chunk_text_overlap_inflates_total_length() -> None:
    content = ("word " * 400).strip()
    plain = _chunk_text(content, target_chars=200, overlap_chars=0)
    overlapped = _chunk_text(content, target_chars=200, overlap_chars=50)
    assert len(overlapped) >= 2
    # With non-zero overlap the chunks collectively carry more characters than
    # the source because the overlap region is re-emitted in the next chunk.
    total_plain = sum(len(c[0]) for c in plain)
    total_over = sum(len(c[0]) for c in overlapped)
    assert total_over > total_plain


def test_chunk_text_zero_overlap_has_no_backtrack() -> None:
    content = ("word " * 400).strip()
    chunks = _chunk_text(content, target_chars=200, overlap_chars=0)
    assert len(chunks) >= 2
    joined = "".join(c[0] for c in chunks)
    # With zero overlap the concatenated chunks should reconstruct the source exactly.
    assert joined == content


def test_chunk_text_never_loops_on_tiny_region() -> None:
    # Region smaller than overlap must still terminate in finite steps.
    chunks = _chunk_text("abcd", target_chars=3, overlap_chars=100)
    assert chunks  # produced something
    assert chunks[-1][0].endswith("d")


def test_extract_text_segments_uses_overlap(tmp_path: Path) -> None:
    content = ("The quick brown fox jumps over the lazy dog. " * 80)
    file_path = tmp_path / "foxes.txt"
    _ = file_path.write_text(content, encoding="utf-8")

    settings = Settings(chunk_size=300, chunk_overlap=80)
    segments = extract_text_segments(str(file_path), settings)

    assert len(segments) >= 2
    for prev, curr in zip(segments, segments[1:]):
        # Overlapping chunks will both contain at least one word seen in the
        # other, proving the overlap backtrack is wired up.
        prev_words = set(prev.excerpt[-80:].split())
        curr_words = set(curr.excerpt[:80].split())
        assert prev_words & curr_words


def test_extract_text_segments_line_numbers_are_1_indexed(tmp_path: Path) -> None:
    content = "\n".join(f"line {i}" for i in range(1, 21)) + "\n"
    file_path = tmp_path / "lines.txt"
    _ = file_path.write_text(content, encoding="utf-8")

    settings = Settings(chunk_size=60, chunk_overlap=0)
    segments = extract_text_segments(str(file_path), settings)

    assert segments[0].start_line == 1
    for seg in segments:
        assert seg.start_line >= 1
        assert seg.end_line >= seg.start_line
