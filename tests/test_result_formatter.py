"""Tests for compact-json and markdown-rich result formatters."""
from __future__ import annotations

import json

import pytest

from ragrag.models import (
    IndexingStats,
    Location,
    PageContext,
    SearchResponse,
    SearchResult,
    TimingInfo,
)
from ragrag.retrieval.result_formatter import (
    _format_page_refs,
    format_as_compact_json,
    format_as_json,
    format_as_markdown,
    format_as_markdown_rich,
)


def _mk_result(**overrides) -> SearchResult:
    defaults = dict(
        rank=1,
        score=1.23,
        path="/tmp/docs/foo.pdf",
        file_type="pdf",
        modality="text",
        page=7,
        excerpt="excerpt body",
        chunk_id="abc",
        title="ADC characteristics",
        summary="Tables 77 through 80 describe ADC accuracy across modes.",
        page_refs=[6, 7, 8, 9],
        line_ranges=None,
        context_pages=[
            PageContext(page=6, page_image_path="/cache/6.webp"),
            PageContext(page=7, page_image_path="/cache/7.webp", page_image_b64="BASE64DATA"),
            PageContext(page=8, page_image_path="/cache/8.webp", page_image_b64="BASE64DATA2"),
        ],
        location=Location(
            path="/tmp/docs/foo.pdf",
            directory="/tmp/docs",
            directory_listing=["a.pdf", "b.pdf", "foo.pdf"],
            listing_truncated=False,
            listing_total=3,
        ),
        rerank_reason="Table 78 on page 7 directly answers the query.",
    )
    defaults.update(overrides)
    return SearchResult(**defaults)


def _mk_response(results: list[SearchResult]) -> SearchResponse:
    return SearchResponse(
        query="ADC INL 16-bit",
        status="complete",
        indexed_now=IndexingStats(files_added=1),
        results=results,
        timing_ms=TimingInfo(total_ms=42.0, query_embedding_ms=10.0, retrieval_ms=1.0),
    )


# --------------------------------------------------------------------------- #
# format_as_compact_json
# --------------------------------------------------------------------------- #

def test_compact_json_trims_excerpt_and_summary() -> None:
    long_text = "x" * 500
    response = _mk_response([_mk_result(excerpt=long_text, summary=long_text)])
    blob = format_as_compact_json(response)
    payload = json.loads(blob)
    result = payload["results"][0]
    assert len(result["excerpt"]) <= 243  # 240 + '...'
    assert result["excerpt"].endswith("...")
    assert len(result["summary"]) <= 243


def test_compact_json_strips_base64_from_non_hero_pages() -> None:
    response = _mk_response([_mk_result()])
    payload = json.loads(format_as_compact_json(response))
    pages = payload["results"][0]["context_pages"]
    # hero page is 7 (result.page). Only that page keeps its base64.
    for ctx in pages:
        if ctx["page"] == 7:
            assert ctx["page_image_b64"] == "BASE64DATA"
        else:
            assert ctx["page_image_b64"] is None
        # Paths always stay.
        assert ctx["page_image_path"]


def test_compact_json_collapses_timing() -> None:
    response = _mk_response([_mk_result()])
    payload = json.loads(format_as_compact_json(response))
    assert payload["timing_ms"] == {"total_ms": 42.0}


def test_compact_json_trims_large_directory_listing() -> None:
    long_listing = [f"f{i:03d}.txt" for i in range(100)]
    result = _mk_result(location=Location(
        path="/tmp/docs/foo.pdf",
        directory="/tmp/docs",
        directory_listing=long_listing,
        listing_truncated=False,
        listing_total=100,
    ))
    response = _mk_response([result])
    payload = json.loads(format_as_compact_json(response))
    listing = payload["results"][0]["location"]["directory_listing"]
    assert len(listing) == 17  # 8 + '...' + 8
    assert listing[8] == "..."
    assert payload["results"][0]["location"]["listing_truncated"] is True


# --------------------------------------------------------------------------- #
# format_as_markdown_rich
# --------------------------------------------------------------------------- #

def test_markdown_rich_has_topic_title_and_metadata() -> None:
    response = _mk_response([_mk_result()])
    out = format_as_markdown_rich(response)
    assert "# Search:" in out
    assert "## 1. ADC characteristics" in out
    assert "(score 1.230)" in out
    assert "**Pages**: 6–9" in out
    assert "Table 78 on page 7 directly answers the query." in out


def test_markdown_rich_embeds_page_images_via_file_uri() -> None:
    response = _mk_response([_mk_result()])
    out = format_as_markdown_rich(response)
    assert "![page 6](file:///cache/6.webp)" in out
    assert "![page 7](file:///cache/7.webp)" in out


def test_markdown_rich_embeds_base64_when_path_missing() -> None:
    """If only base64 is available, emit a data-URI image."""
    result = _mk_result(
        context_pages=[
            PageContext(page=1, page_image_path=None, page_image_b64="SMALL"),
        ],
    )
    out = format_as_markdown_rich(_mk_response([result]))
    assert "![page 1](data:image/webp;base64,SMALL)" in out


def test_markdown_rich_shows_location_block() -> None:
    response = _mk_response([_mk_result()])
    out = format_as_markdown_rich(response)
    assert "**Location**: `/tmp/docs`" in out
    assert "- a.pdf" in out


def test_markdown_rich_handles_empty_results() -> None:
    response = _mk_response([])
    out = format_as_markdown_rich(response)
    assert "_No results._" in out


# --------------------------------------------------------------------------- #
# _format_page_refs helper
# --------------------------------------------------------------------------- #

def test_format_page_refs_contiguous() -> None:
    assert _format_page_refs([1, 2, 3]) == "1–3"


def test_format_page_refs_with_gaps() -> None:
    assert _format_page_refs([1, 2, 3, 7, 8, 12]) == "1–3, 7–8, 12"


def test_format_page_refs_unsorted_input() -> None:
    assert _format_page_refs([3, 1, 2]) == "1–3"


def test_format_page_refs_empty() -> None:
    assert _format_page_refs([]) == ""
