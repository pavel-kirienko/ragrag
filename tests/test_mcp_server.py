"""Unit tests for the ragrag MCP server.

Covers the response trimming logic and the daemon->in-process
fallback dispatch without requiring the optional ``mcp`` package to
be installed — we import the helpers directly.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from ragrag.mcp_server import _run_search, _trim_response


# --------------------------------------------------------------------------- #
# _trim_response
# --------------------------------------------------------------------------- #


def _fake_response(hero_image_path: str | None = None) -> dict:
    return {
        "query": "ADC sampling rate",
        "status": "complete",
        "timing_ms": {"total_ms": 1234.5},
        "results": [
            {
                "rank": 1,
                "score": 9.5,
                "path": "/docs/stm32.pdf",
                "title": "ADC Electrical Characteristics",
                "summary": "ADC INL, sampling, conversion.",
                "page_refs": [166, 167, 168],
                "page": 166,
                "excerpt": "Table 78 — ADC INL..." + "x" * 1000,  # long excerpt to test truncation
                "rerank_reason": "best match for ADC INL",
                "location": {
                    "path": "/docs/stm32.pdf",
                    "directory": "/docs",
                    "directory_listing": ["stm32.pdf", "rp2040.pdf"],
                    "listing_truncated": False,
                    "listing_total": 2,
                },
                "context_pages": [
                    {"page": 166, "page_image_path": hero_image_path, "text": ""},
                    {"page": 167, "page_image_path": None, "text": ""},
                ],
            },
        ],
    }


def test_trim_response_drops_large_context_pages_and_truncates_excerpt() -> None:
    trimmed = _trim_response(_fake_response(), image_mode="none", max_inline_image_kb=0)
    assert trimmed["query"] == "ADC sampling rate"
    assert trimmed["total_ms"] == 1234.5
    assert len(trimmed["results"]) == 1
    r = trimmed["results"][0]
    assert r["title"] == "ADC Electrical Characteristics"
    assert len(r["excerpt"]) <= 400
    assert r["location"]["directory"] == "/docs"
    assert r["hero_page"] == 166
    assert r["hero_page_image_path"] is None
    assert r["hero_page_image_b64"] is None


def test_trim_response_base64_inlines_hero_image(tmp_path: Path) -> None:
    img = tmp_path / "hero.webp"
    img.write_bytes(b"\x00\x01\x02fake webp")
    trimmed = _trim_response(
        _fake_response(str(img)),
        image_mode="base64",
        max_inline_image_kb=64,
    )
    r = trimmed["results"][0]
    assert r["hero_page_image_path"] == str(img)
    assert r["hero_page_image_b64"] == base64.b64encode(img.read_bytes()).decode("ascii")


def test_trim_response_respects_budget(tmp_path: Path) -> None:
    img = tmp_path / "big.webp"
    img.write_bytes(b"x" * 20_000)
    trimmed = _trim_response(
        _fake_response(str(img)),
        image_mode="base64",
        max_inline_image_kb=1,  # 1024 bytes, less than the image
    )
    r = trimmed["results"][0]
    assert r["hero_page_image_path"] == str(img)
    assert r["hero_page_image_b64"] is None  # over budget, not inlined


def test_trim_response_path_mode_keeps_paths_drops_b64(tmp_path: Path) -> None:
    img = tmp_path / "x.webp"
    img.write_bytes(b"wx")
    trimmed = _trim_response(
        _fake_response(str(img)),
        image_mode="path",
        max_inline_image_kb=64,
    )
    r = trimmed["results"][0]
    assert r["hero_page_image_path"] == str(img)
    assert r["hero_page_image_b64"] is None


# --------------------------------------------------------------------------- #
# _run_search with daemon fallback
# --------------------------------------------------------------------------- #


def test_run_search_falls_back_to_inprocess_when_daemon_unreachable(monkeypatch) -> None:
    """DaemonClient raises -> _run_search catches and calls _inprocess_search."""
    from ragrag.daemon.client import DaemonError

    class _BadClient:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, **_kwargs):
            raise DaemonError(-1, "no daemon for you")

    monkeypatch.setattr("ragrag.daemon.client.DaemonClient", _BadClient)

    called = {}

    def _fake_inprocess(index_path, query, top_k, paths):
        called["yes"] = (index_path, query, top_k, tuple(paths))
        return {
            "query": query,
            "status": "complete",
            "timing_ms": {"total_ms": 10.0},
            "results": [],
        }

    monkeypatch.setattr("ragrag.mcp_server._inprocess_search", _fake_inprocess)

    result = _run_search(
        index_path="/tmp/fakeidx",
        query="hello",
        top_k=3,
        paths=["/tmp/fakeidx"],
        image_mode="none",
        max_inline_image_kb=0,
    )
    assert called["yes"] == ("/tmp/fakeidx", "hello", 3, ("/tmp/fakeidx",))
    assert result["query"] == "hello"
    assert result["status"] == "complete"
    assert result["results"] == []


def test_run_search_uses_daemon_when_reachable(monkeypatch) -> None:
    class _GoodClient:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, **_kwargs):
            return {
                "query": _kwargs.get("query"),
                "status": "complete",
                "timing_ms": {"total_ms": 42.0},
                "results": [
                    {
                        "rank": 1,
                        "score": 5.0,
                        "path": "/a.pdf",
                        "title": "A",
                        "summary": "",
                        "page_refs": [1],
                        "excerpt": "hello world",
                        "context_pages": [],
                        "location": {
                            "path": "/a.pdf",
                            "directory": "/",
                            "directory_listing": [],
                            "listing_truncated": False,
                            "listing_total": 0,
                        },
                    }
                ],
            }

    monkeypatch.setattr("ragrag.daemon.client.DaemonClient", _GoodClient)
    result = _run_search(
        index_path="/tmp/fakeidx",
        query="hi",
        top_k=1,
        paths=["/tmp/fakeidx"],
        image_mode="none",
        max_inline_image_kb=0,
    )
    assert result["query"] == "hi"
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "A"
