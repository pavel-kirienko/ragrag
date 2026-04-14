"""Unit tests for the VLM topic client — prompt assembly + JSON parsing.

We never call a real VLM here. A stub ``Handle`` returns canned strings
that mimic the structure of Qwen2.5-VL's replies (valid JSON, malformed
JSON, prose wrappers, code fences) and we verify the client's parsing,
retry, and failure behaviour.
"""
from __future__ import annotations

from typing import Any

import pytest

from ragrag.extractors.vlm_topic_client import (
    PdfTopicAssignment,
    TextTopic,
    VLMTopicClient,
    VLMTopicClientError,
    _extract_json_block,
    _parse_pdf_topic_json,
    _parse_text_topic_json,
)


class _StubHandle:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, int]] = []

    def generate(self, text: str, images=None, *, max_new_tokens=512, temperature=0.0) -> str:
        self.calls.append((text, len(images) if images else 0))
        if not self.replies:
            raise AssertionError("stub handle ran out of replies")
        return self.replies.pop(0)


# --------------------------------------------------------------------------- #
# JSON extraction + parsing helpers
# --------------------------------------------------------------------------- #

def test_extract_json_block_strips_code_fences() -> None:
    raw = "```json\n{\"pages\": []}\n```"
    assert _extract_json_block(raw, "{") == '{"pages": []}'


def test_extract_json_block_finds_embedded_object() -> None:
    raw = 'Here is the JSON:\n{"pages": [1, 2]}\nEnd.'
    assert _extract_json_block(raw, "{") == '{"pages": [1, 2]}'


def test_extract_json_block_raises_on_missing_block() -> None:
    with pytest.raises(ValueError):
        _extract_json_block("this is just prose", "{")


def test_parse_pdf_topic_json_basic() -> None:
    raw = '''{
      "pages": [
        {"page": 1, "topics": [
          {"id": "t1", "is_continuation": false, "title": "Intro", "summary": "Overview of the part"}
        ]},
        {"page": 2, "topics": [
          {"id": "t1", "is_continuation": true}
        ]}
      ]
    }'''
    out = _parse_pdf_topic_json(raw, [1, 2], max_topics=16)
    assert [a.page for a in out] == [1, 2]
    assert out[0].topic_id == "t1"
    assert out[0].is_continuation is False
    assert out[0].title == "Intro"
    assert out[1].is_continuation is True


def test_parse_pdf_topic_json_allows_multiple_topics_per_page() -> None:
    raw = '''{"pages": [
      {"page": 5, "topics": [
        {"id": "adc", "is_continuation": true},
        {"id": "dac", "is_continuation": false, "title": "DAC characteristics", "summary": "-"}
      ]}
    ]}'''
    out = _parse_pdf_topic_json(raw, [5], max_topics=16)
    assert len(out) == 2
    assert {a.topic_id for a in out} == {"adc", "dac"}


def test_parse_pdf_topic_json_caps_new_topics() -> None:
    pages_json = ",".join(
        f'{{"page": {p}, "topics": [{{"id": "new{p}", "is_continuation": false, "title": "T{p}", "summary": "s"}}]}}'
        for p in range(1, 6)
    )
    raw = '{"pages": [' + pages_json + ']}'
    out = _parse_pdf_topic_json(raw, [1, 2, 3, 4, 5], max_topics=2)
    # Only the first 2 new topics should survive (T1, T2); subsequent are dropped.
    kept_ids = {a.topic_id for a in out}
    assert "new1" in kept_ids
    assert "new2" in kept_ids
    assert "new3" not in kept_ids


def test_parse_pdf_topic_json_rejects_non_object_top_level() -> None:
    with pytest.raises(ValueError):
        _parse_pdf_topic_json("[]", [1], max_topics=16)


def test_parse_pdf_topic_json_rejects_page_outside_window() -> None:
    raw = '{"pages": [{"page": 99, "topics": [{"id": "x", "is_continuation": false, "title": "T", "summary": "s"}]}]}'
    with pytest.raises(ValueError):
        _parse_pdf_topic_json(raw, [1, 2], max_topics=16)


def test_parse_text_topic_json_basic() -> None:
    raw = '[{"title": "header api", "summary": "public interface", "ranges": [[1, 40]]}]'
    out = _parse_text_topic_json(raw, "a\n" * 50, absolute_line_offset=0)
    assert len(out) == 1
    assert out[0].title == "header api"
    assert out[0].ranges == [(1, 40)]


def test_parse_text_topic_json_clamps_to_window_and_shifts() -> None:
    content = "a\n" * 20
    raw = '[{"title": "x", "summary": "s", "ranges": [[1, 100]]}]'
    out = _parse_text_topic_json(raw, content, absolute_line_offset=100)
    # 100 (lines in content) → clamped to 21 (20 lines + 1 for trailing), shifted by 100
    assert out[0].ranges[0][0] == 101
    assert out[0].ranges[0][1] > 100  # clamped to content length then shifted


def test_parse_text_topic_json_accepts_non_contiguous_ranges() -> None:
    raw = '[{"title": "x", "summary": "s", "ranges": [[1, 5], [20, 30]]}]'
    out = _parse_text_topic_json(raw, "a\n" * 40, absolute_line_offset=0)
    assert out[0].ranges == [(1, 5), (20, 30)]


def test_parse_text_topic_json_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        _parse_text_topic_json("[]", "a\n", absolute_line_offset=0)


# --------------------------------------------------------------------------- #
# VLMTopicClient — retry & error paths
# --------------------------------------------------------------------------- #

def test_client_retries_then_succeeds() -> None:
    bad = "not json at all"
    good = '{"pages": [{"page": 1, "topics": [{"id": "t1", "is_continuation": false, "title": "T", "summary": "s"}]}]}'
    handle = _StubHandle([bad, good])
    client = VLMTopicClient(handle)
    out = client.identify_pdf_topics([1], [object()], ["text"], {}, max_topics_per_call=16)
    assert len(out) == 1
    assert handle.calls[1][0].startswith("Your previous response could not be parsed")


def test_client_raises_after_max_retries() -> None:
    handle = _StubHandle(["garbage"] * 5)
    client = VLMTopicClient(handle, max_retries=3)
    with pytest.raises(VLMTopicClientError):
        client.identify_pdf_topics([1], [object()], ["text"], {}, max_topics_per_call=16)
    assert len(handle.calls) == 3


def test_client_identify_text_topics_basic() -> None:
    handle = _StubHandle(['[{"title": "whole file", "summary": "x", "ranges": [[1, 10]]}]'])
    client = VLMTopicClient(handle)
    out = client.identify_text_topics("line\n" * 10, language_hint="Python")
    assert len(out) == 1
    assert out[0].title == "whole file"


def test_client_identify_text_topics_failure_raises() -> None:
    handle = _StubHandle(["bogus"] * 5)
    client = VLMTopicClient(handle, max_retries=3)
    with pytest.raises(VLMTopicClientError):
        client.identify_text_topics("content\n", language_hint="text")


def test_identify_pdf_topics_rejects_mismatched_inputs() -> None:
    handle = _StubHandle([])
    client = VLMTopicClient(handle)
    with pytest.raises(ValueError):
        client.identify_pdf_topics([1, 2], [object()], ["only one"], {})


def test_identify_pdf_topics_empty_window_is_noop() -> None:
    handle = _StubHandle([])
    client = VLMTopicClient(handle)
    assert client.identify_pdf_topics([], [], [], {}) == []
