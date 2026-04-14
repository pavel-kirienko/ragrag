"""Format SearchResponse objects for CLI / MCP / dashboard output.

Four formats are supported:

- ``json`` — full SearchResponse via ``model_dump_json``. Programmatic
  parsing target.
- ``compact-json`` — same shape but trimmed for LLM agents with tight
  token budgets: excerpts capped at 240 chars, base64 page images limited
  to the hero page only, directory listings trimmed to 16 entries.
- ``markdown`` — legacy human-readable Markdown (kept for back-compat).
- ``markdown-rich`` — new topic-aware Markdown that embeds page-image
  references, the topic title / summary / rerank reason, and the
  ``Location`` block.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ragrag.models import SearchResponse, SearchResult


# --------------------------------------------------------------------------- #
# JSON family
# --------------------------------------------------------------------------- #

def format_as_json(response: SearchResponse) -> str:
    """Serialize a SearchResponse to JSON with 2-space indent."""
    return response.model_dump_json(indent=2)


def format_as_compact_json(response: SearchResponse) -> str:
    """LLM-friendly compact JSON: small excerpts, trimmed base64, no redundancy."""
    payload = response.model_dump()
    for result in payload.get("results", []) or []:
        _compact_result(result)
    timing = payload.get("timing_ms") or {}
    if timing:
        payload["timing_ms"] = {"total_ms": timing.get("total_ms", 0.0)}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _compact_result(result: dict[str, Any]) -> None:
    """In-place shrink of a single result dict."""
    excerpt = (result.get("excerpt") or "")
    if len(excerpt) > 240:
        result["excerpt"] = excerpt[:240] + "..."
    summary = (result.get("summary") or "")
    if len(summary) > 240:
        result["summary"] = summary[:240] + "..."
    # Keep base64 for at most the hero context page; others fall back to paths only.
    context_pages = result.get("context_pages") or []
    hero_page = result.get("page")
    b64_budget = 1
    for ctx in context_pages:
        if b64_budget > 0 and ctx.get("page_image_b64"):
            if hero_page is not None and ctx.get("page") != hero_page:
                ctx["page_image_b64"] = None
            else:
                b64_budget -= 1
        else:
            ctx["page_image_b64"] = None
    # Trim directory listing
    location = result.get("location") or {}
    listing = location.get("directory_listing") or []
    if len(listing) > 16:
        location["directory_listing"] = listing[:8] + ["..."] + listing[-8:]
        location["listing_truncated"] = True


# --------------------------------------------------------------------------- #
# Markdown family
# --------------------------------------------------------------------------- #

def format_as_markdown(response: SearchResponse) -> str:
    """Legacy human-readable Markdown. Kept as an alias for --markdown."""
    lines: list[str] = []
    lines.append(f'# Search Results: "{response.query}"')
    lines.append("")
    lines.append(
        f"**Status**: {response.status} | **Total**: {len(response.results)} results | "
        f"**Time**: {response.timing_ms.total_ms}ms"
    )
    lines.append("")
    lines.append("## Indexing")
    stats = response.indexed_now
    lines.append(
        f"- Added: {stats.files_added} files | Updated: {stats.files_updated} | "
        f"Skipped (unchanged): {stats.files_skipped_unchanged}"
    )
    lines.append("")
    if response.results:
        lines.append("## Results")
        lines.append("")
        for result in response.results:
            lines.append(f"### {result.rank}. {result.path} (score: {result.score})")
            parts = [f"**Type**: {result.file_type}", f"**Modality**: {result.modality}"]
            if result.page is not None:
                parts.append(f"**Page**: {result.page}")
            elif result.start_line is not None and result.end_line is not None:
                parts.append(f"**Lines**: {result.start_line}-{result.end_line}")
            lines.append(" | ".join(parts))
            excerpt = result.excerpt or ""
            if len(excerpt) > 200:
                excerpt = excerpt[:200] + "..."
            lines.append(f"> {excerpt}")
            lines.append("")
    if response.skipped_files:
        lines.append("## Skipped Files")
        lines.append("")
        for skipped in response.skipped_files:
            lines.append(f"- {skipped.path}: {skipped.reason}")
        lines.append("")
    if response.errors:
        lines.append("## Errors")
        lines.append("")
        for error in response.errors:
            lines.append(f"- {error}")
        lines.append("")
    return "\n".join(lines)


def format_as_markdown_rich(response: SearchResponse) -> str:
    """Topic-aware Markdown with embedded page images and location blocks."""
    lines: list[str] = []
    lines.append(f'# Search: "{response.query}"')
    lines.append("")
    lines.append(
        f"**Status**: {response.status} · **Results**: {len(response.results)} · "
        f"**Time**: {response.timing_ms.total_ms:.0f} ms"
    )
    lines.append("")
    stats = response.indexed_now
    if stats.files_added or stats.files_updated or stats.files_skipped_unchanged:
        lines.append(
            f"Indexed: +{stats.files_added} added, ~{stats.files_updated} updated, "
            f"={stats.files_skipped_unchanged} unchanged"
        )
        lines.append("")

    if response.results:
        for result in response.results:
            lines.extend(_format_rich_result(result))
            lines.append("")
    else:
        lines.append("_No results._")
        lines.append("")

    if response.skipped_files:
        lines.append("## Skipped")
        for skipped in response.skipped_files:
            lines.append(f"- `{skipped.path}`: {skipped.reason}")
        lines.append("")
    if response.errors:
        lines.append("## Errors")
        for error in response.errors:
            lines.append(f"- {error}")
        lines.append("")

    return "\n".join(lines)


def _format_rich_result(result: SearchResult) -> list[str]:
    lines: list[str] = []
    title = result.title or os.path.basename(result.path)
    header = f"## {result.rank}. {title}  (score {result.score:.3f})"
    lines.append(header)

    meta_bits: list[str] = [f"**File**: `{result.path}`"]
    if result.page_refs:
        meta_bits.append(
            f"**Pages**: {_format_page_refs(result.page_refs)}"
        )
    if result.line_ranges:
        meta_bits.append(
            f"**Lines**: " + ", ".join(f"{a}-{b}" for a, b in result.line_ranges)
        )
    meta_bits.append(f"**Modality**: {result.modality}")
    lines.append(" · ".join(meta_bits))

    if result.summary:
        lines.append(f"_{result.summary}_")
    if result.rerank_reason:
        lines.append(f"> **Rerank reason:** {result.rerank_reason}")

    if result.excerpt:
        excerpt = result.excerpt.strip()
        if len(excerpt) > 600:
            excerpt = excerpt[:600] + "..."
        lines.append("")
        for line in excerpt.split("\n"):
            lines.append(f"> {line}")

    # Inline page images via file:// URIs (markdown clients either render
    # them or show the link text).
    if result.context_pages:
        lines.append("")
        for ctx in result.context_pages:
            if ctx.page_image_path:
                uri = f"file://{ctx.page_image_path}"
                lines.append(f"![page {ctx.page}]({uri})")
            elif ctx.page_image_b64:
                lines.append(f"![page {ctx.page}](data:image/webp;base64,{ctx.page_image_b64})")

    if result.location:
        lines.append("")
        lines.append(f"**Location**: `{result.location.directory}`")
        listing = result.location.directory_listing
        if listing:
            total = result.location.listing_total
            header = (
                f"Directory contents ({total}):"
                if not result.location.listing_truncated
                else f"Directory contents ({total}, truncated):"
            )
            lines.append(header)
            for entry in listing:
                lines.append(f"  - {entry}")

    return lines


def _format_page_refs(page_refs: list[int]) -> str:
    """Turn [1,2,3,7,8] into '1–3, 7–8'."""
    if not page_refs:
        return ""
    sorted_refs = sorted(set(page_refs))
    ranges: list[tuple[int, int]] = []
    start = sorted_refs[0]
    prev = start
    for p in sorted_refs[1:]:
        if p == prev + 1:
            prev = p
            continue
        ranges.append((start, prev))
        start = prev = p
    ranges.append((start, prev))
    parts = [f"{a}" if a == b else f"{a}–{b}" for a, b in ranges]
    return ", ".join(parts)
