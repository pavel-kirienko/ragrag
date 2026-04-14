"""Build the ``Location`` block attached to every search result.

``Location`` is deliberately minimal: it tells the LLM consumer where a
hit lives on disk and what other files sit alongside it. No cross-
document graph, no include/import parsing, no fuzzy sibling match. If
the consumer wants to look at a neighbour, it can issue another ragrag
query.

The block is computed on the search path (not at index time), so it
reflects the current filesystem state every call. One ``os.scandir``
per distinct result path; no persistent cache.
"""
from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Iterable

from ragrag.models import Location


logger = logging.getLogger(__name__)


def build_location(
    path: str,
    *,
    max_entries: int = 64,
    respect_gitignore: bool = True,
) -> Location:
    """Construct a ``Location`` for ``path``.

    Args:
        path: absolute (or abspath-able) path to the source file hit.
        max_entries: cap on the directory listing. When the directory has
            more entries we return the first ``max_entries // 2`` and the
            last ``max_entries // 2`` by alphabetical order, joined by an
            ellipsis marker so the consumer knows the list is truncated.
        respect_gitignore: if True and a ``.gitignore`` sits in the
            directory (or an ancestor up to the filesystem root), entries
            matching any ignore pattern are excluded from the listing.

    Errors walking the directory return an otherwise-valid ``Location``
    with an empty listing — we never let a filesystem quirk break a
    search response.
    """
    abs_path = os.path.abspath(path)
    directory = os.path.dirname(abs_path)

    entries = _scandir_safe(directory)
    if respect_gitignore:
        patterns = _collect_gitignore_patterns(directory)
        if patterns:
            entries = [e for e in entries if not _matches_any(e, patterns)]

    entries.sort(key=str.lower)
    total = len(entries)
    truncated = False
    if total > max_entries:
        half = max_entries // 2
        head = entries[:half]
        tail = entries[-half:]
        listing = head + ["..."] + tail
        truncated = True
    else:
        listing = entries

    return Location(
        path=abs_path,
        directory=directory,
        directory_listing=listing,
        listing_truncated=truncated,
        listing_total=total,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _scandir_safe(directory: str) -> list[str]:
    try:
        with os.scandir(directory) as it:
            result: list[str] = []
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        result.append(entry.name + "/")
                    else:
                        result.append(entry.name)
                except OSError:
                    result.append(entry.name)
            return result
    except (OSError, PermissionError) as exc:
        logger.debug("location builder could not scan %s: %s", directory, exc)
        return []


def _collect_gitignore_patterns(start_dir: str) -> list[str]:
    """Walk upward from ``start_dir`` collecting ``.gitignore`` patterns.

    Returns a flat list of glob patterns. This is intentionally simpler
    than git's real matching — we only support plain globs, not negation,
    not directory-relative rooted patterns. The goal is "don't leak
    .git/ or build/ in a listing", not full gitignore compatibility.
    """
    patterns: list[str] = []
    try:
        current = Path(start_dir).resolve()
    except OSError:
        return patterns
    root = current.anchor
    while True:
        gitignore = current / ".gitignore"
        if gitignore.is_file():
            try:
                with open(gitignore, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("!"):
                            continue
                        patterns.append(line.rstrip("/"))
            except OSError:
                pass
        if str(current) == root or current.parent == current:
            break
        current = current.parent
    return patterns


def _matches_any(entry: str, patterns: Iterable[str]) -> bool:
    name = entry.rstrip("/")
    for pat in patterns:
        if fnmatch.fnmatch(name, pat):
            return True
    return False
