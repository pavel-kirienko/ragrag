"""File discovery with filtering and staleness tracking.

Discovers files from input paths, applies filters (hidden files, symlinks,
extensions, limits), and returns absolute paths with skip reasons.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ragrag.models import SkippedFile

if TYPE_CHECKING:
    from ragrag.config import Settings


# ragrag's own config files should never end up in the search index even
# when they live next to the corpus. Indexing them clutters the store
# with tiny JSON rows that will occasionally rank near the top and
# confuse the benchmark. Hidden-file filtering already drops
# ``.ragrag.json`` but ``ragrag.json`` sneaks through.
_RAGRAG_CONFIG_BASENAMES: frozenset[str] = frozenset({"ragrag.json", ".ragrag.json"})


def discover_files(
    paths: list[str], settings: Settings
) -> tuple[list[str], list[SkippedFile]]:
    """Discover files from input paths with filtering.

    Args:
        paths: List of file or directory paths to discover.
        settings: Settings object with filtering configuration.

    Returns:
        Tuple of (list of absolute file paths, list of SkippedFile objects).
        Files are absolute paths. Skipped files include reason for skip.
    """
    discovered: list[str] = []
    skipped: list[SkippedFile] = []

    for path in paths:
        abs_path = os.path.abspath(path)
        real_path = os.path.realpath(abs_path)

        if os.path.isfile(abs_path):
            # Single file: check file type
            if os.path.basename(real_path) in _RAGRAG_CONFIG_BASENAMES:
                skipped.append(
                    SkippedFile(path=real_path, reason="ragrag config file")
                )
            elif not _is_supported_file(real_path):
                skipped.append(
                    SkippedFile(path=real_path, reason="unsupported file type")
                )
            else:
                discovered.append(real_path)

        elif os.path.isdir(abs_path):
            # Directory: recursive walk
            _walk_directory(
                abs_path, discovered, skipped, settings
            )

        else:
            skipped.append(SkippedFile(path=real_path, reason="not a file or directory"))

    return discovered, skipped


def _walk_directory(
    dir_path: str,
    discovered: list[str],
    skipped: list[SkippedFile],
    settings: Settings,
) -> None:
    """Recursively walk directory and collect files."""
    try:
        for root, dirs, files in os.walk(
            dir_path, followlinks=settings.follow_symlinks
        ):
            # Filter directories in-place to control recursion
            if not settings.include_hidden:
                dirs[:] = [d for d in dirs if not d.startswith(".")]

            # Process files
            for filename in files:
                if not settings.include_hidden and filename.startswith("."):
                    continue

                if filename in _RAGRAG_CONFIG_BASENAMES:
                    # Silently drop ragrag's own config rather than
                    # adding it to skipped[] — the user did not ask to
                    # index their own knobs file.
                    continue

                file_path = os.path.join(root, filename)
                real_path = os.path.realpath(file_path)

                # Check if symlink and follow_symlinks is False
                if os.path.islink(file_path) and not settings.follow_symlinks:
                    skipped.append(SkippedFile(path=real_path, reason="symlink"))
                    continue

                # Check file type
                if not _is_supported_file(real_path):
                    skipped.append(
                        SkippedFile(path=real_path, reason="unsupported file type")
                    )
                    continue

                discovered.append(real_path)

    except (OSError, PermissionError) as e:
        skipped.append(SkippedFile(path=dir_path, reason=f"walk error: {e}"))


def _is_supported_file(path: str) -> bool:
    from ragrag.models import get_file_type
    return get_file_type(path) is not None
