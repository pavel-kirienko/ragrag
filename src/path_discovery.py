"""File discovery with filtering and staleness tracking.

Discovers files from input paths, applies filters (hidden files, symlinks,
extensions, roots, limits), and returns absolute paths with skip reasons.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.models import SUPPORTED_EXTENSIONS, SkippedFile

if TYPE_CHECKING:
    from src.config import Settings


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
            # Single file: check extension and roots
            if not _is_supported_file(real_path):
                skipped.append(
                    SkippedFile(path=real_path, reason="unsupported extension")
                )
            elif not _is_allowed_root(real_path, settings.ALLOWED_ROOTS):
                skipped.append(
                    SkippedFile(path=real_path, reason="outside allowed roots")
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

    # Apply MAX_FILES_PER_REQUEST limit
    if len(discovered) > settings.MAX_FILES_PER_REQUEST:
        truncated = discovered[settings.MAX_FILES_PER_REQUEST :]
        discovered = discovered[: settings.MAX_FILES_PER_REQUEST]
        for path in truncated:
            skipped.append(SkippedFile(path=path, reason="limit reached"))

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
            dir_path, followlinks=settings.FOLLOW_SYMLINKS
        ):
            # Filter directories in-place to control recursion
            if not settings.INCLUDE_HIDDEN_FILES:
                dirs[:] = [d for d in dirs if not d.startswith(".")]

            # Process files
            for filename in files:
                if not settings.INCLUDE_HIDDEN_FILES and filename.startswith("."):
                    continue

                file_path = os.path.join(root, filename)
                real_path = os.path.realpath(file_path)

                # Check if symlink and FOLLOW_SYMLINKS is False
                if os.path.islink(file_path) and not settings.FOLLOW_SYMLINKS:
                    skipped.append(SkippedFile(path=real_path, reason="symlink"))
                    continue

                # Check extension
                if not _is_supported_file(real_path):
                    skipped.append(
                        SkippedFile(path=real_path, reason="unsupported extension")
                    )
                    continue

                # Check allowed roots
                if not _is_allowed_root(real_path, settings.ALLOWED_ROOTS):
                    skipped.append(
                        SkippedFile(path=real_path, reason="outside allowed roots")
                    )
                    continue

                discovered.append(real_path)

    except (OSError, PermissionError) as e:
        skipped.append(SkippedFile(path=dir_path, reason=f"walk error: {e}"))


def _is_supported_file(path: str) -> bool:
    """Check if file extension is supported."""
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def _is_allowed_root(path: str, allowed_roots: list[str] | None) -> bool:
    """Check if path is within allowed roots (if configured)."""
    if allowed_roots is None:
        return True
    return any(path.startswith(root) for root in allowed_roots)
