"""File state tracking for staleness detection.

Tracks file hashes and metadata to detect when files have changed
and need re-indexing.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import TYPE_CHECKING

from src.models import FileState

if TYPE_CHECKING:
    pass


def compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of file content.

    Args:
        path: Absolute path to file.

    Returns:
        Hex string of SHA-256 hash.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class FileStateTracker:
    """In-memory tracker of file state for staleness detection.

    Stores file hash, size, mtime, and associated point IDs.
    Detects staleness by comparing current hash with stored hash.
    """

    def __init__(self) -> None:
        """Initialize empty state tracker."""
        self._state: dict[str, FileState] = {}

    def check_staleness(self, path: str) -> tuple[bool, FileState]:
        """Check if file needs re-indexing.

        Computes current file hash and compares with stored state.
        A file is stale if:
        - It's new (not in state dict)
        - Its hash has changed

        Args:
            path: Absolute path to file.

        Returns:
            Tuple of (needs_reindex, current_state).
            needs_reindex is True if file is new or modified.
            current_state is the FileState object for the file.
        """
        current_hash = compute_file_hash(path)
        stat = os.stat(path)
        current_state = FileState(
            path=path,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            content_hash_sha256=current_hash,
            last_indexed_at=time.time(),
            point_ids=[],
        )

        # Check if file is new or hash changed
        if path not in self._state:
            return True, current_state

        stored_state = self._state[path]
        if stored_state.content_hash_sha256 != current_hash:
            return True, current_state

        return False, current_state

    def mark_indexed(self, path: str, point_ids: list[str]) -> None:
        """Mark file as indexed with associated point IDs.

        Updates the stored state with the current file hash and point IDs.

        Args:
            path: Absolute path to file.
            point_ids: List of point IDs from vector database.
        """
        current_hash = compute_file_hash(path)
        stat = os.stat(path)
        self._state[path] = FileState(
            path=path,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            content_hash_sha256=current_hash,
            last_indexed_at=time.time(),
            point_ids=point_ids,
        )

    def get_point_ids(self, path: str) -> list[str]:
        """Get stored point IDs for a file.

        Args:
            path: Absolute path to file.

        Returns:
            List of point IDs, or empty list if file not indexed.
        """
        if path not in self._state:
            return []
        return self._state[path].point_ids
