"""File state tracking for staleness detection.

Tracks file hashes and metadata to detect when files have changed
and need re-indexing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import TYPE_CHECKING, cast

from src.models import FileState

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


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
    """Tracker of file state for staleness detection.

    Stores file hash, size, mtime, and associated point IDs.
    Detects staleness by comparing current hash with stored hash.
    Persists state to disk at ``{index_path}/file_state.json``.
    """

    def __init__(self, index_path: str) -> None:
        """Initialize tracker and load persisted state if available."""
        self._state_file: str = os.path.join(index_path, "file_state.json")
        self._state: dict[str, FileState]
        if os.path.isfile(self._state_file):
            try:
                with open(self._state_file, encoding="utf-8") as f:
                    raw_data = cast(object, json.load(f))
                if not isinstance(raw_data, dict):
                    raise ValueError("invalid file state format")
                raw = cast(dict[str, object], raw_data)

                state: dict[str, FileState] = {}
                for key, value in raw.items():
                    state[key] = FileState.model_validate(value)
                self._state = state
            except Exception:
                logger.warning("Corrupt file_state.json, starting fresh")
                self._state = {}
        else:
            self._state = {}

    def save(self) -> None:
        tmp = self._state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: v.model_dump() for k, v in self._state.items()}, f)
        os.replace(tmp, self._state_file)

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

    def mark_indexed(
        self,
        path: str,
        point_ids: list[str],
        file_state: FileState | None = None,
    ) -> None:
        """Mark file as indexed with associated point IDs.

        Updates the stored state with the current file hash and point IDs.

        Args:
            path: Absolute path to file.
            point_ids: List of point IDs from vector database.
            file_state: Precomputed file state from ``check_staleness``.
        """
        if file_state is None:
            current_hash = compute_file_hash(path)
            stat = os.stat(path)
            file_state = FileState(
                path=path,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                content_hash_sha256=current_hash,
                last_indexed_at=time.time(),
                point_ids=point_ids,
            )
        else:
            file_state = file_state.model_copy(update={"point_ids": point_ids})

        self._state[path] = file_state
        self.save()

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
