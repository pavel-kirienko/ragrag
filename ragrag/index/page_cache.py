"""On-disk cache of rendered PDF page images.

PDF pages are cached as WebP files keyed by ``(file_sha256, page_number)``.
The cache is write-once per page — ingestion writes each page before the
PIL image is dropped, and search reads the file back to attach to result
``context_pages``. Size is capped by a soft LRU sweep that deletes the
oldest-atime files when the directory exceeds ``page_cache_max_mb``.

Layout::

    <root>/<sha[:2]>/<sha>/<page>.webp
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image


logger = logging.getLogger(__name__)


class PageImageCache:
    """WebP-backed page-image cache with LRU eviction."""

    def __init__(self, root: str | os.PathLike[str], *, max_mb: int = 1024) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_mb) * 1024 * 1024
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    def _path_for(self, sha: str, page: int) -> Path:
        safe_sha = "".join(c for c in sha if c.isalnum())
        return self.root / safe_sha[:2] / safe_sha / f"{int(page)}.webp"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def has(self, sha: str, page: int) -> bool:
        return self._path_for(sha, page).is_file()

    def get(self, sha: str, page: int) -> Optional[Path]:
        p = self._path_for(sha, page)
        if p.is_file():
            try:
                os.utime(p, None)  # bump atime/mtime so LRU sweep keeps it
            except OSError:
                pass
            return p
        return None

    def put(
        self,
        sha: str,
        page: int,
        image: Image.Image,
        *,
        quality: int = 85,
        method: int = 6,
    ) -> Path:
        """Write ``image`` to the cache. Idempotent: if the file exists we
        keep the old one and bump its mtime."""
        p = self._path_for(sha, page)
        if p.exists():
            try:
                os.utime(p, None)
            except OSError:
                pass
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".webp.tmp")
        try:
            image.save(tmp, format="WEBP", quality=quality, method=method)
            os.replace(tmp, p)
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise
        self._maybe_evict()
        return p

    def evict_file(self, sha: str) -> None:
        """Drop every cached page for a given source file."""
        safe_sha = "".join(c for c in sha if c.isalnum())
        file_dir = self.root / safe_sha[:2] / safe_sha
        if not file_dir.is_dir():
            return
        for entry in file_dir.iterdir():
            try:
                entry.unlink()
            except OSError:
                pass
        try:
            file_dir.rmdir()
        except OSError:
            pass

    def size_bytes(self) -> int:
        total = 0
        for entry in _walk_files(self.root):
            try:
                total += entry.stat().st_size
            except OSError:
                pass
        return total

    # ------------------------------------------------------------------ #
    # Eviction
    # ------------------------------------------------------------------ #

    def _maybe_evict(self) -> None:
        with self._lock:
            total = self.size_bytes()
            if total <= self.max_bytes:
                return
            logger.info(
                "Page cache %d MB > %d MB, sweeping LRU",
                total // (1024 * 1024), self.max_bytes // (1024 * 1024),
            )
            # Sort by mtime ascending; delete oldest until under cap.
            files: list[Path] = list(_walk_files(self.root))
            files.sort(key=lambda p: _safe_mtime(p))
            for entry in files:
                if total <= self.max_bytes:
                    break
                try:
                    total -= entry.stat().st_size
                    entry.unlink()
                except OSError:
                    pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _walk_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
