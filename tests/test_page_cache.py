"""Tests for the PageImageCache."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from PIL import Image

from ragrag.index.page_cache import PageImageCache


def _img(color: str = "white", size: tuple[int, int] = (80, 60)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def test_put_then_get_roundtrip(tmp_path: Path) -> None:
    cache = PageImageCache(tmp_path)
    sha = "a" * 64
    result = cache.put(sha, 1, _img("red"))
    assert result.is_file()
    assert result.suffix == ".webp"
    assert cache.has(sha, 1)
    loaded = cache.get(sha, 1)
    assert loaded == result
    assert cache.get(sha, 99) is None


def test_put_is_idempotent(tmp_path: Path) -> None:
    cache = PageImageCache(tmp_path)
    sha = "b" * 64
    p1 = cache.put(sha, 1, _img("blue"))
    size1 = p1.stat().st_size
    p2 = cache.put(sha, 1, _img("green"))  # should not rewrite
    assert p1 == p2
    assert p1.stat().st_size == size1


def test_path_layout_uses_sha_prefix(tmp_path: Path) -> None:
    cache = PageImageCache(tmp_path)
    sha = "abcd" + ("f" * 60)
    p = cache.put(sha, 7, _img())
    assert p.parent.parent.name == "ab"
    assert p.parent.name == sha
    assert p.name == "7.webp"


def test_evict_file_drops_everything(tmp_path: Path) -> None:
    cache = PageImageCache(tmp_path)
    sha = "c" * 64
    for page in range(1, 6):
        cache.put(sha, page, _img())
    cache.evict_file(sha)
    for page in range(1, 6):
        assert not cache.has(sha, page)


def test_lru_sweep_trims_when_over_cap(tmp_path: Path) -> None:
    """LRU eviction under a cap: oldest files get dropped first."""
    # Write a handful of images, then manually invoke the sweep with a tiny
    # cap so the oldest files are evicted deterministically.
    cache = PageImageCache(tmp_path, max_mb=1024)
    sha = "d" * 64
    p1 = cache.put(sha, 1, _img())
    time.sleep(0.01)
    p2 = cache.put(sha, 2, _img())
    time.sleep(0.01)
    p3 = cache.put(sha, 3, _img())
    assert p1.is_file() and p2.is_file() and p3.is_file()

    # Shrink the cap to one file's worth and trigger the sweep.
    cache.max_bytes = p3.stat().st_size
    cache._maybe_evict()

    # The newest file survives; older ones are gone.
    assert p3.is_file()
    assert not p1.is_file()


def test_put_creates_directories(tmp_path: Path) -> None:
    cache = PageImageCache(tmp_path / "nested" / "deeper")
    sha = "e" * 64
    p = cache.put(sha, 42, _img())
    assert p.is_file()


def test_size_bytes_nonzero_after_put(tmp_path: Path) -> None:
    cache = PageImageCache(tmp_path)
    assert cache.size_bytes() == 0
    cache.put("f" * 64, 1, _img())
    assert cache.size_bytes() > 0
