"""Unit tests for the Location builder."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ragrag.retrieval.location_builder import build_location


def _touch(dir_: Path, names: list[str]) -> None:
    for n in names:
        (dir_ / n).write_text("x", encoding="utf-8")


def test_build_location_small_dir(tmp_path: Path) -> None:
    names = ["a.c", "b.c", "c.h"]
    _touch(tmp_path, names)
    loc = build_location(str(tmp_path / "a.c"))
    assert loc.path == str((tmp_path / "a.c").resolve()) or loc.path.endswith("a.c")
    assert loc.directory == str(tmp_path)
    assert set(loc.directory_listing) == set(names)
    assert loc.listing_truncated is False
    assert loc.listing_total == 3


def test_build_location_flags_directories_with_slash(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "f.txt").write_text("x")
    loc = build_location(str(tmp_path / "f.txt"))
    assert "sub/" in loc.directory_listing
    assert "f.txt" in loc.directory_listing


def test_build_location_truncates_large_listing(tmp_path: Path) -> None:
    names = [f"file_{i:03d}.txt" for i in range(100)]
    _touch(tmp_path, names)
    loc = build_location(str(tmp_path / "file_000.txt"), max_entries=10)
    assert loc.listing_truncated is True
    assert loc.listing_total == 100
    # Listing is head + '...' + tail
    assert "..." in loc.directory_listing
    assert len(loc.directory_listing) == 11  # 5 head + ... + 5 tail


def test_build_location_respects_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("build\n*.log\n", encoding="utf-8")
    (tmp_path / "main.c").write_text("x")
    (tmp_path / "out.log").write_text("x")
    (tmp_path / "build").mkdir()

    loc = build_location(str(tmp_path / "main.c"), respect_gitignore=True)
    names = set(loc.directory_listing)
    assert "main.c" in names
    assert "out.log" not in names
    assert "build/" not in names


def test_build_location_ignores_gitignore_when_disabled(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (tmp_path / "a.log").write_text("x")
    (tmp_path / "b.txt").write_text("x")

    loc = build_location(str(tmp_path / "b.txt"), respect_gitignore=False)
    assert "a.log" in loc.directory_listing


def test_build_location_inherits_parent_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "secret.txt").write_text("x")
    (sub / "public.txt").write_text("x")

    loc = build_location(str(sub / "public.txt"), respect_gitignore=True)
    names = set(loc.directory_listing)
    assert "public.txt" in names
    # Parent .gitignore rules apply to the subdirectory too
    assert "secret.txt" not in names


def test_build_location_handles_unreadable_directory(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")

    def _fail(_path):
        raise PermissionError("nope")

    # Force the scandir to fail; the builder should still return a valid Location.
    monkeypatch.setattr("os.scandir", _fail)
    loc = build_location(str(f))
    assert loc.directory_listing == []
    assert loc.listing_total == 0
