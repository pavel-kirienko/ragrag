"""Parent-side wrapper that runs the VLM topic chunker in a subprocess.

The in-process approach of unloading the embedder, loading the VLM,
planning chunks, unloading the VLM, and reloading the embedder is
fragile on 8 GB cards: bnb 4-bit leaves non-PyTorch CUDA context
state behind after the unload, which corrupts the allocator for the
next load. Running the VLM in a child process sidesteps this — the
driver reclaims the child's entire CUDA context on exit, so the
parent's allocator stays pristine and the embedder loads cleanly.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from ragrag.config import Settings
from ragrag.models import Chunk, FileType


logger = logging.getLogger(__name__)


class SubprocessVLMPlannerError(RuntimeError):
    """Raised when the VLM subprocess worker fails in a way the parent
    cannot recover from (non-zero exit, unreadable output, etc.).
    """


class SubprocessVLMPlanner:
    """Run :mod:`ragrag.extractors.vlm_topic_worker` as a one-shot child.

    The caller hands over a batch of files and receives back a dict
    mapping each file path to its plan result (either a ``list[Chunk]``
    on success, or an error string on failure). The whole lifecycle of
    the VLM — load, every plan call, unload — happens inside the
    child. The parent process never imports the VLM model class and
    never touches CUDA until the embed phase.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def plan_files(
        self, files: list[tuple[str, str, FileType]]
    ) -> dict[str, Any]:
        """Plan chunks for a batch of files.

        Args:
            files: list of ``(path, sha256, file_type)`` tuples. Only
                ``FileType.PDF`` and ``FileType.TEXT`` are supported;
                image files are not chunked by the VLM.

        Returns:
            ``{path: list[Chunk] | {"error": str}}``. Paths are the
            same strings passed in (not resolved).
        """
        if not files:
            return {}

        request_files = []
        for path, sha256, ftype in files:
            if ftype == FileType.PDF:
                type_str = "pdf"
            elif ftype == FileType.TEXT:
                type_str = "text"
            else:
                continue
            request_files.append({"path": path, "sha256": sha256, "type": type_str})

        if not request_files:
            return {}

        request = {
            "settings": self.settings.model_dump(mode="json"),
            "files": request_files,
        }

        cmd = [sys.executable, "-m", "ragrag.extractors.vlm_topic_worker"]
        logger.info(
            "Spawning VLM topic worker for %d file(s): %s",
            len(request_files), " ".join(cmd),
        )
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(request),
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SubprocessVLMPlannerError(f"python interpreter not found: {exc}") from exc

        if proc.stderr:
            for line in proc.stderr.splitlines():
                if line.strip():
                    logger.debug("vlm-worker: %s", line)

        if proc.returncode != 0:
            raise SubprocessVLMPlannerError(
                f"VLM worker exited with code {proc.returncode}; "
                f"stderr tail: {proc.stderr[-400:] if proc.stderr else '(none)'}"
            )

        results: dict[str, Any] = {}
        done_seen = False
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("VLM worker emitted non-JSON line: %.200s", line)
                continue

            status = payload.get("status")
            if status == "fatal":
                raise SubprocessVLMPlannerError(
                    f"VLM worker fatal: {payload.get('error') or 'unknown'}"
                )
            if status == "done":
                done_seen = True
                continue

            path = payload.get("path")
            if not isinstance(path, str):
                continue
            if status == "ok":
                chunk_dicts = payload.get("chunks") or []
                try:
                    chunks = [Chunk.model_validate(d) for d in chunk_dicts]
                except Exception as exc:  # noqa: BLE001
                    results[path] = {"error": f"chunk validate failed: {exc}"}
                    continue
                results[path] = chunks
            elif status == "error":
                results[path] = {"error": payload.get("error") or "unknown"}

        if not done_seen:
            raise SubprocessVLMPlannerError(
                "VLM worker did not emit the done sentinel — output was truncated"
            )

        return results
