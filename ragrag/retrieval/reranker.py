"""VLM-backed listwise reranker (Phase D).

The reranker lives in a long-lived child process so the VLM load
cost is paid once per daemon session rather than once per query.
The parent manages the worker lifecycle: spawn on first use,
reuse across subsequent queries, and tear down on shutdown (or on
a protocol error).

On tight GPUs that cannot fit both ColQwen3 and the VLM
simultaneously, the reranker is simply not enabled: the daemon's
engine cache only instantiates a reranker when
``settings.reranker_model`` is set to something other than ``none``.
The roadmap calls this out as acceptable fallback behaviour.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from typing import Any, Optional

from ragrag.config import Settings
from ragrag.models import SearchResult


logger = logging.getLogger(__name__)


class RerankerError(RuntimeError):
    """Raised when the reranker subprocess fails in a way the caller
    cannot recover from automatically."""


class VLMReranker:
    """Wraps a long-lived :mod:`ragrag.retrieval.vlm_rerank_worker`.

    Thread-safe: the rerank call is serialised behind a mutex so the
    daemon's small RPC thread pool cannot interleave writes on the
    worker's stdin. One process per reranker instance; call
    :meth:`close` when tearing down.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen[str]] = None
        self._ready: bool = False
        self._model_id = settings.vlm_model_id
        self._quantization = settings.vlm_quantization
        self._image_max_side = int(getattr(settings, "rerank_image_max_side", 640))
        self._max_new_tokens = int(getattr(settings, "rerank_max_new_tokens", 384))
        self._oversample = int(getattr(settings, "rerank_oversample", 3))
        self._max_candidates = int(getattr(settings, "rerank_max_candidates", 10))

    # ------------------------------------------------------------------ #
    # Worker lifecycle
    # ------------------------------------------------------------------ #

    def _spawn(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = [
            sys.executable,
            "-m",
            "ragrag.retrieval.vlm_rerank_worker",
            "--model-id", self._model_id,
            "--quantization", self._quantization,
            "--image-max-side", str(self._image_max_side),
            "--max-new-tokens", str(self._max_new_tokens),
        ]
        logger.info("Spawning VLM rerank worker: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr inherits from parent so the user sees the model
            # load + any warnings in the normal log stream.
            stderr=None,
            text=True,
            bufsize=1,  # line buffered
        )
        # The worker emits a {"status": "ready"} (or "fatal") line after
        # the weights are loaded. Block here so the first rerank call
        # does not race the load.
        ready_line = self._proc.stdout.readline() if self._proc.stdout else ""
        if not ready_line:
            raise RerankerError("rerank worker exited before emitting ready")
        try:
            payload = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            raise RerankerError(f"rerank worker emitted non-JSON ready line: {ready_line!r}") from exc
        if payload.get("status") == "fatal":
            raise RerankerError(f"rerank worker fatal: {payload.get('error')}")
        if payload.get("status") != "ready":
            raise RerankerError(f"unexpected ready payload: {payload}")
        self._ready = True
        logger.info(
            "VLM rerank worker ready (model=%s, device=%s)",
            payload.get("model_id"), payload.get("device"),
        )

    def close(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            try:
                if self._proc.poll() is None and self._proc.stdin:
                    self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=3.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            self._ready = False

    # ------------------------------------------------------------------ #
    # Public rerank entry point
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Return a re-ordered copy of ``results``.

        On any failure, fall back to the input order with no changes.
        The caller should always trust the return value.
        """
        if not results:
            return results
        candidates = results[: self._max_candidates]
        with self._lock:
            try:
                self._spawn()
                ranks = self._send_rerank(query, candidates)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM rerank failed: %s — falling back to MaxSim order", exc)
                self.close()
                return results

        if not ranks:
            return results

        # Reorder by the worker's rank assignment; unranked candidates
        # keep their original relative order at the end.
        rank_by_id = {r["id"]: r for r in ranks}
        ordered: list[SearchResult] = []
        leftover: list[SearchResult] = []
        sorted_ranks = sorted(
            (r for r in ranks if r["id"] < len(candidates)),
            key=lambda r: (r.get("rank") or 999, -float(r.get("score") or 0)),
        )
        placed: set[int] = set()
        for r in sorted_ranks:
            i = int(r["id"])
            if 0 <= i < len(candidates) and i not in placed:
                placed.add(i)
                merged = candidates[i].model_copy(update={"rerank_reason": r.get("reason") or None})
                ordered.append(merged)
        for i, c in enumerate(candidates):
            if i not in placed:
                leftover.append(c)

        # Re-rank only affects the slice we sent to the worker; the
        # rest of results stays in the original MaxSim order.
        tail = results[self._max_candidates:]
        merged_top = ordered + leftover
        return _reassign_ranks(merged_top + tail)

    # ------------------------------------------------------------------ #
    # Internal: request / response
    # ------------------------------------------------------------------ #

    def _send_rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[dict[str, Any]]:
        payload_candidates: list[dict[str, Any]] = []
        for i, c in enumerate(candidates):
            pages = _format_pages(c.page_refs or [])
            image_path = _pick_hero_image(c)
            payload_candidates.append(
                {
                    "id": i,
                    "title": c.title or "",
                    "summary": c.summary or "",
                    "pages": pages,
                    "excerpt": (c.excerpt or "")[:400],
                    "image_path": image_path,
                }
            )

        request = {
            "cmd": "rerank",
            "query": query,
            "candidates": payload_candidates,
        }
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline()
        if not response_line:
            raise RerankerError("rerank worker closed stdout")
        payload = json.loads(response_line)
        if payload.get("status") == "error":
            raise RerankerError(f"worker rerank error: {payload.get('error')}")
        if payload.get("status") != "ok":
            raise RerankerError(f"unexpected rerank response: {payload}")
        ranks = payload.get("ranks") or []
        if not isinstance(ranks, list):
            raise RerankerError("rerank response missing 'ranks'")
        return ranks


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _format_pages(pages: list[int]) -> str:
    if not pages:
        return ""
    sorted_pages = sorted(set(pages))
    ranges: list[str] = []
    start = prev = sorted_pages[0]
    for p in sorted_pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = p
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    joined = ", ".join(ranges)
    return joined[:100]


def _pick_hero_image(result: SearchResult) -> Optional[str]:
    """Return the filesystem path of a representative image for the result."""
    if result.context_pages:
        for page_ctx in result.context_pages:
            if page_ctx.page_image_path:
                return page_ctx.page_image_path
    return None


def _reassign_ranks(results: list[SearchResult]) -> list[SearchResult]:
    out: list[SearchResult] = []
    for idx, r in enumerate(results, start=1):
        out.append(r.model_copy(update={"rank": idx}))
    return out
