"""Local mmap-backed multivector store for ragrag.

The class is still named ``QdrantStore`` for legacy reasons (every caller
imports it by that name) but it no longer uses Qdrant. It implements the
same retrieval contract — multivector MaxSim brute-force — over a small
append-only on-disk format that is:

  * **mmap-cheap on startup** — opening a 100 k-point store is O(1)
    instead of O(N) deserialization. Qdrant's ``LocalCollection``
    deserialised every pickled point into RAM at ``__init__``, which made
    a 4 k-point datasheet index unloadable on a 32 GiB box.
  * **streaming-friendly during ingest** — each ``upsert`` appends to
    three flat files; nothing is buffered in process memory. Indexing a
    600-page PDF no longer spikes RSS to 20 GiB.
  * **fast at query time** — MaxSim is one ``query @ all_tokens.T`` BLAS
    matmul followed by a single ``np.maximum.reduceat`` per-document max,
    then a sum across query tokens. Sub-100 ms on a 4 k-point datasheet.

On-disk layout (under ``<index_path>/store/<collection_name>/``):

  ``manifest.json``     — dim, num_rows, schema version
  ``payloads.jsonl``    — one JSON object per row, in row order
  ``offsets.bin``       — int64 array, shape (num_rows, 2): (token_offset, n_tokens)
  ``vectors.bin``       — float32 array, shape (sum_n_tokens, dim), C-contiguous
  ``deleted.json``      — sorted list of soft-deleted row indices
  ``id_index.json``     — segment_id → row index

Old Qdrant ``collection/`` directories from earlier ragrag versions are
ignored — the user has to re-index with ``--new`` once after the upgrade.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ragrag.models import MultiVector, Segment

logger = logging.getLogger(__name__)


COLLECTION_NAME = "ragrag_segments"

_MANIFEST_VERSION = 1


@dataclass
class ScoredPoint:
    """Minimal duck-typed replacement for the old server-style scored point.

    The only attributes the rest of the codebase touches are ``score`` and
    ``payload`` (see ``ragrag/retrieval/search_engine.py``).
    """
    id: str
    score: float
    payload: dict[str, Any]


class QdrantStore:
    """Append-only multivector store with mmap'd vector pages.

    The legacy class name is preserved because the surrounding code still
    imports ``QdrantStore``; under the hood there is no Qdrant.
    """

    def __init__(self, path: str, collection_name: str, embedding_dim: int = 320) -> None:
        self.collection_name = collection_name
        self.embedding_dim = int(embedding_dim)
        self._dir = Path(path) / "store" / collection_name
        self._dir.mkdir(parents=True, exist_ok=True)

        self._manifest_path = self._dir / "manifest.json"
        self._payloads_path = self._dir / "payloads.jsonl"
        self._offsets_path = self._dir / "offsets.bin"
        self._vectors_path = self._dir / "vectors.bin"
        self._deleted_path = self._dir / "deleted.json"
        self._id_index_path = self._dir / "id_index.json"

        self._payloads: list[dict[str, Any]] = []
        self._offsets: np.ndarray = np.zeros((0, 2), dtype=np.int64)
        self._deleted: set[int] = set()
        self._id_to_idx: dict[str, int] = {}
        # Cache of mmap'd vectors; invalidated whenever vectors.bin grows.
        self._vectors_mmap: np.ndarray | None = None
        self._vectors_mmap_size: int = -1

        self._load_or_init()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_or_init(self) -> None:
        if not self._manifest_path.exists():
            self._save_manifest()
            return

        with open(self._manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("version") != _MANIFEST_VERSION:
            raise RuntimeError(
                f"Unsupported store schema version {manifest.get('version')!r} "
                f"in {self._manifest_path}; expected {_MANIFEST_VERSION}. "
                "Re-create the index with --new."
            )
        if int(manifest.get("dim", -1)) != self.embedding_dim:
            raise RuntimeError(
                f"Embedding dim mismatch: store has {manifest.get('dim')}, "
                f"caller wants {self.embedding_dim}. Re-create with --new."
            )

        if self._payloads_path.exists():
            with open(self._payloads_path, encoding="utf-8") as f:
                self._payloads = [json.loads(line) for line in f if line.strip()]
        if self._offsets_path.exists() and self._offsets_path.stat().st_size > 0:
            self._offsets = np.fromfile(self._offsets_path, dtype=np.int64).reshape(-1, 2)
        if self._deleted_path.exists():
            with open(self._deleted_path, encoding="utf-8") as f:
                self._deleted = set(int(x) for x in json.load(f))
        if self._id_index_path.exists():
            with open(self._id_index_path, encoding="utf-8") as f:
                self._id_to_idx = {str(k): int(v) for k, v in json.load(f).items()}
        else:
            # Rebuild from payloads order if the cache is missing.
            self._id_to_idx = {p["segment_id"]: i for i, p in enumerate(self._payloads)}

        if len(self._payloads) != len(self._offsets):
            raise RuntimeError(
                f"Store {self._dir} is corrupt: {len(self._payloads)} payloads vs "
                f"{len(self._offsets)} offsets. Re-create with --new."
            )

    def _save_manifest(self) -> None:
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": _MANIFEST_VERSION,
                    "dim": self.embedding_dim,
                    "num_rows": len(self._payloads),
                },
                f,
            )

    def _save_deleted(self) -> None:
        with open(self._deleted_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self._deleted), f)

    def _save_id_index(self) -> None:
        with open(self._id_index_path, "w", encoding="utf-8") as f:
            json.dump(self._id_to_idx, f)

    # ------------------------------------------------------------------
    # mmap cache
    # ------------------------------------------------------------------

    def _get_vectors_view(self) -> np.ndarray:
        if not self._vectors_path.exists() or self._vectors_path.stat().st_size == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        size = self._vectors_path.stat().st_size
        if self._vectors_mmap is None or self._vectors_mmap_size != size:
            mm = np.memmap(self._vectors_path, dtype=np.float32, mode="r")
            self._vectors_mmap = mm.reshape(-1, self.embedding_dim)
            self._vectors_mmap_size = size
        return self._vectors_mmap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, segment: Segment, vector: MultiVector) -> None:
        """Insert or update a single segment with its multivector embedding."""
        self.upsert_many([(segment, vector)])

    def upsert_many(self, pairs: list[tuple[Segment, MultiVector]]) -> None:
        """Append-only upsert for a batch of (segment, multivector) pairs.

        If a ``segment_id`` already exists in the store, the previous row is
        soft-deleted and the new vector is appended at the end. This keeps
        ingest O(1) per segment without rewriting the vectors file.
        """
        if not pairs:
            return

        # Compute the starting token offset by inspecting the file size — that
        # is the canonical end-of-data position regardless of in-memory state.
        if self._vectors_path.exists():
            cur_token_offset = self._vectors_path.stat().st_size // (self.embedding_dim * 4)
        else:
            cur_token_offset = 0

        new_offsets_rows: list[tuple[int, int]] = []
        with open(self._vectors_path, "ab") as fv, \
             open(self._offsets_path, "ab") as fo, \
             open(self._payloads_path, "a", encoding="utf-8") as fp:
            for segment, vector in pairs:
                arr = np.ascontiguousarray(np.asarray(vector, dtype=np.float32))
                if arr.ndim != 2 or arr.shape[1] != self.embedding_dim:
                    raise ValueError(
                        f"multivector shape {arr.shape} does not match dim {self.embedding_dim}"
                    )
                n_tokens = int(arr.shape[0])
                if n_tokens == 0:
                    raise ValueError("refusing to upsert empty multivector")
                arr.tofile(fv)

                offset_row = np.array([[cur_token_offset, n_tokens]], dtype=np.int64)
                offset_row.tofile(fo)
                new_offsets_rows.append((cur_token_offset, n_tokens))
                cur_token_offset += n_tokens

                payload = segment.model_dump()
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

                # Soft-delete any previous row carrying the same segment_id.
                prev_idx = self._id_to_idx.get(segment.segment_id)
                row_idx = len(self._payloads)
                if prev_idx is not None:
                    self._deleted.add(prev_idx)
                self._id_to_idx[segment.segment_id] = row_idx
                self._payloads.append(payload)
        # Append to the in-memory offsets array in one shot.
        self._offsets = np.concatenate(
            [self._offsets, np.asarray(new_offsets_rows, dtype=np.int64)],
            axis=0,
        )
        self._vectors_mmap = None  # invalidate cache; next search remaps.
        self._save_manifest()
        self._save_id_index()
        if prev_idx is not None or self._deleted:
            self._save_deleted()

    def delete_by_ids(self, point_ids: list[str]) -> None:
        """Soft-delete points by segment id (the row stays in vectors.bin)."""
        changed = False
        for pid in point_ids:
            idx = self._id_to_idx.pop(pid, None)
            if idx is not None:
                self._deleted.add(idx)
                changed = True
        if changed:
            self._save_deleted()
            self._save_id_index()

    def search(
        self,
        query_vector: MultiVector,
        top_k: int = 10,
        path_filter: list[str] | None = None,
    ) -> list[ScoredPoint]:
        """Brute-force MaxSim retrieval over all (non-deleted, filtered) points."""
        if not self._payloads:
            return []
        query_mv = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32))
        if query_mv.ndim != 2 or query_mv.shape[1] != self.embedding_dim:
            raise ValueError(
                f"query multivector shape {query_mv.shape} does not match dim {self.embedding_dim}"
            )

        all_vecs = self._get_vectors_view()
        if all_vecs.shape[0] == 0:
            return []

        n_rows = len(self._payloads)
        active = np.ones(n_rows, dtype=bool)
        if self._deleted:
            for idx in self._deleted:
                if 0 <= idx < n_rows:
                    active[idx] = False
        if path_filter is not None:
            path_set = set(path_filter)
            for i, payload in enumerate(self._payloads):
                if active[i] and payload.get("path") not in path_set:
                    active[i] = False

        # ColBERT MaxSim:
        #   sim       (n_query, T_total)   = query_mv @ all_vecs.T
        #   per_doc[q, i] = max(sim[q, t] for t in doc_i)
        #   doc_score[i]  = sum_q per_doc[q, i]
        sim = query_mv @ all_vecs.T  # float32 BLAS matmul
        seg_starts = self._offsets[:, 0].astype(np.int64)
        per_doc_max = np.maximum.reduceat(sim, seg_starts, axis=1)
        doc_scores = per_doc_max.sum(axis=0)
        doc_scores = np.where(active, doc_scores, -np.inf)

        n_active = int(active.sum())
        if n_active == 0:
            return []
        k = min(top_k, n_active)
        top_idx = np.argpartition(-doc_scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-doc_scores[top_idx])]

        results: list[ScoredPoint] = []
        for i in top_idx:
            payload = self._payloads[int(i)]
            results.append(
                ScoredPoint(
                    id=str(payload.get("segment_id", "")),
                    score=float(doc_scores[int(i)]),
                    payload=payload,
                )
            )
        return results

    def get_collection_info(self) -> dict:
        return {
            "points_count": len(self._payloads) - len(self._deleted),
            "collection_name": self.collection_name,
        }
