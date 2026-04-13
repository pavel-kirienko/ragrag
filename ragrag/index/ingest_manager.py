from __future__ import annotations

import importlib
import logging
import time
from typing import Callable, cast
from PIL import Image

logger = logging.getLogger(__name__)

from ragrag.config import Settings
from ragrag.embedding.colqwen_embedder import ColQwenEmbedder
from ragrag.extractors.text_extractor import extract_text_segments
from ragrag.file_state import FileStateTracker
from ragrag.index.qdrant_store import QdrantStore
from ragrag.models import FileType, IndexingStats, Modality, Segment, SkippedFile, get_file_type
from ragrag.path_discovery import discover_files


class IngestManager:
    def __init__(self, embedder: ColQwenEmbedder, store: QdrantStore, settings: Settings):
        self.embedder: ColQwenEmbedder = embedder
        self.store: QdrantStore = store
        self.settings: Settings = settings
        self.file_tracker: FileStateTracker = FileStateTracker(settings.index_path)

    def ingest_paths(self, paths: list[str]) -> tuple[IndexingStats, list[SkippedFile], list[str]]:
        stats = IndexingStats()
        per_file_skipped: list[SkippedFile] = []

        file_paths, discovery_skipped = discover_files(paths, self.settings)

        t_start = time.time()
        for idx, file_path in enumerate(file_paths):
            if time.time() - t_start > self.settings.indexing_timeout:
                per_file_skipped.extend(
                    SkippedFile(path=fp, reason="indexing timeout")
                    for fp in file_paths[idx:]
                )
                break
            try:
                existing_point_ids = self.file_tracker.get_point_ids(file_path)
                needs_reindex, current_state = self.file_tracker.check_staleness(file_path)

                if not needs_reindex:
                    stats.files_skipped_unchanged += 1
                    continue

                logger.info("Indexing %s (%d/%d)", file_path, idx + 1, len(file_paths))
                was_previously_indexed = len(existing_point_ids) > 0
                if was_previously_indexed:
                    self.store.delete_by_ids(existing_point_ids)

                file_type = get_file_type(file_path)
                if file_type is None:
                    raise ValueError("unsupported file type")

                segments, images = self._extract_segments(file_path, file_type)
                vectors = self._embed_segments(segments, images)
                self.store.upsert_many(list(zip(segments, vectors)))

                self.file_tracker.mark_indexed(
                    file_path,
                    [segment.segment_id for segment in segments],
                    file_state=current_state,
                )

                if was_previously_indexed:
                    stats.files_updated += 1
                else:
                    stats.files_added += 1

            except Exception as exc:
                per_file_skipped.append(
                    SkippedFile(path=file_path, reason=f"ingest error: {exc}")
                )

        if stats.files_added == 0 and stats.files_updated == 0:
            logger.debug("All %d files unchanged, searching existing index", len(file_paths))
            logger.debug(
                "Index up to date: %d files unchanged, %d added, %d updated",
                stats.files_skipped_unchanged, stats.files_added, stats.files_updated,
            )
        else:
            logger.info(
                "Index up to date: %d files unchanged, %d added, %d updated",
                stats.files_skipped_unchanged, stats.files_added, stats.files_updated,
            )
        return stats, discovery_skipped + per_file_skipped, file_paths

    def _embed_segments(
        self,
        segments: list[Segment],
        images: list[Image.Image],
    ) -> list:
        """Embed a file's segments, batching text and keeping images single.

        Text is batched by ``settings.text_batch_size``. On any batch failure we
        fall back to per-item embedding so a single bad chunk doesn't poison an
        otherwise-valid file; per-item failures bubble up to the caller and fail
        the file as they did before.

        Images are embedded one at a time on this 8 GB GPU tier — batching
        images risks OOM during indexing for little gain (see plan rationale).
        The ``image_index`` invariant is preserved: IMAGE segments consume the
        ``images`` list in emitted order.
        """
        vectors: list = [None] * len(segments)

        text_indices = [i for i, s in enumerate(segments) if s.modality == Modality.TEXT]
        batch_size = max(1, int(self.settings.text_batch_size))
        for start in range(0, len(text_indices), batch_size):
            group = text_indices[start : start + batch_size]
            group_texts = [segments[i].excerpt for i in group]
            try:
                group_vecs = self.embedder.embed_text_chunks(group_texts)
            except Exception as exc:
                logger.warning(
                    "Batched text embed of %d chunks failed (%s); retrying singles",
                    len(group_texts), exc,
                )
                group_vecs = [self.embedder.embed_text_chunk(t) for t in group_texts]
            for idx, vec in zip(group, group_vecs):
                vectors[idx] = vec

        image_index = 0
        for i, segment in enumerate(segments):
            if segment.modality == Modality.TEXT:
                continue
            if segment.modality == Modality.IMAGE:
                if image_index >= len(images):
                    raise ValueError("missing image for image-modality segment")
                vectors[i] = self.embedder.embed_image(images[image_index])
                image_index += 1
            else:
                raise ValueError(f"unsupported modality: {segment.modality}")

        if image_index != len(images):
            raise ValueError("image list does not match image-modality segments")

        for i, v in enumerate(vectors):
            if v is None:
                raise ValueError(f"embedding missing for segment at index {i}")
        return vectors

    def _extract_segments(
        self, file_path: str, file_type: FileType
    ) -> tuple[list[Segment], list[Image.Image]]:
        if file_type == FileType.TEXT:
            return extract_text_segments(file_path, self.settings), []
        if file_type == FileType.PDF:
            pdf_module = importlib.import_module("ragrag.extractors.pdf_extractor")
            extract_pdf = cast(
                Callable[[str, Settings], tuple[list[Segment], list[Image.Image]]],
                getattr(pdf_module, "extract_pdf_segments"),
            )
            return extract_pdf(file_path, self.settings)
        if file_type == FileType.IMAGE:
            image_module = importlib.import_module("ragrag.extractors.image_extractor")
            extract_image = cast(
                Callable[[str, Settings], tuple[list[Segment], list[Image.Image]]],
                getattr(image_module, "extract_image_segments"),
            )
            return extract_image(file_path, self.settings)
