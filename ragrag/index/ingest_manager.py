from __future__ import annotations

import importlib
import logging
import time
from typing import Callable, Iterator, Optional, cast
from PIL import Image

logger = logging.getLogger(__name__)

from ragrag.config import Settings
from ragrag.embedding.colqwen_embedder import ColQwenEmbedder
from ragrag.extractors.text_extractor import iter_text_segments
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

                segment_ids = self._stream_embed_and_store(file_path, file_type)

                self.file_tracker.mark_indexed(
                    file_path,
                    segment_ids,
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

    def _stream_embed_and_store(self, file_path: str, file_type: FileType) -> list[str]:
        """Stream segments through embed → upsert without buffering the whole file.

        Text segments are accumulated up to ``settings.text_batch_size`` then
        flushed in one batched forward pass. Image segments flush any pending
        text first (to preserve emission order), then run a single-item image
        embed and upsert immediately. The peak memory footprint is roughly:

            one PIL page image + ``text_batch_size`` text chunks + one
            image multivector

        which is independent of the source document length. On any text-batch
        failure we fall back to per-item embedding so one bad chunk doesn't
        poison an otherwise-valid file.
        """
        text_batch_size = max(1, int(self.settings.text_batch_size))
        pending_text: list[Segment] = []
        all_segment_ids: list[str] = []

        def flush_text() -> None:
            if not pending_text:
                return
            try:
                vecs = self.embedder.embed_text_chunks([s.excerpt for s in pending_text])
            except Exception as exc:
                logger.warning(
                    "Batched text embed of %d chunks failed (%s); retrying singles",
                    len(pending_text), exc,
                )
                vecs = [self.embedder.embed_text_chunk(s.excerpt) for s in pending_text]
            self.store.upsert_many(list(zip(pending_text, vecs)))
            all_segment_ids.extend(s.segment_id for s in pending_text)
            pending_text.clear()

        for segment, image in self._iter_segments(file_path, file_type):
            if segment.modality == Modality.TEXT:
                pending_text.append(segment)
                if len(pending_text) >= text_batch_size:
                    flush_text()
            elif segment.modality == Modality.IMAGE:
                # Preserve the emission order: any buffered text from the prior
                # page must land before this image's page does.
                flush_text()
                if image is None:
                    raise ValueError("missing image for image-modality segment")
                vec = self.embedder.embed_image(image)
                self.store.upsert_many([(segment, vec)])
                all_segment_ids.append(segment.segment_id)
                # Drop refs so the next iteration can reclaim the memory.
                del image, vec
            else:
                raise ValueError(f"unsupported modality: {segment.modality}")

        flush_text()
        return all_segment_ids

    def _extract_segments(
        self, file_path: str, file_type: FileType
    ) -> tuple[list[Segment], list[Image.Image]]:
        """Materialize all segments + images for a file.

        Eager wrapper retained for tests; production ingest goes through
        :meth:`_iter_segments` / :meth:`_stream_embed_and_store`.
        """
        segments: list[Segment] = []
        images: list[Image.Image] = []
        for segment, image in self._iter_segments(file_path, file_type):
            segments.append(segment)
            if image is not None:
                images.append(image)
        return segments, images

    def _iter_segments(
        self, file_path: str, file_type: FileType
    ) -> Iterator[tuple[Segment, Optional[Image.Image]]]:
        if file_type == FileType.TEXT:
            return iter_text_segments(file_path, self.settings)
        if file_type == FileType.PDF:
            pdf_module = importlib.import_module("ragrag.extractors.pdf_extractor")
            iter_pdf = cast(
                Callable[[str, Settings], Iterator[tuple[Segment, Optional[Image.Image]]]],
                getattr(pdf_module, "iter_pdf_segments"),
            )
            return iter_pdf(file_path, self.settings)
        if file_type == FileType.IMAGE:
            image_module = importlib.import_module("ragrag.extractors.image_extractor")
            iter_image = cast(
                Callable[[str, Settings], Iterator[tuple[Segment, Optional[Image.Image]]]],
                getattr(image_module, "iter_image_segments"),
            )
            return iter_image(file_path, self.settings)
        raise ValueError(f"unsupported file type: {file_type}")
