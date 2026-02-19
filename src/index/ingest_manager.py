from __future__ import annotations

import importlib
import time
from typing import Callable, cast
from PIL import Image

from src.config import Settings
from src.embedding.colqwen_embedder import ColQwenEmbedder
from src.extractors.text_extractor import extract_text_segments
from src.file_state import FileStateTracker
from src.index.qdrant_store import QdrantStore
from src.models import FileType, IndexingStats, Modality, Segment, SkippedFile, get_file_type
from src.path_discovery import discover_files


class IngestManager:
    def __init__(self, embedder: ColQwenEmbedder, store: QdrantStore, settings: Settings):
        self.embedder: ColQwenEmbedder = embedder
        self.store: QdrantStore = store
        self.settings: Settings = settings
        self.file_tracker: FileStateTracker = FileStateTracker()

    def ingest_paths(self, paths: list[str]) -> tuple[IndexingStats, list[SkippedFile]]:
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
                needs_reindex, _ = self.file_tracker.check_staleness(file_path)

                if not needs_reindex:
                    stats.files_skipped_unchanged += 1
                    continue

                was_previously_indexed = len(existing_point_ids) > 0
                if was_previously_indexed:
                    self.store.delete_by_ids(existing_point_ids)

                file_type = get_file_type(file_path)
                if file_type is None:
                    raise ValueError("unsupported file type")

                segments, images = self._extract_segments(file_path, file_type)

                image_index = 0
                for segment in segments:
                    if segment.modality == Modality.TEXT:
                        vector = self.embedder.embed_text_chunk(segment.excerpt)
                    elif segment.modality == Modality.IMAGE:
                        if image_index >= len(images):
                            raise ValueError("missing image for image-modality segment")
                        vector = self.embedder.embed_image(images[image_index])
                        image_index += 1
                    else:
                        raise ValueError(f"unsupported modality: {segment.modality}")

                    self.store.upsert(segment, vector)

                if image_index != len(images):
                    raise ValueError("image list does not match image-modality segments")

                self.file_tracker.mark_indexed(
                    file_path, [segment.segment_id for segment in segments]
                )

                if was_previously_indexed:
                    stats.files_updated += 1
                else:
                    stats.files_added += 1

            except Exception as exc:
                per_file_skipped.append(
                    SkippedFile(path=file_path, reason=f"ingest error: {exc}")
                )

        return stats, discovery_skipped + per_file_skipped

    def _extract_segments(
        self, file_path: str, file_type: FileType
    ) -> tuple[list[Segment], list[Image.Image]]:
        if file_type == FileType.TEXT:
            return extract_text_segments(file_path, self.settings), []
        if file_type == FileType.PDF:
            pdf_module = importlib.import_module("src.extractors.pdf_extractor")
            extract_pdf = cast(
                Callable[[str, Settings], tuple[list[Segment], list[Image.Image]]],
                getattr(pdf_module, "extract_pdf_segments"),
            )
            return extract_pdf(file_path, self.settings)
        if file_type == FileType.IMAGE:
            image_module = importlib.import_module("src.extractors.image_extractor")
            extract_image = cast(
                Callable[[str, Settings], tuple[list[Segment], list[Image.Image]]],
                getattr(image_module, "extract_image_segments"),
            )
            return extract_image(file_path, self.settings)
