"""Image extraction for ragrag.

Loads image files and creates visual segments with OCR metadata.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

from ragrag.config import Settings
from ragrag.models import FileType, Modality, Segment
from ragrag.extractors.ocr import ocr_image

logger = logging.getLogger(__name__)


def iter_image_segments(
    path: str, settings: Settings
) -> Iterator[tuple[Segment, Optional[Image.Image]]]:
    """Yield a single ``(segment, image)`` pair for an image file."""
    try:
        image = Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return

    ocr_text = ocr_image(image)
    excerpt = ocr_text if ocr_text else f"Image file: {os.path.basename(path)}"

    yield (
        Segment(
            segment_id=str(uuid.uuid4()),
            path=str(Path(path).resolve()),
            file_type=FileType.IMAGE,
            modality=Modality.IMAGE,
            excerpt=excerpt,
        ),
        image,
    )


def extract_image_segments(path: str, settings: Settings) -> tuple[list[Segment], list[Image.Image]]:
    """Eager wrapper around :func:`iter_image_segments` for legacy callers."""
    segments: list[Segment] = []
    images: list[Image.Image] = []
    for segment, image in iter_image_segments(path, settings):
        segments.append(segment)
        if image is not None:
            images.append(image)
    return segments, images
