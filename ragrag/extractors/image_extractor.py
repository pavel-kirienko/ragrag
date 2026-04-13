"""Image extraction for ragrag.

Loads image files and creates visual segments with OCR metadata.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from PIL import Image

from ragrag.config import Settings
from ragrag.models import FileType, Modality, Segment
from ragrag.extractors.ocr import ocr_image

logger = logging.getLogger(__name__)


def extract_image_segments(path: str, settings: Settings) -> tuple[list[Segment], list[Image.Image]]:
    """Load an image file and create visual segment + OCR metadata.
    
    Args:
        path: Absolute path to the image file.
        settings: Settings object (for future extensibility).
    
    Returns:
        Tuple of (segments, images):
        - segments: List with exactly 1 Segment on success, empty list on error.
        - images: List with exactly 1 PIL Image on success, empty list on error.
    """
    try:
        # Load image and convert to RGB
        image = Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return ([], [])
    
    # Run OCR for metadata
    ocr_text = ocr_image(image)
    
    # Create excerpt: use OCR text if non-empty, else filename
    if ocr_text:
        excerpt = ocr_text
    else:
        excerpt = f"Image file: {os.path.basename(path)}"
    
    # Create segment
    segment = Segment(
        segment_id=str(uuid.uuid4()),
        path=str(Path(path).resolve()),
        file_type=FileType.IMAGE,
        modality=Modality.IMAGE,
        excerpt=excerpt,
    )
    
    return ([segment], [image])
