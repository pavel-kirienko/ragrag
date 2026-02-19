"""PDF extraction for ragrag.

Extracts text chunks and page images from PDF files using PyMuPDF.
For each page:
  - Renders full page as a PIL Image (for visual embedding)
  - Extracts native text; falls back to OCR if text is sparse
  - Produces both IMAGE and TEXT modality Segments
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import pymupdf  # PyMuPDF 1.24+
    _mupdf = pymupdf
except ImportError:
    import fitz as _mupdf  # type: ignore[no-reattr]

from PIL import Image

from src.config import Settings
from src.extractors.ocr import ocr_image
from src.models import FileType, Modality, Segment

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def extract_pdf_segments(
    path: str,
    settings: Settings,
) -> tuple[list[Segment], list[Image.Image]]:
    """Extract text chunks and page images from a PDF.

    Returns:
        (segments, page_images) where page_images[i] corresponds to page i.
        Both lists are empty on error (corrupt/encrypted PDF).
    """
    resolved = str(Path(path).resolve())

    # --- Open document ---
    try:
        doc = _mupdf.open(path)
    except Exception as e:
        logger.warning(f"Failed to open PDF {path}: {e}")
        return [], []

    # --- Encrypted guard ---
    if doc.is_encrypted:
        logger.warning(f"PDF is encrypted, skipping: {path}")
        doc.close()
        return [], []

    segments: list[Segment] = []
    page_images: list[Image.Image] = []

    for page_index in range(len(doc)):
        page_number = page_index + 1  # 1-indexed

        try:
            page = doc[page_index]

            # 1. Extract native text
            native_text: str = str(page.get_text(sort=True))  # type: ignore[union-attr]

            # 2. Render page to PIL Image
            pix = page.get_pixmap(dpi=settings.pdf_dpi)
            pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)  # type: ignore[arg-type]
            del pix  # free memory immediately

            page_images.append(pil_image)

            # 3. OCR fallback if native text is sparse
            if len(native_text.strip()) < settings.ocr_threshold:
                ocr_text = ocr_image(pil_image)
                text_for_segment = ocr_text
            else:
                text_for_segment = native_text

            # 4. Always create an IMAGE-modality segment for visual embedding
            image_excerpt = text_for_segment.strip() or f"Visual match on page {page_number}."
            segments.append(
                Segment(
                    segment_id=str(uuid.uuid4()),
                    path=resolved,
                    file_type=FileType.PDF,
                    modality=Modality.IMAGE,
                    page=page_number,
                    excerpt=image_excerpt[:500],  # keep excerpt manageable
                )
            )

            # 5. If text is available, chunk it into TEXT segments
            if text_for_segment.strip():
                text_chunks = _chunk_text(text_for_segment, settings.chunk_size)
                for chunk in text_chunks:
                    segments.append(
                        Segment(
                            segment_id=str(uuid.uuid4()),
                            path=resolved,
                            file_type=FileType.PDF,
                            modality=Modality.TEXT,
                            page=page_number,
                            excerpt=chunk,
                        )
                    )

        except RuntimeError as e:
            logger.warning(f"Error processing page {page_number} of {path}: {e}")
            continue

    doc.close()
    return segments, page_images


def _chunk_text(text: str, target_chars: int) -> list[str]:
    """Split text into chunks of ~target_chars at word boundaries.

    Args:
        text: Text to split.
        target_chars: Target chunk size in characters.

    Returns:
        List of non-empty text chunks.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + target_chars, length)

        if end < length:
            # Walk back to a word boundary (space or newline)
            boundary = end
            while boundary > start and text[boundary] not in (' ', '\n', '\t'):
                boundary -= 1
            if boundary == start:
                # No word boundary found; hard-cut at target
                boundary = end
            end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end

    return chunks
