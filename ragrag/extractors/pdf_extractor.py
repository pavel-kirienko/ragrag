"""PDF extraction for ragrag.

Extracts text chunks and page images from PDF files using PyMuPDF.
Pages are streamed one at a time so the consumer can embed and upsert
without buffering the whole document — peak memory is one page image
plus a small text-chunk batch instead of O(file_size).
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Iterator, Optional

try:
    import pymupdf  # PyMuPDF 1.24+
    _mupdf = pymupdf
except ImportError:
    import fitz as _mupdf  # type: ignore[no-reattr]

from PIL import Image

from ragrag.config import Settings
from ragrag.extractors.ocr import ocr_image
from ragrag.extractors.text_extractor import _chunk_text
from ragrag.models import FileType, Modality, Segment

logger = logging.getLogger(__name__)


def iter_pdf_segments(
    path: str,
    settings: Settings,
) -> Iterator[tuple[Segment, Optional[Image.Image]]]:
    """Lazily yield ``(segment, image_or_None)`` tuples for a PDF.

    For each page we yield the IMAGE segment (with its rendered PIL Image)
    first, then the per-chunk TEXT segments (image=None). The PIL Image is
    only referenced for the duration of the yield — once the consumer
    finishes embedding and discards it, the page is freed before the next
    page is rendered. This keeps RSS bounded for large datasheets.
    """
    resolved = str(Path(path).resolve())

    try:
        doc = _mupdf.open(path)
    except Exception as e:
        logger.warning(f"Failed to open PDF {path}: {e}")
        return

    if doc.is_encrypted:
        logger.warning(f"PDF is encrypted, skipping: {path}")
        doc.close()
        return

    try:
        total_pages = len(doc)
        for page_index in range(total_pages):
            page_number = page_index + 1
            try:
                logger.info("  Page %d/%d of %s", page_number, total_pages, path)
                page = doc[page_index]

                native_text: str = str(page.get_text(sort=True))  # type: ignore[union-attr]

                pix = page.get_pixmap(dpi=settings.pdf_dpi)
                pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)  # type: ignore[arg-type]
                del pix

                if len(native_text.strip()) < settings.ocr_threshold:
                    ocr_text = ocr_image(pil_image)
                    text_for_segment = ocr_text
                else:
                    text_for_segment = native_text

                image_excerpt = text_for_segment.strip() or f"Visual match on page {page_number}."
                yield (
                    Segment(
                        segment_id=str(uuid.uuid4()),
                        path=resolved,
                        file_type=FileType.PDF,
                        modality=Modality.IMAGE,
                        page=page_number,
                        excerpt=image_excerpt[:500],
                    ),
                    pil_image,
                )
                # Drop our local reference so the only live ref is whatever
                # the consumer kept (typically none after embed).
                del pil_image

                if text_for_segment.strip():
                    for chunk_text, _start_line, _end_line in _chunk_text(
                        text_for_segment,
                        target_chars=settings.chunk_size,
                        overlap_chars=settings.chunk_overlap,
                    ):
                        yield (
                            Segment(
                                segment_id=str(uuid.uuid4()),
                                path=resolved,
                                file_type=FileType.PDF,
                                modality=Modality.TEXT,
                                page=page_number,
                                excerpt=chunk_text,
                            ),
                            None,
                        )

            except RuntimeError as e:
                logger.warning(f"Error processing page {page_number} of {path}: {e}")
                continue
    finally:
        doc.close()


def extract_pdf_segments(
    path: str,
    settings: Settings,
) -> tuple[list[Segment], list[Image.Image]]:
    """Eager wrapper around :func:`iter_pdf_segments`.

    Materializes every segment and every page image — used by the legacy
    test path. Production ingest should iterate lazily via
    :func:`iter_pdf_segments` to avoid holding the full document in RAM.
    """
    segments: list[Segment] = []
    images: list[Image.Image] = []
    for segment, image in iter_pdf_segments(path, settings):
        segments.append(segment)
        if image is not None:
            images.append(image)
    return segments, images
