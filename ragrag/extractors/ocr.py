"""Tesseract OCR wrapper for image text extraction."""

import logging
from typing import Optional

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def is_tesseract_available() -> bool:
    """Check if tesseract binary is accessible.
    
    Returns:
        True if Tesseract is installed and accessible, False otherwise.
    """
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        logger.warning(f"Tesseract not available: {e}")
        return False


def ocr_image(image: Image.Image, lang: str = "eng") -> str:
    """Run Tesseract OCR on a PIL Image, return extracted text.
    
    Args:
        image: PIL Image object to extract text from.
        lang: Tesseract language code (default: "eng" for English).
    
    Returns:
        Extracted text from the image, stripped of excessive whitespace.
        Returns empty string if OCR fails or Tesseract is not available.
    """
    try:
        # Convert to grayscale for better OCR performance
        if image.mode != "L":
            image = image.convert("L")
        
        # Run Tesseract OCR
        text = pytesseract.image_to_string(image, lang=lang)
        
        # Strip excessive whitespace
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        return ""
