"""
Contract text extraction from PDF, DOCX, and plain text files.

Provides a single entry point: extract_text(file_path) → str.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(file_path: str) -> str:
    """Extract plain text from a contract file (PDF, DOCX, or TXT).

    Args:
        file_path: Path to the contract file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext in (".docx", ".doc"):
        return _extract_docx(path)
    elif ext == ".txt":
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use PDF, DOCX, or TXT.")


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF file using pdfplumber.

    Args:
        path: Path to PDF file.

    Returns:
        Concatenated text from all pages.
    """
    raise NotImplementedError("PDF extraction not yet implemented")


def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx.

    Args:
        path: Path to DOCX file.

    Returns:
        Concatenated paragraph text.
    """
    raise NotImplementedError("DOCX extraction not yet implemented")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip common artifacts from extracted text.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    raise NotImplementedError("Text cleaning not yet implemented")
