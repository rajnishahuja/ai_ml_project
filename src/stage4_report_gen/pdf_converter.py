"""
Stage 4 DOCX → PDF converter.

Uses `docx2pdf`, which delegates to LibreOffice (headless on Linux/macOS)
or Microsoft Word (Windows). The conversion is best-effort: the Stage 4
pipeline must NOT crash if LibreOffice/Word isn't installed — the DOCX is
still a valid deliverable on its own. PDF failure logs a warning and
returns None; callers handle that case.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def convert_to_pdf(docx_path: Path | str) -> Optional[Path]:
    """Convert a .docx file to .pdf alongside it.

    Args:
        docx_path: Existing .docx file path.

    Returns:
        Path to the .pdf on success. None on failure (missing converter,
        binary error, etc.). The Stage 4 pipeline must tolerate None.
    """
    src = Path(docx_path)
    if not src.exists():
        logger.warning("convert_to_pdf: source %s does not exist.", src)
        return None

    pdf_path = src.with_suffix(".pdf")

    try:
        from docx2pdf import convert
    except ImportError:
        logger.warning(
            "convert_to_pdf: docx2pdf not installed — install with "
            "`pip install docx2pdf`. Returning None; DOCX still available."
        )
        return None

    try:
        convert(str(src), str(pdf_path))
    except Exception as e:                                            # noqa: BLE001
        # docx2pdf raises a variety of subprocess / OSError when LibreOffice
        # isn't on PATH. Treat all of them the same — log and return None.
        logger.warning(
            "convert_to_pdf: PDF conversion failed (%s). DOCX still available.",
            e,
        )
        return None

    if not pdf_path.exists():
        logger.warning("convert_to_pdf: %s not produced. Returning None.", pdf_path)
        return None

    logger.info("Wrote PDF report to %s", pdf_path)
    return pdf_path
