"""Tests for src/common/preprocessing.py."""

import pytest


class TestExtractText:
    """Tests for extract_text() dispatcher."""

    def test_extract_txt_file(self, tmp_path):
        """Plain text file should be read and cleaned."""
        raise NotImplementedError

    def test_extract_pdf_file(self, tmp_path):
        """PDF extraction should return string content."""
        raise NotImplementedError

    def test_unsupported_format_raises(self, tmp_path):
        """Unsupported file extension should raise ValueError."""
        raise NotImplementedError


class TestCleanText:
    """Tests for clean_text()."""

    def test_removes_extra_whitespace(self):
        raise NotImplementedError

    def test_preserves_paragraph_breaks(self):
        raise NotImplementedError
