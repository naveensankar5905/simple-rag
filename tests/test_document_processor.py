"""
test_document_processor.py
──────────────────────────
Unit and functional tests for the DocumentProcessor module.
Tests cover: initialisation, PDF extraction, TXT extraction,
text cleaning, and sliding-window chunking logic.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open


# ══════════════════════════════════════════════════════════════════════════════
#  INITIALISATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentProcessorInit:
    """Tests for DocumentProcessor.__init__."""

    def test_init_defaults(self, document_processor):
        """Default chunk_size=500, chunk_overlap=50."""
        assert document_processor.chunk_size == 500
        assert document_processor.chunk_overlap == 50

    def test_init_custom_params(self):
        """Custom chunk_size and chunk_overlap are stored correctly."""
        from document_processor import DocumentProcessor
        dp = DocumentProcessor(chunk_size=200, chunk_overlap=30)
        assert dp.chunk_size == 200
        assert dp.chunk_overlap == 30


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT EXTRACTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestTextExtraction:
    """Tests for _extract_txt, _extract_pdf and process()."""

    def test_process_txt_file(self, document_processor, tmp_txt_file):
        """Processing a .txt file returns chunks and full text."""
        chunks, full_text = document_processor.process(str(tmp_txt_file))
        assert isinstance(chunks, list)
        assert isinstance(full_text, str)
        assert len(full_text) > 0
        assert len(chunks) >= 1

    def test_extract_txt_content(self, document_processor, tmp_txt_file, sample_short_text):
        """_extract_txt reads the file content correctly."""
        result = document_processor._extract_txt(tmp_txt_file)
        assert result == sample_short_text

    def test_process_unsupported_file_type(self, document_processor, tmp_unsupported_file):
        """Unsupported file types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            document_processor.process(str(tmp_unsupported_file))

    def test_process_pdf_with_pdfplumber(self, document_processor, tmp_path):
        """PDF extraction via pdfplumber (mocked)."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        # Mock pdfplumber
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page one content here."
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            chunks, text = document_processor.process(str(pdf_file))

        assert "Page one content here" in text
        assert len(chunks) >= 1

    def test_process_pdf_multiple_pages(self, document_processor, tmp_path):
        """PDF with multiple pages joins them with double newlines."""
        pdf_file = tmp_path / "multi.pdf"
        pdf_file.write_bytes(b"fake")

        page1 = MagicMock()
        page1.extract_text.return_value = "First page."
        page2 = MagicMock()
        page2.extract_text.return_value = "Second page."
        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            chunks, text = document_processor.process(str(pdf_file))

        assert "First page" in text
        assert "Second page" in text

    def test_process_pdf_page_returns_none(self, document_processor, tmp_path):
        """Pages returning None text are skipped."""
        pdf_file = tmp_path / "empty_page.pdf"
        pdf_file.write_bytes(b"fake")

        page1 = MagicMock()
        page1.extract_text.return_value = None  # Empty page
        page2 = MagicMock()
        page2.extract_text.return_value = "Only real content."
        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            chunks, text = document_processor.process(str(pdf_file))

        assert "Only real content" in text

    def test_process_pdf_fallback_to_pypdf2(self, document_processor, tmp_path):
        """When pdfplumber import fails, falls back to PyPDF2."""
        pdf_file = tmp_path / "fallback.pdf"
        pdf_file.write_bytes(b"fake pdf")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PyPDF2 extracted text here."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        # Directly test _extract_pdf by simulating pdfplumber ImportError
        # We patch pdfplumber.open to raise ImportError, and PyPDF2 to return mock
        with patch("pdfplumber.open", side_effect=ImportError("no pdfplumber")):
            with patch("PyPDF2.PdfReader", return_value=mock_reader):
                from document_processor import DocumentProcessor
                dp = DocumentProcessor()
                text = dp._extract_pdf(pdf_file)

        assert "PyPDF2 extracted text" in text


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT CLEANING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestTextCleaning:
    """Tests for DocumentProcessor._clean."""

    def test_clean_removes_carriage_returns(self, document_processor):
        """\\r\\n and \\r are normalised to \\n."""
        text = "Hello\r\nWorld\rFoo"
        result = document_processor._clean(text)
        assert "\r" not in result
        assert "Hello\nWorld\nFoo" == result

    def test_clean_collapses_inline_whitespace(self, document_processor):
        """Multiple spaces/tabs collapse to a single space."""
        text = "Hello     World\t\tFoo"
        result = document_processor._clean(text)
        assert "Hello World Foo" == result

    def test_clean_limits_consecutive_newlines(self, document_processor):
        """Three or more newlines are reduced to exactly two."""
        text = "Hello\n\n\n\n\nWorld"
        result = document_processor._clean(text)
        assert result == "Hello\n\nWorld"

    def test_clean_strips_leading_trailing(self, document_processor):
        """Leading and trailing whitespace is stripped."""
        text = "   Hello World   "
        result = document_processor._clean(text)
        assert result == "Hello World"

    def test_clean_combined(self, document_processor):
        """All cleaning rules applied together."""
        text = "  \r\n  Hello   \r\n\n\n\n  World  \r  "
        result = document_processor._clean(text)
        assert "\r" not in result
        assert "   " not in result  # No triple spaces
        assert result == result.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  CHUNKING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestChunking:
    """Tests for DocumentProcessor._chunk."""

    def test_chunk_empty_text(self, document_processor):
        """Empty text returns empty list."""
        assert document_processor._chunk("") == []

    def test_chunk_single_chunk(self, document_processor, sample_short_text):
        """Text shorter than chunk_size produces exactly 1 chunk."""
        chunks = document_processor._chunk(sample_short_text)
        assert len(chunks) == 1

    def test_chunk_multiple_chunks(self, document_processor_small):
        """Text longer than chunk_size produces multiple chunks."""
        text = " ".join([f"word{i}" for i in range(25)])
        chunks = document_processor_small._chunk(text)
        assert len(chunks) > 1

    def test_chunk_overlap_present(self, document_processor_small):
        """Overlapping words appear in consecutive chunks."""
        text = " ".join([f"word{i}" for i in range(25)])
        chunks = document_processor_small._chunk(text)
        if len(chunks) >= 2:
            # Last 2 words of chunk 0 should appear at start of chunk 1
            words_chunk0 = chunks[0].split()
            words_chunk1 = chunks[1].split()
            overlap_words = words_chunk0[-document_processor_small.chunk_overlap:]
            assert overlap_words == words_chunk1[:document_processor_small.chunk_overlap]

    def test_chunk_exact_boundary(self):
        """Text with exactly chunk_size words → 1 chunk."""
        from document_processor import DocumentProcessor
        dp = DocumentProcessor(chunk_size=10, chunk_overlap=2)
        text = " ".join([f"w{i}" for i in range(10)])
        chunks = dp._chunk(text)
        assert len(chunks) == 1

    def test_chunk_no_empty_chunks(self, document_processor_small):
        """No chunk should be empty or whitespace-only."""
        text = " ".join([f"word{i}" for i in range(30)])
        chunks = document_processor_small._chunk(text)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_chunk_all_words_covered(self, document_processor_small):
        """All words from the original text appear in at least one chunk."""
        words = [f"word{i}" for i in range(25)]
        text = " ".join(words)
        chunks = document_processor_small._chunk(text)
        all_chunked_words = set()
        for chunk in chunks:
            all_chunked_words.update(chunk.split())
        for word in words:
            assert word in all_chunked_words


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTIONAL / INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentProcessorIntegration:
    """End-to-end tests for the full process() pipeline."""

    def test_full_pipeline_txt(self, document_processor, tmp_txt_file):
        """Full process pipeline: txt → extract → clean → chunk."""
        chunks, text = document_processor.process(str(tmp_txt_file))
        assert isinstance(chunks, list)
        assert isinstance(text, str)
        assert len(text) > 0
        assert all(isinstance(c, str) for c in chunks)
        assert all(len(c) > 0 for c in chunks)

    def test_full_pipeline_returns_cleaned_text(self, document_processor, tmp_path):
        """Returned text is cleaned (no excessive whitespace)."""
        f = tmp_path / "dirty.txt"
        f.write_text("  Hello   \r\n\n\n\n  World  ", encoding="utf-8")
        chunks, text = document_processor.process(str(f))
        assert "\r" not in text
        assert "   " not in text

    def test_process_long_text_multiple_chunks(self, tmp_path, sample_long_text):
        """Long text produces multiple chunks with default settings."""
        from document_processor import DocumentProcessor
        dp = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        f = tmp_path / "long.txt"
        f.write_text(sample_long_text, encoding="utf-8")
        chunks, text = dp.process(str(f))
        assert len(chunks) > 1
