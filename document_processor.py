"""
document_processor.py
─────────────────────
Legacy PDF / TXT processor — kept for the existing test suite.

For the full multimodal pipeline (PDF, TXT, Image, Audio, Video) use
MultimodalProcessor from multimodal_processor.py instead.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Tuple


class DocumentProcessor:
    """Extract text from PDF / TXT files and split into overlapping chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, file_path: str) -> Tuple[list[str], str]:
        """Return (chunks, full_text) for the given file."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            text = self._extract_pdf(path)
        elif suffix == ".txt":
            text = self._extract_txt(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        text = self._clean(text)
        chunks = self._chunk(text)
        return chunks, text

    # ── Extractors ─────────────────────────────────────────────────────────────

    def _extract_pdf(self, path: Path) -> str:
        try:
            import pdfplumber
            pages: list[str] = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        pages.append(extracted)
            return "\n\n".join(pages)
        except ImportError:
            # Fallback: PyPDF2
            import PyPDF2
            pages: list[str] = []
            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
            return "\n\n".join(pages)

    def _extract_txt(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    # ── Text helpers ───────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        """Remove excessive whitespace and non-printable characters."""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[^\S\n]+", " ", text)   # collapse inline spaces
        text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 consecutive newlines
        return text.strip()

    def _chunk(self, text: str) -> list[str]:
        """
        Sliding-window character-level chunking.
        Tries to split on sentence/paragraph boundaries.
        """
        if not text:
            return []

        words = text.split()
        chunks: list[str] = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words).strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(words):
                break
            start = end - self.chunk_overlap  # slide back for overlap

        return chunks
