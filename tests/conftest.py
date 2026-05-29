"""
conftest.py
───────────
Shared fixtures for the DocMind RAG test suite.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Ensure project root is on sys.path so imports work ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLE DATA FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_short_text():
    """A short piece of text (fewer than 500 words)."""
    return (
        "The European Union's Artificial Intelligence Act establishes a "
        "comprehensive regulatory framework for AI systems. It categorises "
        "AI applications by risk level and imposes obligations accordingly."
    )


@pytest.fixture
def sample_long_text():
    """A longer text that will produce multiple chunks with default settings."""
    # ~600 words → should produce 2 chunks with chunk_size=500, overlap=50
    base = (
        "Artificial intelligence systems are transforming industries worldwide. "
        "The EU AI Act is the first comprehensive legal framework for AI. "
        "It classifies AI systems into risk categories including unacceptable, "
        "high, limited, and minimal risk. High-risk AI systems must comply "
        "with strict requirements before being placed on the market. "
    )
    return " ".join([base] * 25)  # ~625 words


@pytest.fixture
def sample_chunks():
    """Pre-built list of text chunks for testing downstream modules."""
    return [
        "The EU AI Act establishes a risk-based regulatory framework for AI.",
        "High-risk AI systems must undergo conformity assessment.",
        "General-purpose AI models with systemic risk have additional obligations.",
    ]


@pytest.fixture
def sample_question():
    return "What is the EU AI Act?"


@pytest.fixture
def sample_answer():
    return (
        "The EU AI Act is the first comprehensive legal framework for "
        "artificial intelligence, establishing risk-based regulation."
    )


@pytest.fixture
def sample_ground_truth():
    return (
        "The EU AI Act is the world's first comprehensive regulatory "
        "framework for artificial intelligence systems."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPORARY FILE FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_txt_file(tmp_path, sample_short_text):
    """Create a temporary .txt file with sample content."""
    file = tmp_path / "test_document.txt"
    file.write_text(sample_short_text, encoding="utf-8")
    return file


@pytest.fixture
def tmp_unsupported_file(tmp_path):
    """Create a temporary .docx file (unsupported format)."""
    file = tmp_path / "test_document.docx"
    file.write_text("This is a fake docx.", encoding="utf-8")
    return file


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def document_processor():
    """A DocumentProcessor with default settings."""
    from document_processor import DocumentProcessor
    return DocumentProcessor()


@pytest.fixture
def document_processor_small():
    """A DocumentProcessor with small chunk_size for easier testing."""
    from document_processor import DocumentProcessor
    return DocumentProcessor(chunk_size=10, chunk_overlap=2)


@pytest.fixture
def evaluator_instance():
    """An Evaluator instance with mocked sentence-transformer model."""
    from evaluator import Evaluator
    evaluator = Evaluator()
    return evaluator
