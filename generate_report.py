#!/usr/bin/env python3
"""
generate_report.py
Generate the DocMind full project PDF report using fpdf2.
"""

from fpdf import FPDF
from pathlib import Path

OUT = Path(r"c:\Users\R.S.Naveensankar\Downloads\files1\Report\DocMind_Full_Report.pdf")

# ── Colour palette ─────────────────────────────────────────────────────────────
C_NAVY   = (15,  23,  42)
C_BLUE   = (14, 165, 233)
C_VIOLET = (124, 58, 237)
C_SLATE  = (51,  65,  85)
C_LGRAY  = (248, 250, 252)
C_BGRAY  = (226, 232, 240)
C_WHITE  = (255, 255, 255)
C_GREEN  = (21, 128,  61)
C_LGREEN = (240, 253, 244)
C_DCODE  = (30,  30,  46)
C_LCODE  = (205, 214, 244)
C_YELL   = (234, 179,   8)
C_LBLUE  = (239, 246, 255)

L = 15    # left margin
W = 180   # usable page width


# ── PDF class ──────────────────────────────────────────────────────────────────

class PDF(FPDF):

    # ── Page chrome ──────────────────────────────────────────────────────────

    def header(self):
        if self.page_no() == 1:
            return
        self.set_y(6)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*C_SLATE)
        self.cell(0, 0, "DocMind  -  AI Document Assistant  |  Assignment Report",
                  align="C")
        self.set_draw_color(*C_BGRAY)
        self.set_line_width(0.3)
        self.line(L, 12, L + W, 12)
        self.set_y(17)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-13)
        self.set_draw_color(*C_BGRAY)
        self.set_line_width(0.3)
        self.line(L, self.get_y(), L + W, self.get_y())
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*C_SLATE)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    # ── Heading helpers ───────────────────────────────────────────────────────

    def h1(self, text):
        self.ln(5)
        self.set_fill_color(*C_BLUE)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", 13)
        self.set_x(L)
        self.cell(W, 9, f"  {text}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_text_color(*C_NAVY)

    def h2(self, text):
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*C_BLUE)
        self.set_x(L)
        self.cell(W, 7, text, new_x="LMARGIN", new_y="NEXT")
        y = self.get_y()
        self.set_draw_color(*C_BLUE)
        self.set_line_width(0.4)
        self.line(L, y, L + W, y)
        self.ln(3)
        self.set_text_color(*C_NAVY)

    def h3(self, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C_VIOLET)
        self.set_x(L)
        self.cell(W, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(*C_NAVY)

    def h4(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*C_NAVY)
        self.set_x(L)
        self.cell(W, 5.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    # ── Body content helpers ──────────────────────────────────────────────────

    def body(self, text, indent=0):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*C_NAVY)
        self.set_x(L + indent)
        self.multi_cell(W - indent, 5.5, text)
        self.ln(1)

    def bullet(self, text, level=0, bold_prefix=""):
        indent = level * 6
        self.set_x(L + 4 + indent)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*C_NAVY)
        marker = "-"
        if bold_prefix:
            # Draw marker + bold key + normal value
            self.set_font("Helvetica", "", 9)
            self.cell(4, 5, marker)
            self.set_font("Helvetica", "B", 9)
            self.cell(self.get_string_width(bold_prefix) + 2, 5, bold_prefix)
            self.set_font("Helvetica", "", 9)
            self.multi_cell(W - 12 - indent - self.get_string_width(bold_prefix), 5, text)
        else:
            self.cell(4, 5, marker)
            self.multi_cell(W - 10 - indent, 5, text)

    def kv(self, key, value, key_w=55):
        self.set_x(L)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C_BLUE)
        self.cell(key_w, 5.5, key + ":")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*C_NAVY)
        self.multi_cell(W - key_w, 5.5, str(value))

    def code(self, text):
        self.set_fill_color(*C_DCODE)
        self.set_text_color(*C_LCODE)
        self.set_font("Courier", "", 8)
        self.set_x(L)
        self.multi_cell(W, 4.8, text, fill=True)
        self.set_text_color(*C_NAVY)
        self.ln(2)

    def note(self, text):
        self.set_fill_color(*C_LBLUE)
        self.set_draw_color(*C_BLUE)
        self.set_line_width(0.3)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*C_SLATE)
        self.set_x(L)
        self.multi_cell(W, 5, text, border="L", fill=True)
        self.set_text_color(*C_NAVY)
        self.ln(2)

    def success_box(self, text):
        self.set_fill_color(*C_LGREEN)
        self.set_draw_color(*C_GREEN)
        self.set_line_width(0.5)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C_GREEN)
        self.set_x(L)
        self.multi_cell(W, 6, text, border="L", fill=True)
        self.set_text_color(*C_NAVY)
        self.ln(2)

    # ── Table helper ──────────────────────────────────────────────────────────

    def _split_cell(self, text, cw):
        """Split text into lines that fit inside column width cw."""
        words = str(text).split()
        if not words:
            return [""]
        lines, cur = [], ""
        for w in words:
            test = (cur + " " + w).strip() if cur else w
            if self.get_string_width(test) <= cw - 4:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines if lines else [""]

    def table(self, headers, rows, col_widths=None, font_size=8.5):
        """Render a full table with auto-height rows."""
        if col_widths is None:
            col_widths = [W // len(headers)] * len(headers)

        lh = font_size * 0.65  # line height

        # Header row
        self.set_fill_color(*C_BLUE)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", font_size)
        self.set_x(L)
        for h, cw in zip(headers, col_widths):
            self.cell(cw, 7, str(h), border=1, fill=True)
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", font_size)
        for idx, row in enumerate(rows):
            fill_color = C_LGRAY if idx % 2 == 0 else C_WHITE
            self.set_fill_color(*fill_color)
            self.set_text_color(*C_NAVY)

            # Pre-split every cell and find row height
            split_cells = [self._split_cell(cell, cw)
                           for cell, cw in zip(row, col_widths)]
            max_lines = max(len(sc) for sc in split_cells)
            row_h = max(6, max_lines * (lh + 1.5) + 2)

            y0 = self.get_y()
            x0 = L
            for sc, cw in zip(split_cells, col_widths):
                # Draw filled border box
                self.set_xy(x0, y0)
                self.cell(cw, row_h, "", border=1, fill=True)
                # Draw text lines inside
                for li, line in enumerate(sc):
                    self.set_xy(x0 + 1.5, y0 + 1.5 + li * (lh + 1.5))
                    self.cell(cw - 3, lh, line)
                x0 += cw

            self.set_y(y0 + row_h)

        self.ln(3)

    # ── Cover Page ────────────────────────────────────────────────────────────

    def cover_page(self):
        self.add_page()
        # Dark gradient background  (solid dark navy)
        self.set_fill_color(*C_NAVY)
        self.rect(0, 0, 210, 297, "F")

        # Decorative accent strip
        self.set_fill_color(*C_BLUE)
        self.rect(0, 120, 210, 5, "F")
        self.set_fill_color(*C_VIOLET)
        self.rect(0, 125, 210, 2, "F")

        # Main title
        self.set_y(50)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*C_WHITE)
        self.cell(0, 15, "DocMind", align="C", new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "", 16)
        self.set_text_color(*C_BLUE)
        self.cell(0, 10, "AI Document Assistant", align="C",
                  new_x="LMARGIN", new_y="NEXT")

        self.ln(4)

        # Subtitle
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(148, 163, 184)  # slate-400
        self.cell(0, 8,
                  "RAG Pipeline: Implementation, Testing & Evaluation",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "Full Project Report", align="C",
                  new_x="LMARGIN", new_y="NEXT")

        # Tech badges
        self.ln(12)
        badges = ["Python 3.11", "Streamlit", "ChromaDB",
                  "Ollama llama3.2:3b", "MiniLM-L6-v2", "pytest 100% cov"]
        badge_w = 28
        start_x = (210 - len(badges) * badge_w) / 2
        for i, b in enumerate(badges):
            bx = start_x + i * badge_w
            self.set_fill_color(*C_VIOLET)
            self.set_text_color(*C_WHITE)
            self.set_font("Helvetica", "", 7)
            self.set_xy(bx, self.get_y())
            self.cell(badge_w - 2, 6, b, fill=True, align="C")

        # Stats strip
        self.set_y(175)
        stats = [("4", "Core Modules"), ("152", "Test Cases"),
                 ("100%", "Coverage"), ("8", "Eval Metrics"), ("11", "Q Types")]
        sw = W / len(stats)
        self.set_x(L)
        for val, label in stats:
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*C_BLUE)
            cx = self.get_x()
            self.cell(sw, 12, val, align="C")
            self.set_xy(cx, self.get_y() + 12)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(148, 163, 184)
            self.cell(sw, 6, label, align="C")
            self.set_xy(cx + sw, self.get_y() - 12)

        # Meta info at bottom
        self.set_y(240)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(148, 163, 184)
        self.cell(0, 7, "Author: R.S. Naveensankar", align="C",
                  new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 7,
                  "GitHub: https://github.com/naveensankar5905/simple-rag",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 7, "May 2025", align="C",
                  new_x="LMARGIN", new_y="NEXT")

    # ── Table of Contents ─────────────────────────────────────────────────────

    def toc_page(self):
        self.add_page()
        self.h1("Table of Contents")
        sections = [
            ("1",  "Project Overview"),
            ("2",  "System Architecture"),
            ("3",  "Technical Stack & Prerequisites"),
            ("4",  "Core Module Design"),
            ("   4.1", "document_processor.py"),
            ("   4.2", "vector_store.py"),
            ("   4.3", "qa_engine.py"),
            ("   4.4", "evaluator.py"),
            ("   4.5", "app.py  (Streamlit UI)"),
            ("5",  "Setup Instructions"),
            ("6",  "How to Use DocMind"),
            ("7",  "Evaluation Metrics"),
            ("8",  "Question Types & Answer Formats"),
            ("9",  "Configuration Reference"),
            ("10", "Test Report  --  What, How & Why"),
            ("   10.1", "DocumentProcessor tests (24)"),
            ("   10.2", "VectorStore tests (11)"),
            ("   10.3", "QAEngine tests (54)"),
            ("   10.4", "Evaluator tests (63)"),
            ("   10.5", "Testing strategy & mocking philosophy"),
            ("   10.6", "Coverage results"),
            ("11", "Troubleshooting"),
            ("12", "Project Structure"),
            ("13", "Dependencies"),
        ]
        self.set_font("Helvetica", "", 10)
        for num, title in sections:
            bold = not num.startswith(" ")
            self.set_x(L)
            if bold:
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(*C_NAVY)
            else:
                self.set_font("Helvetica", "", 9.5)
                self.set_text_color(*C_SLATE)
            self.cell(20, 7, num)
            self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*C_NAVY)


# ── Report sections ────────────────────────────────────────────────────────────

def section_overview(p):
    p.add_page()
    p.h1("1.  Project Overview")
    p.body(
        "DocMind is a production-ready Retrieval-Augmented Generation (RAG) application "
        "that lets users upload any PDF or TXT document, ask natural-language questions about it, "
        "and instantly evaluate the AI-generated answer against a user-supplied ground-truth using "
        "eight NLP metrics. It is built entirely in Python and runs locally with no cloud dependencies."
    )
    p.h2("Key Features")
    feats = [
        ("Document Upload",       "PDF (pdfplumber + PyPDF2 fallback) and plain TXT"),
        ("Smart Chunking",        "Sliding-window, 500 words / 50-word overlap"),
        ("Dense Embeddings",      "all-MiniLM-L6-v2 via sentence-transformers (384-dim)"),
        ("Vector Store",          "ChromaDB in-memory with cosine similarity indexing"),
        ("LLM Integration",       "llama3.2:3b via local Ollama REST API (no internet)"),
        ("Question-Type Router",  "Auto-detects 11 types; selects format-specific system prompt"),
        ("Evaluation Matrix",     "Token F1, Precision, Recall, ROUGE-1/2/L, Semantic Sim, EM"),
        ("Web UI",                "Streamlit  --  3-column layout: Upload | Q&A | Evaluate"),
    ]
    p.table(
        ["Feature", "Description"],
        feats,
        [50, 130],
    )
    p.h2("Why RAG?")
    p.body(
        "Large language models hallucinate when asked about content they were not trained on. "
        "RAG grounds every answer in the actual document: relevant chunks are retrieved from the "
        "vector store and injected into the LLM context, so the model can only say what is in "
        "the document. The evaluation matrix then quantifies how accurate that answer is compared "
        "to a human-written ground truth."
    )


def section_architecture(p):
    p.add_page()
    p.h1("2.  System Architecture")
    p.body(
        "DocMind is divided into four independent pipeline modules orchestrated by the Streamlit "
        "UI. Each module has a single responsibility, is independently testable, and has its own "
        "test file."
    )
    p.h2("Pipeline Flow")
    p.code(
        "User uploads PDF/TXT\n"
        "       |\n"
        "       v\n"
        " DocumentProcessor          extract text  ->  clean  ->  sliding-window chunks\n"
        "       |\n"
        "       v\n"
        " VectorStore                embed chunks with MiniLM-L6-v2  ->  store in ChromaDB\n"
        "       |\n"
        " User types question\n"
        "       |\n"
        "       v\n"
        " VectorStore.query()        embed question  ->  cosine search  ->  top-k chunks\n"
        "       |\n"
        "       v\n"
        " QAEngine.answer()          detect question type  ->  build prompt  ->  Ollama API\n"
        "       |\n"
        "       v\n"
        " User supplies ground truth\n"
        "       |\n"
        "       v\n"
        " Evaluator.evaluate()       8 NLP metrics  ->  overall score  ->  verdict"
    )
    p.h2("Module Responsibilities")
    p.table(
        ["Module", "File", "Responsibility"],
        [
            ("DocumentProcessor", "document_processor.py",
             "PDF/TXT text extraction, whitespace cleaning, sliding-window chunking"),
            ("VectorStore", "vector_store.py",
             "Sentence-transformer embedding, ChromaDB CRUD, cosine similarity retrieval"),
            ("QAEngine", "qa_engine.py",
             "Question-type detection, system-prompt assembly, Ollama REST call, error handling"),
            ("Evaluator", "evaluator.py",
             "Token F1, ROUGE-1/2/L, semantic cosine similarity, EM, overall weighted score"),
            ("Streamlit UI", "app.py",
             "3-column layout, session state, cached singletons, upload/Q&A/evaluation flow"),
        ],
        [38, 52, 90],
    )


def section_stack(p):
    p.add_page()
    p.h1("3.  Technical Stack & Prerequisites")
    p.h2("AI Models")
    p.table(
        ["Model", "Purpose", "Size", "Source"],
        [
            ("llama3.2:3b",    "Answer generation (LLM)", "~2 GB",   "Ollama (local)"),
            ("all-MiniLM-L6-v2", "Text embeddings",      "~90 MB",  "Hugging Face / sentence-transformers"),
        ],
        [48, 60, 22, 50],
    )
    p.h2("Python Libraries")
    p.table(
        ["Library", "Version", "Role"],
        [
            ("streamlit",            ">=1.35", "Web UI framework"),
            ("pdfplumber",           ">=0.11", "PDF text extraction (primary)"),
            ("PyPDF2",               ">=3.0",  "PDF extraction fallback"),
            ("sentence-transformers",">=2.7",  "MiniLM-L6-v2 embeddings"),
            ("torch",                ">=2.2",  "Required by sentence-transformers"),
            ("chromadb",             ">=0.5",  "In-memory vector database"),
            ("rouge-score",          ">=0.1",  "ROUGE-1/2/L evaluation metrics"),
            ("numpy",                ">=1.26", "Numerical utilities"),
            ("pytest",               ">=8.0",  "Test framework"),
            ("pytest-cov",           ">=5.0",  "Coverage reporting"),
            ("pytest-mock",          ">=3.14", "Enhanced mocking utilities"),
        ],
        [55, 30, 95],
    )
    p.h2("Prerequisites")
    p.table(
        ["Requirement", "Version", "Check command"],
        [
            ("Python",         ">= 3.10",  "python --version"),
            ("pip",            ">= 23",    "pip --version"),
            ("Ollama",         "latest",   "ollama --version"),
            ("llama3.2:3b",    "pulled",   "ollama list"),
            ("Git (optional)", "any",      "git --version"),
        ],
        [55, 30, 95],
    )
    p.note(
        "Hardware note: llama3.2:3b runs on CPU (slow) or GPU/Apple Silicon (fast). "
        "Minimum 8 GB RAM is recommended."
    )


def section_modules(p):
    p.add_page()
    p.h1("4.  Core Module Design")

    # 4.1 document_processor.py
    p.h2("4.1  document_processor.py")
    p.body(
        "Handles all document ingestion. Accepts a file path and returns a tuple "
        "(chunks: list[str], full_text: str). Supports PDF and TXT; raises ValueError "
        "for any other extension."
    )
    p.h3("Class: DocumentProcessor")
    p.kv("chunk_size", "int = 500   (words per chunk)")
    p.kv("chunk_overlap", "int = 50    (overlap between consecutive chunks)")
    p.ln(2)
    p.h4("Key methods:")
    p.table(
        ["Method", "Description"],
        [
            ("process(file_path)",   "Main entry point. Calls extractor -> _clean -> _chunk"),
            ("_extract_pdf(path)",   "pdfplumber primary; ImportError falls back to PyPDF2"),
            ("_extract_txt(path)",   "Path.read_text(encoding=utf-8, errors=replace)"),
            ("_clean(text)",         "Normalise \\r\\n -> \\n, collapse spaces, limit consecutive newlines to 2"),
            ("_chunk(text)",         "Sliding window on word list. step = chunk_size - chunk_overlap"),
        ],
        [55, 125],
    )
    p.h3("Sliding-Window Chunking Algorithm")
    p.body(
        "The text is split on whitespace into a word list. Starting at index 0, each chunk takes "
        "chunk_size consecutive words. The next chunk starts at (previous_start + chunk_size - "
        "chunk_overlap), guaranteeing that the last chunk_overlap words of one chunk are also the "
        "first chunk_overlap words of the next. This preserves sentence context across boundaries."
    )

    # 4.2 vector_store.py
    p.add_page()
    p.h2("4.2  vector_store.py")
    p.body(
        "Wraps ChromaDB with a sentence-transformer embedding function. "
        "All data is stored in-memory (no disk or server required). "
        "Uses cosine similarity metric for nearest-neighbour search."
    )
    p.h3("Class: VectorStore")
    p.kv("EMBEDDING_MODEL", "\"all-MiniLM-L6-v2\"  (384-dimensional, MIT licence)")
    p.kv("COLLECTION_NAME", "\"rag_docs\"")
    p.ln(2)
    p.h4("Key methods:")
    p.table(
        ["Method", "Description"],
        [
            ("__init__()",              "Creates chromadb.Client() and get_or_create_collection with cosine metadata"),
            ("add_documents(chunks)",   "Generates UUIDs, calls _col.add(documents=chunks, ids=uuids). No-op for empty list."),
            ("query(question, n=3)",    "Caps n to collection count (avoids ChromaDB error). Returns {documents, distances}."),
            ("count()",                 "Delegates to _col.count()."),
        ],
        [55, 125],
    )
    p.note(
        "The n_results cap in query() is essential: ChromaDB raises an error if you request "
        "more results than documents stored. The guard `n = min(n, self.count())` prevents "
        "this when the collection has fewer than top-k documents."
    )

    # 4.3 qa_engine.py
    p.add_page()
    p.h2("4.3  qa_engine.py")
    p.body(
        "Connects to a locally running Ollama process via HTTP (POST to "
        "http://localhost:11434/api/generate). Assembles a question-type-aware prompt "
        "and returns the generated answer. All HTTP calls use the standard library "
        "urllib.request (no extra dependencies)."
    )
    p.h3("Question Type Detection")
    p.body(
        "The standalone function detect_question_type(question) applies a set of regex "
        "patterns (case-insensitive) in priority order and returns one of 11 type tokens:"
    )
    p.table(
        ["Type", "Trigger patterns (examples)", "Answer format guided by system prompt"],
        [
            ("YES_NO",  "does / is / can / will / has / did / was / were / should / would / could / do / have / are", "Starts with YES -- / NO -- / PARTIALLY --"),
            ("WHEN",    "when did / when was",   "Exact date/range first, then context"),
            ("WHERE",   "where is / where was",  "Location immediately + spatial context"),
            ("WHO",     "who is / who was",       "Name + title/affiliation from document"),
            ("WHY",     "why did / why is",        "Reason + evidence + consequence"),
            ("HOW",     "how does / how to",       "Numbered steps or cause->effect"),
            ("COMPARE", "compare / difference / versus / contrast", "Side-by-side with contrast language"),
            ("LIST",    "list / what are / name all", "Bullet list + item count"),
            ("DEFINE",  "define / what does X mean / meaning of", "One-line definition then elaboration"),
            ("SUMMARY", "summarise / overview / describe", "5-8 sentences of flowing prose"),
            ("WHAT",    "fallback for all others",  "One-sentence ID + 2-4 supporting details"),
        ],
        [22, 62, 96],
        font_size=8,
    )
    p.h3("Ollama API Parameters")
    p.table(
        ["Parameter", "Value", "Purpose"],
        [
            ("model",       "llama3.2:3b", "Local Ollama model tag"),
            ("temperature", "0.2",         "Low temperature for factual, deterministic answers"),
            ("top_p",       "0.9",         "Nucleus sampling for coherent output"),
            ("num_predict", "768",         "Max output tokens"),
        ],
        [40, 30, 110],
    )
    p.h4("Error handling in _call_ollama:")
    p.bullet("urllib.error.URLError  ->  returns 'Could not connect to Ollama: <reason>'")
    p.bullet("Any other exception   ->  returns 'Unexpected error: <str(e)>'")
    p.body("Both paths return a string (never raise), so the UI always shows readable feedback.")

    # 4.4 evaluator.py
    p.add_page()
    p.h2("4.4  evaluator.py")
    p.body(
        "Computes eight NLP metrics comparing a model-generated prediction against a "
        "human-provided ground truth. Returns an EvalResult dataclass with all scores "
        "and a human-readable verdict."
    )
    p.h3("EvalResult dataclass fields")
    p.table(
        ["Field", "Type", "Description"],
        [
            ("exact_match",        "bool",  "True if normalised prediction == normalised ground truth"),
            ("token_precision",    "float", "Fraction of predicted tokens found in ground truth"),
            ("token_recall",       "float", "Fraction of ground-truth tokens found in prediction"),
            ("token_f1",           "float", "Harmonic mean of precision and recall"),
            ("rouge1",             "float", "Unigram overlap F-measure (rouge-score library or manual fallback)"),
            ("rouge2",             "float", "Bigram overlap F-measure"),
            ("rougeL",             "float", "Longest Common Subsequence F-measure"),
            ("semantic_similarity","float", "Cosine similarity of MiniLM-L6-v2 sentence embeddings"),
            ("overall_score",      "float", "Weighted composite (see formula below)"),
            ("verdict",            "str",   "Excellent / Good / Partial / Weak / Poor"),
            ("details",            "dict",  "Normalised prediction and ground truth (first 200 chars each)"),
        ],
        [42, 18, 120],
    )
    p.h3("Overall Score Formula  (weights sum to 1.0)")
    p.code(
        "overall_score = ( 0.35 * semantic_similarity\n"
        "               + 0.25 * token_f1\n"
        "               + 0.20 * rougeL\n"
        "               + 0.15 * rouge1\n"
        "               + 0.05 * (1.0 if exact_match else 0.0) )"
    )
    p.h3("Verdict Thresholds")
    p.table(
        ["Score Range", "Verdict",   "Meaning"],
        [
            (">= 85%", "EXCELLENT", "Answer is nearly perfect"),
            (">= 70%", "GOOD",      "Covers most key content"),
            (">= 50%", "PARTIAL",   "Partially correct, missing detail"),
            (">= 30%", "WEAK",      "Significant gaps"),
            ("<  30%", "POOR",      "Answer does not match ground truth"),
        ],
        [40, 35, 105],
    )
    p.h3("Fallback Strategy")
    p.bullet(
        "ROUGE: If rouge-score library is unavailable, falls back to a pure-Python "
        "n-gram F-measure (_manual_rouge_n) and an O(mn) dynamic-programming LCS "
        "implementation (_manual_rouge_l)."
    )
    p.bullet(
        "Semantic similarity: If sentence-transformers fails to load, falls back to "
        "Jaccard similarity on normalised token sets."
    )
    p.bullet(
        "Sentence-transformer model is loaded lazily (first call) and cached at the "
        "class level (_st_model), so subsequent calls are instant."
    )

    # 4.5 app.py
    p.add_page()
    p.h2("4.5  app.py  (Streamlit Web UI)")
    p.body(
        "The Streamlit front-end orchestrates the four pipeline modules via a "
        "3-column layout and persists state across page reruns using st.session_state."
    )
    p.h3("Column Layout")
    p.table(
        ["Column", "Contents"],
        [
            ("Left -- Upload",    "File uploader (PDF/TXT), chunking + embedding pipeline, pipeline info, question types guide"),
            ("Middle -- Q&A",     "Question input, Ask button, question-type badge, answer display, source-chunks expander"),
            ("Right -- Evaluate", "Ground-truth text area, Evaluate button, 8-metric grid, colour-coded verdict, metric explanations"),
        ],
        [40, 140],
    )
    p.h3("Session State variables")
    p.table(
        ["Key", "Description"],
        [
            ("vector_store",   "VectorStore instance for the active document"),
            ("doc_stats",      "Dict with chunk_count and char_count"),
            ("qa_history",     "List of last 5 (question, answer) pairs"),
            ("doc_name",       "Name of the currently loaded document"),
            ("last_answer",    "Most recent answer text"),
            ("last_qtype",     "Detected question type of last query"),
            ("last_sources",   "Retrieved chunk texts from last query"),
            ("last_question",  "Text of last question asked"),
        ],
        [38, 142],
    )
    p.h3("Cached singletons  (@st.cache_resource)")
    p.bullet("DocumentProcessor()  --  created once per session")
    p.bullet("QAEngine()           --  created once per session")
    p.bullet("Evaluator()          --  created once per session")
    p.body(
        "Caching avoids model reloads on every Streamlit rerun triggered by user interaction."
    )
    p.note(
        "app.py is excluded from automated pytest coverage because it requires an interactive "
        "browser session. It is validated manually."
    )


def section_setup(p):
    p.add_page()
    p.h1("5.  Setup Instructions")

    steps = [
        ("Step 1 -- Get project files",
         "Place all source files in a single folder (e.g. rag_assistant/):\n"
         "app.py, document_processor.py, vector_store.py, qa_engine.py, evaluator.py, requirements.txt"),
        ("Step 2 -- Create virtual environment",
         "python3 -m venv .venv\n"
         "# macOS/Linux:  source .venv/bin/activate\n"
         "# Windows PS:   .venv\\Scripts\\Activate.ps1"),
        ("Step 3 -- Install Python dependencies",
         "pip install --upgrade pip\npip install -r requirements.txt"),
        ("Step 4 -- Install Ollama",
         "macOS:   brew install ollama\n"
         "Linux:   curl -fsSL https://ollama.com/install.sh | sh\n"
         "Windows: download installer from ollama.com"),
        ("Step 5 -- Pull llama3.2:3b model",
         "ollama pull llama3.2:3b\n"
         "# Downloads ~2 GB. Verify: ollama list"),
        ("Step 6 -- Start Ollama server",
         "ollama serve\n"
         "# Keep this terminal open. Should print: Listening on 127.0.0.1:11434"),
        ("Step 7 -- Launch DocMind",
         "streamlit run app.py\n"
         "# Open browser at http://localhost:8501"),
    ]

    for title, content in steps:
        p.h3(title)
        p.code(content)


def section_usage(p):
    p.add_page()
    p.h1("6.  How to Use DocMind")

    p.h2("Step 1 -- Upload a Document (left column)")
    p.bullet("Click Browse files or drag-and-drop a PDF or TXT file.")
    p.bullet("DocMind extracts text, splits into overlapping chunks, embeds with MiniLM, and stores in ChromaDB.")
    p.bullet("The sidebar shows chunk count and character count once indexing is done.")

    p.h2("Step 2 -- Ask a Question (middle column)")
    p.bullet("Type any natural-language question about your document.")
    p.bullet("Click Ask ->. DocMind auto-detects the question type and selects the matching answer format.")
    p.bullet("The detected type appears as a badge above the answer.")
    p.bullet("Expand Source chunks to see exactly which document passages were retrieved.")

    p.h2("Step 3 -- Evaluate the Answer (right column)")
    p.bullet("Paste your own expected / ground-truth answer into the text area.")
    p.bullet("Click Evaluate. All 8 metrics are computed instantly -- no external API needed.")
    p.bullet("A colour-coded metric grid and verdict are displayed.")

    p.h2("Sidebar Controls")
    p.table(
        ["Control", "Description"],
        [
            ("Top-k slider (1-8)", "Number of document chunks retrieved for each query (default 3)"),
            ("Document status",    "Shows chunk count and character count of loaded document"),
            ("Clear document",     "Removes current document and resets session state"),
            ("Q&A history",        "Shows the last 5 question-answer pairs"),
        ],
        [48, 132],
    )


def section_metrics(p):
    p.add_page()
    p.h1("7.  Evaluation Metrics")
    p.body(
        "Eight metrics are computed and combined into an overall weighted score. "
        "All metrics are normalised to [0, 1]."
    )
    p.table(
        ["Metric", "How it works", "Weight"],
        [
            ("Token Precision",     "Fraction of predicted words that appear in ground truth", "--"),
            ("Token Recall",        "Fraction of ground-truth words that appear in prediction", "--"),
            ("Token F1",            "Harmonic mean of precision and recall on normalised tokens", "25%"),
            ("ROUGE-1",             "Unigram overlap F-measure between prediction and ground truth", "15%"),
            ("ROUGE-2",             "Bigram overlap F-measure", "--"),
            ("ROUGE-L",             "Longest Common Subsequence F-measure", "20%"),
            ("Semantic Similarity", "Cosine distance between all-MiniLM-L6-v2 sentence embeddings", "35%"),
            ("Exact Match",         "1.0 if normalised strings are identical; else 0.0", "5% bonus"),
        ],
        [42, 110, 22],
    )
    p.note(
        "Semantic Similarity has the highest weight (35%) because embedding-based cosine distance "
        "captures paraphrase and synonym relationships that exact-match or n-gram metrics miss."
    )
    p.h2("Verdict Thresholds")
    p.table(
        ["Score",  "Verdict",   "Meaning"],
        [
            (">= 85%", "EXCELLENT", "Answer is nearly perfect -- all key facts present"),
            (">= 70%", "GOOD",      "Covers most key content with minor gaps"),
            (">= 50%", "PARTIAL",   "Partially correct; significant detail missing"),
            (">= 30%", "WEAK",      "Significant factual gaps or wrong information"),
            ("<  30%", "POOR",      "Answer does not match ground truth"),
        ],
        [25, 32, 123],
    )
    p.h2("Normalisation")
    p.body(
        "Before computing token-level metrics and exact match, both strings are normalised: "
        "lowercased, punctuation removed (replaced by space), and whitespace collapsed. "
        "This ensures 'Hello, World!' == 'hello world' for exact match purposes."
    )


def section_qtypes(p):
    p.add_page()
    p.h1("8.  Question Types & Answer Formats")
    p.body(
        "DocMind's question-type router applies regex patterns (case-insensitive) in priority "
        "order. The detected type is injected into the system prompt as a format hint, so the "
        "LLM knows exactly how to structure its answer."
    )
    p.table(
        ["Type", "Trigger words", "Answer format"],
        [
            ("WHAT",    "what is / what does",       "One-sentence identification + 2-4 supporting details"),
            ("WHEN",    "when did / when was",         "Exact date/range first, then temporal context"),
            ("WHERE",   "where is / where was",        "Location immediately + spatial context"),
            ("WHO",     "who is / who was",             "Name + title/affiliation from document"),
            ("WHY",     "why did / why is",              "Reason in sentence 1 + evidence + consequence chain"),
            ("HOW",     "how does / how to",             "Numbered steps (procedure) or cause->effect (mechanism)"),
            ("YES_NO",  "does/is/can/will/has/did/was/were/should/would/could/do/have/are",
             "Starts with YES -- / NO -- / PARTIALLY --"),
            ("LIST",    "list / what are / name all",  "Bullet list of all relevant items + item count"),
            ("COMPARE", "compare/difference/versus/contrast", "Side-by-side with contrast language"),
            ("DEFINE",  "define / meaning of / what does X mean", "One-line definition then elaboration"),
            ("SUMMARY", "summarise / overview / describe",  "5-8 sentences of flowing prose"),
        ],
        [22, 60, 98],
        font_size=8,
    )
    p.note(
        "Pattern priority: YES_NO patterns are checked first (they use auxiliary verb openers "
        "that could otherwise clash with other patterns). WHAT is the fallback catch-all for "
        "any question that does not match a more specific pattern."
    )


def section_config(p):
    p.add_page()
    p.h1("9.  Configuration Reference")
    p.table(
        ["Parameter", "File", "Default", "Description"],
        [
            ("chunk_size",       "app.py / DocumentProcessor", "500",              "Words per chunk"),
            ("chunk_overlap",    "app.py / DocumentProcessor", "50",               "Overlap words between chunks"),
            ("top_k",            "Sidebar slider",             "3  (range 1-8)",   "Chunks retrieved per query"),
            ("temperature",      "qa_engine.py",               "0.2",              "LLM sampling temperature (lower = more factual)"),
            ("num_predict",      "qa_engine.py",               "768",              "Max tokens generated in one answer"),
            ("top_p",            "qa_engine.py",               "0.9",              "Nucleus sampling threshold"),
            ("embedding_model",  "vector_store.py",            "all-MiniLM-L6-v2","Sentence-transformer model for embeddings"),
            ("llm_model",        "qa_engine.py",               "llama3.2:3b",      "Ollama model tag"),
            ("OLLAMA_URL",       "qa_engine.py",               "localhost:11434",  "Ollama REST endpoint"),
        ],
        [38, 50, 36, 56],
        font_size=8,
    )
    p.h2("Tuning Guidance")
    p.bullet("Increase chunk_size for documents where context spans many paragraphs (e.g. legal texts).")
    p.bullet("Increase chunk_overlap when answers frequently straddle chunk boundaries.")
    p.bullet("Lower temperature (0.1-0.15) for more deterministic, factual answers.")
    p.bullet("Increase top_k (4-6) when relevant context is spread across many sections.")
    p.bullet(
        "Replace llama3.2:3b with a larger model (e.g. llama3.1:8b) for higher answer quality "
        "at the cost of slower generation."
    )


def section_tests(p):
    p.add_page()
    p.h1("10.  Test Report  --  What, How & Why")

    p.success_box(
        "152 tests  |  4 test files  |  100% coverage  |  0 failures  |  ~35 seconds"
    )

    p.body(
        "The test suite validates all four pipeline modules using pytest with unittest.mock. "
        "Every external dependency (Ollama API, ChromaDB, sentence-transformers, "
        "pdfplumber/PyPDF2) is mocked so tests run in ~35 seconds with no GPU, "
        "no network, and no database required."
    )

    # Overview table
    p.h2("Test Suite Overview")
    p.table(
        ["Module", "Test File", "Tests", "Coverage"],
        [
            ("document_processor.py", "test_document_processor.py", "24", "100%"),
            ("vector_store.py",       "test_vector_store.py",       "11", "100%"),
            ("qa_engine.py",          "test_qa_engine.py",          "54", "100%"),
            ("evaluator.py",          "test_evaluator.py",          "63", "100%"),
            ("TOTAL",                 "4 files",                   "152", "100%"),
        ],
        [55, 65, 25, 35],
    )

    # ── 10.1 DocumentProcessor ────────────────────────────────────────────────
    p.add_page()
    p.h2("10.1  DocumentProcessor  (24 tests, 100%)")

    p.h3("What is tested")
    p.table(
        ["Test Class", "Count", "Scope"],
        [
            ("TestDocumentProcessorInit",        "2",  "Default and custom chunk_size / chunk_overlap stored correctly"),
            ("TestTextExtraction",               "7",  "process() for TXT; _extract_pdf via pdfplumber and PyPDF2 fallback; unsupported extension raises ValueError; multi-page join; None-page skipping"),
            ("TestTextCleaning",                 "5",  "_clean: carriage-return normalisation, inline-space collapse, max-2 consecutive newlines, strip, combined rules"),
            ("TestChunking",                     "7",  "_chunk: empty input, single chunk, multiple chunks, overlap words, exact-boundary, no empty chunks, all words covered"),
            ("TestDocumentProcessorIntegration", "3",  "End-to-end process() from disk file through clean and chunk"),
        ],
        [54, 14, 112],
        font_size=8,
    )

    p.h3("How it is implemented")
    p.bullet(
        "Temp files (tmp_path fixture): real OS file I/O -- exercises Path.read_text "
        "encoding handling, not just string content."
    )
    p.bullet(
        "PDF mocking: pdfplumber.open replaced with MagicMock exposing a pages list, "
        "each returning controlled extract_text() values."
    )
    p.bullet(
        "PyPDF2 fallback: pdfplumber.open patched to raise ImportError, then "
        "PyPDF2.PdfReader patched to return a controlled mock."
    )
    p.bullet(
        "document_processor_small fixture (chunk_size=10, overlap=2): keeps synthetic "
        "strings short while reliably producing multiple chunks."
    )

    p.h3("Why this approach")
    p.body(
        "Chunking is the most critical transformation -- incorrect overlap or off-by-one "
        "errors silently degrade retrieval quality. Testing the exact-boundary case (exactly "
        "chunk_size words -> 1 chunk) and the overlap-word check (last N words of chunk[0] == "
        "first N words of chunk[1]) catches the most common sliding-window bugs. Real file I/O "
        "is preferred over mock_open because it exercises actual Path.read_text encoding "
        "handling that mock_open bypasses."
    )

    # ── 10.2 VectorStore ──────────────────────────────────────────────────────
    p.add_page()
    p.h2("10.2  VectorStore  (11 tests, 100%)")

    p.h3("What is tested")
    p.table(
        ["Test Class", "Count", "Scope"],
        [
            ("TestVectorStoreInit",  "2", "chromadb.Client call, SentenceTransformerEmbeddingFunction construction, get_or_create_collection with cosine metadata, class constants"),
            ("TestAddDocuments",     "3", "Empty list -> no-op (add not called); multiple chunks stored with valid UUID ids; single chunk path"),
            ("TestQuery",            "4", "Result pass-through; empty-collection guard returns empty lists without calling ChromaDB; n_results capping; default n_results=3"),
            ("TestCount",            "2", "Delegation to _col.count(); empty collection returns 0"),
        ],
        [40, 14, 126],
        font_size=8,
    )

    p.h3("How it is implemented")
    p.bullet(
        "_make_mock_vs() helper: builds a VectorStore inside a with-patch block so "
        "chromadb.Client and SentenceTransformerEmbeddingFunction are intercepted, "
        "returning the instance and mock collection."
    )
    p.bullet(
        "UUID validation: uuid.UUID(id_str) is called on each generated ID -- raises "
        "ValueError on any malformed string, more robust than len check."
    )
    p.bullet(
        "n_results cap: mock_col.count.return_value = 2, request n_results=10, assert "
        "query called with n_results=2."
    )

    p.h3("Why this approach")
    p.body(
        "ChromaDB starts an in-memory DuckDB process and sentence-transformers downloads "
        "a 90 MB model on first use. Mocking both eliminates a ~15-second test penalty "
        "and removes all network/GPU requirements. UUID validation is more meaningful than "
        "a length check because it confirms the exact UUID4 format."
    )

    # ── 10.3 QAEngine ─────────────────────────────────────────────────────────
    p.add_page()
    p.h2("10.3  QAEngine  (54 tests, 100%)")

    p.h3("What is tested")
    p.table(
        ["Test Class", "Count", "Scope"],
        [
            ("TestDetectQuestionType", "34", "All 11 types; all 14 YES_NO trigger words individually; COMPARE/SUMMARY/LIST/DEFINE patterns; case-insensitivity; leading whitespace; default WHAT fallback"),
            ("TestQAEngineInit",       "3",  "Default and custom model names; OLLAMA_URL constant value"),
            ("TestFormatContext",       "3",  "Multiple chunks labelled [Chunk N]; empty list -> empty string; single chunk"),
            ("TestBuildPrompt",         "5",  "System prompt present; question present; context present; type-specific hint for all 11 types; unknown type -> no crash"),
            ("TestCallOllama",          "5",  "Successful JSON response; URLError -> human message; generic exception -> descriptive message; empty response field; missing response key"),
            ("TestAnswer",              "4",  "Returns (answer, type) tuple; correct type detection; context in Ollama prompt; empty chunk list"),
        ],
        [46, 14, 120],
        font_size=8,
    )

    p.h3("How it is implemented")
    p.bullet(
        "Question type detection: tested with real regex (no mocking) because "
        "detect_question_type() is a pure function with no side effects."
    )
    p.bullet(
        "Ollama API: urllib.request.urlopen patched to return a MagicMock supporting "
        "context manager protocol (__enter__/__exit__), returning controlled bytes."
    )
    p.bullet(
        "_call_ollama errors: urllib.error.URLError and RuntimeError injected via "
        "side_effect to verify both error-handling branches."
    )
    p.bullet(
        "Prompt checks: import SYSTEM_PROMPT from qa_engine and assert `in prompt` "
        "(not equality), so wording changes don't break tests unless structure changes."
    )

    p.h3("Why this approach")
    p.body(
        "The question-type detector is the gate that selects answer format. A missed "
        "detection produces a structurally wrong response even when the LLM answer is "
        "correct. Testing all 14 YES_NO trigger words individually catches any missing "
        "branch in the regex alternation `(does|is|can|...)`. The _call_ollama error tests "
        "ensure the UI never shows a Python traceback: both URLError (Ollama not running) "
        "and unexpected exceptions return polished error strings."
    )

    # ── 10.4 Evaluator ────────────────────────────────────────────────────────
    p.add_page()
    p.h2("10.4  Evaluator  (63 tests, 100%)")

    p.h3("What is tested")
    p.table(
        ["Test Class", "Count", "Scope"],
        [
            ("TestNormalisation",       "9",  "Lowercase, punctuation removal, whitespace collapse, strip, number preservation, empty string, punctuation-only input; _tokenise"),
            ("TestExactMatch",          "3",  "Identical after normalisation; different strings; punctuation stripped before comparison"),
            ("TestTokenF1",             "5",  "Perfect overlap (1/1/1); zero overlap (0/0/0); partial overlap (known fractions); asymmetric lengths; empty prediction"),
            ("TestRouge",               "8",  "Via library; manual fallback on ImportError; manual ROUGE-1/2 with known overlap; no overlap -> 0; ROUGE-L partial/identical/zero"),
            ("TestNgrams",              "4",  "Unigrams, bigrams, empty list, n > length -> empty Counter"),
            ("TestLCS",                 "7",  "Identical, partial, no-common, empty first/second/both, single element"),
            ("TestCosine",              "6",  "Identical -> 1.0; orthogonal -> 0.0; opposite -> -1.0; zero vector a/b/both -> 0.0"),
            ("TestSemanticSimilarity",  "4",  "With mocked model; Jaccard fallback on exception; both-empty -> 1.0; no-overlap -> 0.0"),
            ("TestGetSTModel",          "1",  "Class-level cache: model loaded once, second call returns same instance"),
            ("TestVerdict",             "9",  "All 5 labels; boundary values at 0.85, 0.70, 0.50, 0.30 (both sides)"),
            ("TestEvaluateIntegration", "6",  "Returns EvalResult; all fields in [0,1]; formula verified; identical score >= 0.90; different score < 0.30; details dict"),
            ("TestEvalResult",          "1",  "Dataclass default values"),
        ],
        [46, 13, 121],
        font_size=8,
    )

    p.h3("How it is implemented")
    p.bullet(
        "Mathematical tests (_token_f1, _cosine, _lcs_length, _ngrams, _manual_rouge_n, "
        "_manual_rouge_l): hand-computed expected values with pytest.approx for floats."
    )
    p.bullet(
        "ROUGE library fallback: rouge_score injected into sys.modules as None via "
        "patch.dict, causing ImportError inside _rouge and triggering the manual path."
    )
    p.bullet(
        "Sentence-transformer: _get_st_model patched at the class level (not instance) "
        "so cls._st_model references are intercepted correctly."
    )
    p.bullet(
        "Formula verification: test recomputes expected overall_score from returned "
        "fields using the documented weights and asserts equality -- making weights "
        "testable documentation."
    )
    p.bullet(
        "Boundary values: each threshold tested at exactly x and x-0.01 to pin >= "
        "vs > operator."
    )

    p.h3("Why this approach")
    p.body(
        "The evaluator is the only module with pure mathematical logic that must be "
        "numerically correct. Testing with known inputs (e.g., _token_f1('a b c d', 'a b') "
        "-> precision=0.5, recall=1.0) documents the algorithm precisely and catches any "
        "accidental formula change. The formula-verification integration test is especially "
        "valuable: if a weight changes from 0.35 to 0.53, the test fails even though all "
        "individual metric tests pass."
    )

    # ── 10.5 Strategy ─────────────────────────────────────────────────────────
    p.add_page()
    p.h2("10.5  Testing Strategy")

    p.h3("Unit vs. Functional split")
    p.table(
        ["Level", "What it checks", "Where used"],
        [
            ("Unit",        "A single method with controlled inputs",     "Chunking maths, token F1, cosine, ROUGE, verdict thresholds"),
            ("Integration", "Full public API method, real data flow",     "process(), evaluate(), answer() end-to-end"),
        ],
        [28, 65, 87],
    )

    p.h3("Mocking philosophy")
    p.body(
        "All external dependencies are mocked so tests are fast, deterministic, and "
        "self-contained. No internet, no GPU, no database server required."
    )
    p.table(
        ["Dependency", "Mock technique", "Reason"],
        [
            ("ChromaDB client",          "patch(\"chromadb.Client\")",                    "Avoids DuckDB process start (~2 s)"),
            ("SentenceTransformer model","patch.object(cls, \"_get_st_model\")",           "Avoids 90 MB download + GPU init"),
            ("Ollama HTTP API",          "patch(\"urllib.request.urlopen\")",              "Avoids network; tests success and error paths"),
            ("pdfplumber / PyPDF2",      "patch(\"pdfplumber.open\", ...)",                "Avoids PDF binary parsing"),
            ("rouge_score library",      "patch.dict(\"sys.modules\", {\"rouge_score\": None})", "Tests ImportError fallback branch"),
        ],
        [48, 66, 66],
        font_size=8,
    )

    p.h3("Fixture design (conftest.py)")
    p.body(
        "Shared fixtures follow the principle of minimum viable state:"
    )
    p.bullet("sample_short_text / sample_long_text  --  control chunking predictably")
    p.bullet("tmp_txt_file / tmp_unsupported_file   --  real OS files with known extension")
    p.bullet("document_processor_small (chunk_size=10, overlap=2)  --  short strings, multi-chunk output")
    p.bullet("evaluator_instance  --  plain Evaluator() with no pre-mocked state")

    p.h3("Running the test suite")
    p.code(
        "# All 152 tests + terminal + HTML coverage report\n"
        "pytest\n\n"
        "# HTML report (open coverage_report/index.html)\n"
        "pytest --cov-report=html:coverage_report\n\n"
        "# Single module\n"
        "pytest tests/test_evaluator.py -v\n\n"
        "# Single test class\n"
        "pytest tests/test_qa_engine.py::TestDetectQuestionType -v"
    )

    # ── 10.6 Coverage ─────────────────────────────────────────────────────────
    p.h2("10.6  Coverage Results")
    p.code(
        "Name                    Stmts   Miss  Cover\n"
        "-------------------------------------------\n"
        "document_processor.py      62      0   100%\n"
        "evaluator.py              132      0   100%\n"
        "qa_engine.py               39      0   100%\n"
        "vector_store.py            24      0   100%\n"
        "-------------------------------------------\n"
        "TOTAL                     257      0   100%\n"
    )
    p.success_box(
        "100% overall coverage -- every executable statement in all four pipeline "
        "modules is exercised. Requirement: >= 95%. Achieved: 100%."
    )
    p.note(
        "The HTML coverage report is generated in coverage_report/ by running pytest. "
        "Open coverage_report/index.html for a line-by-line interactive breakdown."
    )


def section_troubleshooting(p):
    p.add_page()
    p.h1("11.  Troubleshooting")

    issues = [
        (
            "'Could not connect to Ollama'",
            "Make sure the Ollama server is running in a separate terminal:",
            "ollama serve\n# Should print: Listening on 127.0.0.1:11434"
        ),
        (
            "'model llama3.2:3b not found'",
            "Pull the model first:",
            "ollama pull llama3.2:3b\n# Wait for full download, then restart ollama serve"
        ),
        (
            "Slow first query / embedding",
            "The MiniLM model (~90 MB) downloads on first use and caches at "
            "~/.cache/huggingface/. Subsequent runs are instant.",
            None
        ),
        (
            "PDF shows empty text",
            "The PDF is likely scanned (image-only). Add a text layer first:",
            "pip install ocrmypdf\nocrmypdf input.pdf output.pdf\n# Then upload output.pdf"
        ),
        (
            "Streamlit port conflict",
            "Use a different port:",
            "streamlit run app.py --server.port 8502"
        ),
        (
            "Windows: torch install fails",
            "Use the CPU or CUDA wheel from pytorch.org/get-started/locally",
            None
        ),
    ]
    for title, desc, cmd in issues:
        p.h3(title)
        p.body(desc)
        if cmd:
            p.code(cmd)


def section_structure(p):
    p.add_page()
    p.h1("12.  Project Structure")
    p.code(
        "rag_assistant/\n"
        "|-  app.py                  # Streamlit UI (3-column: Upload | Q&A | Evaluate)\n"
        "|-  document_processor.py   # PDF/TXT extraction + sliding-window chunking\n"
        "|-  vector_store.py         # ChromaDB + MiniLM-L6-v2 embedding wrapper\n"
        "|-  qa_engine.py            # Ollama API caller + question-type-aware prompt\n"
        "|-  evaluator.py            # 8-metric evaluation engine (ROUGE, F1, Semantic)\n"
        "|-  requirements.txt        # All Python dependencies\n"
        "|-  pytest.ini              # Pytest configuration with coverage settings\n"
        "|-  tests/\n"
        "|   |-  __init__.py\n"
        "|   |-  conftest.py                 # Shared fixtures (sample data, temp files)\n"
        "|   |-  test_document_processor.py  # 24 tests -- extraction, cleaning, chunking\n"
        "|   |-  test_vector_store.py        # 11 tests -- ChromaDB wrapper operations\n"
        "|   |-  test_qa_engine.py           # 54 tests -- question detection, prompt, API\n"
        "|   |-  test_evaluator.py           # 63 tests -- metrics, scoring, verdicts\n"
        "|-  coverage_report/        # HTML coverage report (generated by pytest)\n"
        "|-  Report/                 # Assignment report (PDF, HTML, LaTeX, diagrams)\n"
        "\\-  README.md               # Full project documentation\n"
    )

    p.h2("pytest.ini configuration")
    p.code(
        "[pytest]\n"
        "testpaths = tests\n"
        "python_files = test_*.py\n"
        "python_classes = Test*\n"
        "python_functions = test_*\n"
        "addopts =\n"
        "    -v\n"
        "    --tb=short\n"
        "    --cov=document_processor\n"
        "    --cov=vector_store\n"
        "    --cov=qa_engine\n"
        "    --cov=evaluator\n"
        "    --cov-report=term-missing\n"
        "    --cov-report=html:coverage_report\n"
        "    --cov-fail-under=95\n"
    )


def section_deps(p):
    p.add_page()
    p.h1("13.  Dependencies")
    p.h2("Application (runtime)")
    p.table(
        ["Package", "Min Version", "Role"],
        [
            ("streamlit",             "1.35.0", "Web UI framework"),
            ("pdfplumber",            "0.11.0", "PDF text extraction (primary)"),
            ("PyPDF2",                "3.0.0",  "PDF text extraction (fallback)"),
            ("sentence-transformers", "2.7.0",  "all-MiniLM-L6-v2 embeddings"),
            ("torch",                 "2.2.0",  "Required by sentence-transformers"),
            ("chromadb",              "0.5.0",  "In-memory vector database with cosine indexing"),
            ("rouge-score",           "0.1.2",  "ROUGE-1 / ROUGE-2 / ROUGE-L metrics"),
            ("numpy",                 "1.26.0", "Numerical utilities"),
        ],
        [55, 28, 97],
    )
    p.h2("Testing")
    p.table(
        ["Package", "Min Version", "Role"],
        [
            ("pytest",       "8.0.0",  "Test discovery and execution"),
            ("pytest-cov",   "5.0.0",  "Coverage measurement and reporting"),
            ("pytest-mock",  "3.14.0", "Enhanced mock utilities (mocker fixture)"),
        ],
        [55, 28, 97],
    )
    p.h2("Runtime dependency (not pip)")
    p.body(
        "Ollama (https://ollama.com) must be installed and the llama3.2:3b model "
        "must be pulled (`ollama pull llama3.2:3b`). Ollama serves the LLM locally "
        "on port 11434 -- no internet required during inference."
    )
    p.h2("Project Links")
    p.kv("GitHub Repository", "https://github.com/naveensankar5905/simple-rag")
    p.kv("Video Demo",        "https://photos.app.goo.gl/ammFaEPwREACdEPh9")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = PDF()
    p.set_auto_page_break(auto=True, margin=18)
    p.set_margins(L, 18, 210 - L - W)

    p.cover_page()
    p.toc_page()
    section_overview(p)
    section_architecture(p)
    section_stack(p)
    section_modules(p)
    section_setup(p)
    section_usage(p)
    section_metrics(p)
    section_qtypes(p)
    section_config(p)
    section_tests(p)
    section_troubleshooting(p)
    section_structure(p)
    section_deps(p)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    p.output(str(OUT))
    print(f"PDF written -> {OUT}")
    print(f"Pages: {p.page}")


if __name__ == "__main__":
    main()