# 🧠 DocMind – AI Document Assistant (RAG + Evaluation Matrix)

A production-ready Retrieval-Augmented Generation (RAG) application that lets you upload
any PDF or TXT document, ask natural-language questions, and evaluate the AI's answers
against your own ground-truth using 8 NLP metrics.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 Document Upload | PDF (pdfplumber + PyPDF2 fallback) and TXT |
| ✂️ Smart Chunking | Sliding-window, 500 words / 50-word overlap |
| 🔢 Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| 🗃️ Vector Store | ChromaDB (in-memory, no server needed) |
| 🤖 LLM | `llama3.2:3b` via local Ollama REST API |
| 🎯 Question Types | Auto-detects WHAT / WHEN / HOW / WHERE / WHY / WHO / YES-NO / LIST / COMPARE / DEFINE / SUMMARY |
| 📊 Evaluation Matrix | Token F1, Precision, Recall, ROUGE-1/2/L, Semantic Similarity, Exact Match, Overall Score |
| 🖥️ UI | Streamlit — three-column layout (Upload → Q&A → Evaluate) |

---

## 🏗️ Architecture

```
┌──────────────┐    ┌────────────────────┐    ┌──────────────────────┐
│  User uploads│───▶│ DocumentProcessor  │───▶│     VectorStore      │
│  PDF / TXT   │    │  extract + chunk   │    │  ChromaDB +          │
└──────────────┘    └────────────────────┘    │  MiniLM-L6-v2 embeds │
                                              └──────────┬───────────┘
                                                         │ top-k chunks
┌──────────────┐    ┌────────────────────┐    ┌──────────▼───────────┐
│  Evaluation  │    │     QAEngine       │◀───│   Retrieval result   │
│  Matrix      │    │  llama3.2:3b via   │    └──────────────────────┘
│  (8 metrics) │    │  Ollama REST API   │
└──────┬───────┘    └────────┬───────────┘
       │                     │ answer
       └─────────────────────▼──────────────
                    Streamlit UI
                 (3-column layout)
```

---

## 📋 Prerequisites

| Requirement | Version | Check command |
|---|---|---|
| Python | ≥ 3.10 | `python --version` |
| pip | ≥ 23 | `pip --version` |
| Ollama | latest | `ollama --version` |
| llama3.2:3b model | — | `ollama list` |
| Git (optional) | any | `git --version` |

> **Hardware:** llama3.2:3b runs on CPU (slow) or GPU/Apple Silicon (fast).  
> Minimum 8 GB RAM recommended.

---

## 🚀 Setup Instructions

### Step 1 — Get the project files

Place all six files in a single folder, e.g. `rag_assistant/`:

```
rag_assistant/
├── app.py
├── document_processor.py
├── vector_store.py
├── qa_engine.py
├── evaluator.py
└── requirements.txt
```

### Step 2 — Create a Python virtual environment

```bash
# Navigate into the project folder
cd rag_assistant

# Create virtual environment
python3 -m venv .venv

# Activate it
# macOS / Linux:
source .venv/bin/activate

# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your terminal prompt.

### Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **This installs:** streamlit, pdfplumber, PyPDF2, sentence-transformers,
> torch, chromadb, rouge-score, numpy.

> **Note — Apple Silicon (M1/M2/M3):** PyTorch with MPS support installs automatically. No extra steps needed.

> **Note — NVIDIA GPU:** For CUDA acceleration, replace the `torch` line in
> `requirements.txt` with your CUDA build. Find the right command at
> [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Step 4 — Install Ollama

**macOS (Homebrew):**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**  
Download and run the installer from [ollama.com](https://ollama.com).

### Step 5 — Pull the llama3.2:3b model

Open a **new terminal** and run:

```bash
ollama pull llama3.2:3b
```

This downloads approximately **2 GB**. Wait for it to complete, then verify:

```bash
ollama list
# Should show: llama3.2:3b
```

### Step 6 — Start the Ollama server

Keep this terminal open while you use DocMind:

```bash
ollama serve
```

You should see: `Listening on 127.0.0.1:11434`

### Step 7 — Launch DocMind

In your **original terminal** (with the virtual environment active):

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 🖥️ How to Use

### 1. Upload a Document (left column)
- Click **Browse files** or drag-and-drop a PDF or TXT.
- DocMind extracts text, splits into chunks, embeds them with MiniLM, and stores in ChromaDB.
- The sidebar shows chunk count and character count once indexing is done.

### 2. Ask a Question (middle column)
- Type any question about your document and click **Ask →**.
- DocMind auto-detects the question type (WHAT, WHEN, HOW, etc.) and picks the matching answer format.
- The detected type is shown as a badge above the answer.
- Expand **source chunks** to see exactly which parts of the document were used.

### 3. Evaluate the Answer (right column)
- Paste your own **ground-truth / expected answer** into the text box.
- Click **Evaluate ↗** to compute all 8 metrics instantly.
- A colour-coded metric grid and verdict appear — no external API needed.

---

## 📊 Evaluation Metrics Explained

| Metric | How it works | Weight |
|---|---|---|
| **Token Precision** | Fraction of predicted words found in ground truth | — |
| **Token Recall** | Fraction of ground-truth words found in prediction | — |
| **Token F1** | Harmonic mean of precision and recall | 25% |
| **ROUGE-1** | Unigram overlap F-measure | 15% |
| **ROUGE-2** | Bigram overlap F-measure | — |
| **ROUGE-L** | Longest Common Subsequence F-measure | 20% |
| **Semantic Similarity** | Cosine distance between MiniLM sentence embeddings | 35% |
| **Exact Match** | Normalised string equality (punctuation & case removed) | 5% bonus |
| **Overall Score** | Weighted composite of all the above | — |

**Verdict thresholds:**

| Score | Verdict | Meaning |
|---|---|---|
| ≥ 85% | 🟢 Excellent | Answer is nearly perfect |
| ≥ 70% | 🟡 Good | Covers most key content |
| ≥ 50% | 🟠 Partial | Partially correct, missing detail |
| ≥ 30% | 🔴 Weak | Significant gaps |
| < 30% | ⛔ Poor | Answer does not match ground truth |

---

## 🎯 Question Types & Answer Formats

DocMind automatically detects what kind of question you asked and formats the answer accordingly:

| Type | Trigger words | Answer format |
|---|---|---|
| **WHAT** | What is / What does… | One-sentence identification + 2-4 supporting details |
| **WHEN** | When did / When was… | Exact date/range first, then context |
| **HOW** | How does / How to… | Numbered steps (procedure) or cause→effect (mechanism) |
| **WHERE** | Where is / Where was… | Location immediately + spatial context |
| **WHY** | Why did / Why is… | Reason in sentence 1 + evidence + consequence chain |
| **WHO** | Who is / Who was… | Name + title/affiliation from the document |
| **YES/NO** | Does / Is / Can / Will / Has… | Starts with YES — / NO — / PARTIALLY — |
| **LIST** | List / What are / Name all… | Bullet list of all relevant items + item count |
| **COMPARE** | Compare / Difference / Versus… | Side-by-side description with contrast language |
| **DEFINE** | Define / What does X mean… | One-line definition then elaboration |
| **SUMMARY** | Summarise / Overview / Describe… | 5-8 sentences of flowing prose |

---

## ⚙️ Configuration Reference

| Parameter | File | Default | Description |
|---|---|---|---|
| `chunk_size` | `app.py` | 500 words | Words per chunk |
| `chunk_overlap` | `app.py` | 50 words | Overlap between chunks |
| `top_k` | Sidebar slider | 3 | Chunks retrieved per query |
| `temperature` | `qa_engine.py` | 0.2 | LLM sampling temperature (lower = more factual) |
| `num_predict` | `qa_engine.py` | 768 | Max tokens generated in answer |
| `embedding_model` | `vector_store.py` | all-MiniLM-L6-v2 | Sentence-transformer model |
| `llm_model` | `qa_engine.py` | llama3.2:3b | Ollama model tag |

---

## 🔧 Troubleshooting

**"Could not connect to Ollama"**
```bash
# Make sure Ollama is running:
ollama serve
# In a separate terminal, then retry your question.
```

**"model 'llama3.2:3b' not found"**
```bash
ollama pull llama3.2:3b
# Wait for full download, then restart ollama serve.
```

**Slow first query / embedding**
> The MiniLM model (~90 MB) downloads on first use and caches at
> `~/.cache/huggingface/`. Subsequent runs are instant.

**PDF shows empty text**
> The PDF is likely scanned (image-only). Use `ocrmypdf` to add a text layer:
> ```bash
> pip install ocrmypdf
> ocrmypdf input.pdf output.pdf
> ```
> Then upload `output.pdf` to DocMind.

**Streamlit port already in use**
```bash
streamlit run app.py --server.port 8502
```

**Windows: `torch` install fails**
> Use the CUDA or CPU wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 📁 Project Structure

```
rag_assistant/
├── app.py                  # Streamlit UI (3-column: Upload | Q&A | Evaluate)
├── document_processor.py   # PDF/TXT extraction + sliding-window chunking
├── vector_store.py         # ChromaDB + MiniLM-L6-v2 embedding wrapper
├── qa_engine.py            # Ollama API caller + question-type-aware prompt
├── evaluator.py            # 8-metric evaluation engine (ROUGE, F1, Semantic)
├── requirements.txt        # All Python dependencies
├── pytest.ini              # Pytest configuration with coverage settings
├── tests/                  # Test suite (152 test cases)
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures (sample data, temp files)
│   ├── test_document_processor.py  # 24 tests — extraction, cleaning, chunking
│   ├── test_vector_store.py        # 11 tests — ChromaDB wrapper operations
│   ├── test_qa_engine.py           # 54 tests — question detection, prompt, API
│   └── test_evaluator.py           # 63 tests — metrics, scoring, verdicts
├── coverage_report/        # HTML coverage report (generated by pytest)
├── Report/                 # Assignment report (PDF, HTML, LaTeX, diagrams)
└── README.md               # This file
```

---

## 📦 Dependencies

```
# ── Application ──
streamlit>=1.35.0           # Web UI
pdfplumber>=0.11.0          # PDF text extraction (primary)
PyPDF2>=3.0.0               # PDF text extraction (fallback)
sentence-transformers>=2.7.0 # MiniLM embeddings
torch>=2.2.0                # Required by sentence-transformers
chromadb>=0.5.0             # In-memory vector database
rouge-score>=0.1.2          # ROUGE-1 / ROUGE-2 / ROUGE-L metrics
numpy>=1.26.0               # Numerical utilities

# ── Testing ──
pytest>=8.0.0               # Test framework
pytest-cov>=5.0.0           # Coverage reporting plugin
pytest-mock>=3.14.0         # Enhanced mocking utilities
```

**Runtime dependency (not pip):**
- [Ollama](https://ollama.com) with `llama3.2:3b` pulled

---

## 🧪 Testing

### Overview

DocMind includes a comprehensive test suite built with **pytest** that validates all core modules of the RAG pipeline. The test suite uses **mocking** to isolate each module from heavyweight external dependencies (Ollama API, ChromaDB, sentence-transformers), making tests fast, reliable, and runnable without any GPU or network access.

### Test Structure

| Module | Test File | Tests | Coverage | What's Tested |
|---|---|---|---|---|
| `document_processor.py` | `test_document_processor.py` | 24 | 100% | Init, PDF/TXT extraction, pdfplumber→PyPDF2 fallback, text cleaning rules, sliding-window chunking, full pipeline integration |
| `vector_store.py` | `test_vector_store.py` | 11 | 100% | Client/collection init, add_documents (empty/single/multi), query (results/empty/capped), count |
| `qa_engine.py` | `test_qa_engine.py` | 54 | 100% | All 11 question types + edge cases (case, whitespace, boundary patterns), init, context formatting, prompt building (type-specific hints), Ollama API calls (success/URLError/generic error/empty/missing fields), answer integration |
| `evaluator.py` | `test_evaluator.py` | 63 | 100% | Normalisation, tokenisation, exact match, token F1, ROUGE (library + manual fallback), n-grams, LCS, cosine similarity, semantic similarity (model + Jaccard fallback), verdict thresholds (all 5 + boundary values), overall score formula, EvalResult dataclass |
| **TOTAL** | **4 files** | **152** | **100%** | — |

> **Note:** `app.py` (Streamlit UI) is excluded from automated testing because it requires an interactive browser session. It is tested manually.

### Installing Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### Running the Test Suite

**Run all tests with verbose output:**
```bash
pytest
```

This runs all 152 tests with coverage measurement (configured in `pytest.ini`).

**Run tests for a specific module:**
```bash
pytest tests/test_document_processor.py -v
pytest tests/test_evaluator.py -v
```

**Run a specific test class or function:**
```bash
pytest tests/test_qa_engine.py::TestDetectQuestionType -v
pytest tests/test_evaluator.py::TestCosine::test_cosine_identical_vectors -v
```

### Generating the Coverage Report

**Terminal report (default with pytest.ini):**
```bash
pytest --cov-report=term-missing
```

**HTML coverage report:**
```bash
pytest --cov-report=html:coverage_report
```
Then open `coverage_report/index.html` in your browser for an interactive, line-by-line coverage breakdown.

**Combined command (terminal + HTML):**
```bash
pytest --cov=document_processor --cov=vector_store --cov=qa_engine --cov=evaluator --cov-report=term-missing --cov-report=html:coverage_report
```

### Coverage Results

```
Name                    Stmts   Miss  Cover
-------------------------------------------
document_processor.py      62      0   100%
evaluator.py              132      0   100%
qa_engine.py               39      0   100%
vector_store.py            24      0   100%
-------------------------------------------
TOTAL                     257      0   100%
```

> **100% overall coverage** — exceeds the 95% requirement. Every executable statement in all four pipeline modules is exercised by the test suite.

---

## 🧾 Test Report — What, How & Why

This section provides a comprehensive explanation of the test suite design: what each group of tests covers, how they are implemented, and why each design decision was made.

---

### Module 1 — `document_processor.py` (24 tests, 100%)

#### DocumentProcessor — What is tested

| Class | Test Count | Scope |
|---|---|---|
| `TestDocumentProcessorInit` | 2 | `__init__` parameter storage (default and custom) |
| `TestTextExtraction` | 7 | `process()` for TXT; `_extract_pdf` via pdfplumber and PyPDF2 fallback; unsupported extension error; multi-page join; `None`-page skipping |
| `TestTextCleaning` | 5 | `_clean`: carriage-return normalisation, inline-space collapse, consecutive-newline limit, leading/trailing strip, combined rules |
| `TestChunking` | 7 | `_chunk`: empty input, single chunk, multiple chunks, overlap words, exact-boundary, no empty chunks, full word coverage |
| `TestDocumentProcessorIntegration` | 3 | End-to-end `process()` from file on disk through clean and chunk |

#### DocumentProcessor — How it is implemented

- **Temp files** (`tmp_path` fixture) are used for real file-system I/O — no fake path strings that could differ per OS.
- **PDF mocking**: `pdfplumber.open` is replaced with a `MagicMock` that exposes a `pages` list of mocks, each returning controlled `extract_text()` values. This lets tests exercise the page-iteration and `None`-page guard without a real PDF.
- **PyPDF2 fallback**: `pdfplumber.open` is patched to raise `ImportError`, then `PyPDF2.PdfReader` is patched to return a controlled mock — verifying that the except branch executes correctly.
- **Chunking edge cases**: A `document_processor_small` fixture (chunk\_size=10, chunk\_overlap=2) is used so that short synthetic strings reliably produce multiple chunks, making overlap assertions deterministic.

#### DocumentProcessor — Why this approach

The chunking logic is the most critical transformation in the pipeline — incorrect overlap or off-by-one errors here silently degrade retrieval quality downstream. Testing both the exact-boundary case (exactly `chunk_size` words → 1 chunk) and the overlap-word check (last N words of chunk 0 == first N words of chunk 1) catches the most common sliding-window bugs. Real file I/O with `tmp_path` is preferred over `mock_open` because it exercises the actual `Path.read_text` call and encoding handling, not just the string content.

---

### Module 2 — `vector_store.py` (11 tests, 100%)

#### VectorStore — What is tested

| Class | Test Count | Scope |
|---|---|---|
| `TestVectorStoreInit` | 2 | `chromadb.Client` call, `SentenceTransformerEmbeddingFunction` construction, `get_or_create_collection` with correct name and cosine metadata, class constants |
| `TestAddDocuments` | 3 | Empty list → no-op; multiple chunks stored with valid UUID ids; single chunk path |
| `TestQuery` | 4 | Result pass-through; empty collection guard (returns `{documents:[[]], distances:[[]]}` without calling ChromaDB); `n_results` capping to collection count; default `n_results=3` |
| `TestCount` | 2 | Delegation to `_col.count()`; empty collection returns 0 |

#### VectorStore — How it is implemented

- A helper `_make_mock_vs()` function constructs a `VectorStore` with both `chromadb.Client` and `SentenceTransformerEmbeddingFunction` patched inside a `with patch(...)` block, then returns the instance and the mock collection object.
- UUID validity is checked by calling `uuid.UUID(id_str)` — this raises `ValueError` on any malformed string, so the assertion is precise without hard-coding a specific UUID.
- The `n_results` cap is verified by setting `mock_col.count.return_value = 2` and requesting `n_results=10`, then asserting `mock_col.query` was called with `n_results=2`.

#### VectorStore — Why this approach

ChromaDB starts an in-memory DuckDB process and sentence-transformers downloads a ~90 MB model on first use. Mocking both eliminates a ~15-second test penalty and removes all network/GPU requirements. The UUID check is more robust than `len(id) == 36` because it validates the actual UUID format, catching any regression where IDs are generated by a different method.

---

### Module 3 — `qa_engine.py` (54 tests, 100%)

#### QAEngine — What is tested

| Class | Test Count | Scope |
|---|---|---|
| `TestDetectQuestionType` | 34 | All 11 question types; all 14 YES\_NO trigger words (does/is/can/will/has/did/was/were/should/would/could/do/have/are); COMPARE/SUMMARY/LIST/DEFINE patterns; case-insensitivity; leading-whitespace tolerance; default WHAT fallback |
| `TestQAEngineInit` | 3 | Default and custom model names; `OLLAMA_URL` constant |
| `TestFormatContext` | 3 | Multiple chunks labelled `[Chunk N]`; empty list returns empty string; single chunk |
| `TestBuildPrompt` | 5 | System prompt inclusion; question inclusion; context inclusion; type-specific hint for all 11 types; unknown type produces no crash |
| `TestCallOllama` | 5 | Successful JSON response; `URLError` → human-readable message; generic exception → descriptive message; empty `response` field; missing `response` key |
| `TestAnswer` | 4 | Returns `(answer, type)` tuple; correct type detection end-to-end; context passed to Ollama; empty chunk list |

#### QAEngine — How it is implemented

- **Question type detection** is tested with real regex (no mocking) because the detection function is pure Python with no side effects.
- **Ollama API**: `urllib.request.urlopen` is patched to return a `MagicMock` that supports the context manager protocol (`__enter__`/`__exit__`) and returns controlled `bytes` from `.read()`.
- **`_call_ollama` error paths**: `urllib.error.URLError` and a plain `RuntimeError` are injected via `side_effect` to verify both error-handling branches.
- **Prompt construction**: Tests import `SYSTEM_PROMPT` directly from `qa_engine` and check `in prompt` rather than exact equality, so prompt wording changes don't break tests unless the structural guarantee (system prompt present, question present, type hint present) is violated.

#### QAEngine — Why this approach

The question-type detector is the gate that selects the answer format — a missed or wrong detection produces a structurally wrong response even when the LLM answer is correct. Testing all 14 YES\_NO trigger words individually catches the regex alternation pattern `(does|is|can|...)` having a missing branch. The `_call_ollama` error tests ensure the UI never shows a Python traceback to the user: both `URLError` (Ollama not running) and unexpected exceptions (model crashed) return polished error strings.

---

### Module 4 — `evaluator.py` (63 tests, 100%)

#### Evaluator — What is tested

| Class | Test Count | Scope |
|---|---|---|
| `TestNormalisation` | 9 | Lowercase, punctuation removal, whitespace collapse, strip, number preservation, empty string, punctuation-only input; `_tokenise` on normal and empty string |
| `TestExactMatch` | 3 | Identical after normalisation; different strings; punctuation stripped before comparison |
| `TestTokenF1` | 5 | Perfect overlap (1.0/1.0/1.0); zero overlap (0/0/0); partial overlap (known fractions); asymmetric lengths (different precision vs recall); empty prediction |
| `TestRouge` | 8 | ROUGE via library; manual fallback on `ImportError`; manual ROUGE-1 unigram; manual ROUGE-2 bigram; no-overlap → 0; ROUGE-L partial LCS; ROUGE-L identical → 1.0; ROUGE-L no overlap → 0 |
| `TestNgrams` | 4 | Unigrams, bigrams, empty list, n > length |
| `TestLCS` | 7 | Identical, partial, no-common, empty first, empty second, both empty, single element |
| `TestCosine` | 6 | Identical vectors → 1.0; orthogonal → 0.0; opposite → -1.0; zero vector a, b, both |
| `TestSemanticSimilarity` | 4 | With mocked model (cosine of embeddings); Jaccard fallback on exception; both-empty Jaccard → 1.0; no-overlap Jaccard → 0.0 |
| `TestGetSTModel` | 1 | Class-level cache — model loaded once, second call returns same instance |
| `TestVerdict` | 9 | All 5 verdict labels; exact boundary values at 0.85, 0.70, 0.50, 0.30 |
| `TestEvaluateIntegration` | 6 | Returns `EvalResult`; all fields populated and in [0, 1]; weighted formula verified; identical texts score ≥ 0.90; different texts score < 0.30; `details` dict contains normalised texts |
| `TestEvalResult` | 1 | Dataclass default values |

#### Evaluator — How it is implemented

- **Mathematical tests** (`_token_f1`, `_cosine`, `_lcs_length`, `_ngrams`, `_manual_rouge_n`, `_manual_rouge_l`) use hand-computed expected values with `pytest.approx` for float comparisons. This pins the algorithm output to known-correct values.
- **ROUGE library fallback**: The `rouge_score` module is injected into `sys.modules` as `None` (via `patch.dict`), causing the `from rouge_score import ...` import inside `_rouge` to raise `ImportError` and fall into the manual path.
- **Sentence-transformer mocking**: `_get_st_model` is patched at the class level (not instance level) so that `cls._st_model` references are intercepted correctly.
- **Formula verification**: `test_evaluate_overall_score_formula` recomputes the expected score from the returned fields using the documented formula and asserts equality, making the weight constants testable documentation.
- **Boundary values**: Each verdict threshold (0.85, 0.70, 0.50, 0.30) is tested at both `x` and `x - 0.01` to pin the `>=` vs `>` operator.

#### Evaluator — Why this approach

The evaluator is the only module with pure mathematical logic that must be numerically correct. Testing with known inputs (e.g., `_token_f1("a b c d", "a b")` → precision=0.5, recall=1.0) documents the algorithm precisely and catches any accidental formula change. The formula-verification integration test is particularly valuable: if a weight is accidentally changed from 0.35 to 0.53, the test fails even though all sub-metric tests pass.

---

### Testing Strategy Summary

#### Unit vs. Functional split

| Test level | What it checks | Where used |
|---|---|---|
| **Unit** | A single method with controlled inputs | Chunking maths, token F1, cosine, ROUGE, verdict thresholds |
| **Integration/Functional** | Full public API method, real data flow | `process()`, `evaluate()`, `answer()` end-to-end |

#### Mocking philosophy

External dependencies are mocked to ensure tests are **fast, deterministic, and self-contained**:

| Dependency | Mock technique | Reason |
|---|---|---|
| ChromaDB client | `patch("chromadb.Client")` | Avoids DuckDB process start (~2 s) |
| SentenceTransformer model | `patch.object(cls, "_get_st_model")` | Avoids 90 MB download + GPU init |
| Ollama HTTP API | `patch("urllib.request.urlopen")` | Avoids network; tests both success and error paths |
| pdfplumber / PyPDF2 | `patch("pdfplumber.open", ...)` | Avoids PDF parsing of real binary files |
| rouge_score library | `patch.dict("sys.modules", ...)` | Tests the ImportError fallback branch |

#### Fixture design

Shared fixtures in `conftest.py` follow the principle of **minimum viable state**:
- `sample_short_text` / `sample_long_text` — control chunking behaviour predictably
- `tmp_txt_file` / `tmp_unsupported_file` — real OS files with known content and extensions
- `document_processor_small` (chunk\_size=10, overlap=2) — keeps synthetic test strings short while still exercising multi-chunk logic
- `evaluator_instance` — a plain `Evaluator()` with no pre-mocked state, so each test applies only the mocks it needs

#### How to run the full suite

```bash
# All 152 tests + coverage report (terminal + HTML)
pytest

# HTML report only
pytest --cov-report=html:coverage_report
# Then open coverage_report/index.html

# Single module
pytest tests/test_evaluator.py -v
```

---

## 📄 License

MIT — free to use, modify, and distribute.

