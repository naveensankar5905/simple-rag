import os
import sys
import time
import logging
import warnings

# ── Must be set BEFORE transformers is imported anywhere ──────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"

# ── Fix Windows asyncio "ConnectionResetError [WinError 10054]" ───────────────
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_lost = _ProactorBasePipeTransport._call_connection_lost
        def _patched_lost(self, exc):
            try:
                _orig_lost(self, exc)
            except ConnectionResetError:
                pass
        _ProactorBasePipeTransport._call_connection_lost = _patched_lost
    except Exception:
        pass

# ── Suppress noisy log output ─────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*__path__.*")
warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("easyocr").setLevel(logging.ERROR)

import streamlit as st
from pathlib import Path

from multimodal_processor import MultimodalProcessor, MODALITY_ICON, MODALITY_MODEL
from vector_store import VectorStore
from qa_engine import QAEngine
from web_search import WebSearcher
from agent import RAGAgent, SOURCE_KB, SOURCE_WEB, SOURCE_BOTH
from evaluator import Evaluator

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusRAG – Agentic Multimodal RAG",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Space Grotesk + JetBrains Mono, deep navy + cyan + violet
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --surface2:  #1a2234;
    --border:    #1f2a3f;
    --accent:    #00d4aa;
    --accent2:   #9b7ff0;
    --accent3:   #f472b6;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --warn:      #fbbf24;
    --danger:    #fb7185;
    --green:     #34d399;
    --blue:      #60a5fa;
    --radius:    10px;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
.main .block-container { padding-top: 1.2rem; max-width: 960px; }

/* ── Header ──────────────────────────────────────────────── */
.nexus-header {
    text-align: center;
    padding: 1.8rem 0 1rem;
    margin-bottom: 0.8rem;
}
.nexus-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, var(--accent), var(--accent2), var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(0,212,170,0.15);
}
.nexus-header p {
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
}

/* ── File upload area ────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
.upload-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
}

/* ── File chips ──────────────────────────────────────────── */
.file-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.6rem 0 1rem;
}
.file-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
}
.file-chip .chip-size {
    color: var(--muted);
    font-size: 0.6rem;
}

/* ── Chat messages ───────────────────────────────────────── */
.chat-container { margin-top: 0.5rem; }
.chat-q {
    background: linear-gradient(135deg, rgba(155,127,240,0.12), rgba(244,114,182,0.08));
    border: 1px solid rgba(155,127,240,0.2);
    border-radius: var(--radius);
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
    color: var(--text);
}
.chat-q::before {
    content: '🔮';
    margin-right: 0.5rem;
}
.chat-a {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-bottom: 0.3rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Answer metadata row ─────────────────────────────────── */
.answer-meta {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 1.2rem;
    padding: 0.5rem 0;
    margin-bottom: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}
.meta-item {
    display: flex;
    align-items: center;
    gap: 0.35rem;
}
.meta-label {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 0.6rem;
}

/* ── Source type badges ───────────────────────────────────── */
.src-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
}
.src-kb   { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.3); color: var(--green); }
.src-web  { background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.3); color: var(--warn); }
.src-both { background: rgba(155,127,240,0.12); border: 1px solid rgba(155,127,240,0.3); color: var(--accent2); }

/* ── Confidence badges ───────────────────────────────────── */
.conf-high   { color: var(--green); }
.conf-medium { color: var(--warn); }
.conf-low    { color: var(--danger); }

/* ── Latency ─────────────────────────────────────────────── */
.latency { color: var(--muted); }

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0a0e1a !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.3rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Inputs ──────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.1) !important;
}

/* ── Expanders ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Source / chunk cards ────────────────────────────────── */
.chunk-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-top: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    color: var(--muted);
    line-height: 1.55;
}
.chunk-label {
    font-size: 0.6rem;
    color: var(--accent2);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.2rem;
    font-weight: 600;
}
.web-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-top: 0.35rem;
}
.web-title { font-size: 0.8rem; font-weight: 600; color: var(--accent); }
.web-url   { font-size: 0.65rem; color: var(--muted); font-family: 'JetBrains Mono', monospace; margin: 0.1rem 0; }
.web-snip  { font-size: 0.75rem; color: var(--text); line-height: 1.5; }

/* ── Agent trace ─────────────────────────────────────────── */
.trace-step {
    padding: 0.3rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
}
.trace-step:last-child { border-bottom: none; }

/* ── Modality badges ─────────────────────────────────────── */
.badge { display:inline-block; padding:0.15rem 0.5rem; border-radius:20px; font-size:0.65rem; font-family:'JetBrains Mono',monospace; font-weight:500; }
.badge-green  { background:rgba(0,212,170,0.1);  color:var(--accent);  border:1px solid rgba(0,212,170,0.2); }
.badge-purple { background:rgba(155,127,240,0.1);color:var(--accent2); border:1px solid rgba(155,127,240,0.2); }
.badge-blue   { background:rgba(96,165,250,0.1); color:var(--blue);    border:1px solid rgba(96,165,250,0.2); }
.badge-orange { background:rgba(251,191,36,0.1); color:var(--warn);    border:1px solid rgba(251,191,36,0.2); }
.badge-red    { background:rgba(251,113,133,0.1);color:var(--danger);  border:1px solid rgba(251,113,133,0.2); }
.badge-gray   { background:rgba(100,116,139,0.1);color:var(--muted);   border:1px solid rgba(100,116,139,0.2); }

/* ── Evaluation ──────────────────────────────────────────── */
.eval-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    margin-top: 0.8rem;
}
.eval-title {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.8rem;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin-bottom: 0.8rem;
}
.metric-cell {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.7rem;
    text-align: center;
}
.metric-cell .val {
    font-size: 1.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.metric-cell .lbl {
    font-size: 0.58rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.1rem;
}
.verdict-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0.7rem;
    background: var(--bg);
    border-radius: 8px;
    border: 1px solid var(--border);
}
.verdict-label {
    font-size: 0.65rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    white-space: nowrap;
}
.score-bar-bg { flex: 1; height: 7px; background: var(--border); border-radius: 4px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
.em-badge { display:inline-block; padding:0.12rem 0.5rem; border-radius:20px; font-size:0.6rem; font-family:'JetBrains Mono',monospace; font-weight:600; }
.em-yes { background:rgba(52,211,153,0.12); color:var(--green); border:1px solid rgba(52,211,153,0.25); }
.em-no  { background:rgba(251,113,133,0.1); color:var(--danger); border:1px solid rgba(251,113,133,0.2); }

/* ── GT box ──────────────────────────────────────────────── */
.gt-box {
    background: var(--surface);
    border: 1px solid var(--accent2);
    border-left: 3px solid var(--accent2);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    margin-top: 0.5rem;
    font-size: 0.88rem;
    line-height: 1.7;
}

/* ── Sidebar stats ───────────────────────────────────────── */
.stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    text-align: center;
    margin-bottom: 0.35rem;
}
.stat-card .num { font-size: 1.3rem; font-weight: 700; color: var(--accent); font-family: 'JetBrains Mono', monospace; }
.stat-card .label { font-size: 0.6rem; color: var(--muted); font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 1px; }

.kb-file-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.3rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.78rem;
}
.kb-file-row:last-child { border-bottom: none; }

/* ── Misc ────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }
.section-sep { border-top: 1px solid var(--border); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "vector_store":        None,
    "uploaded_file_names": set(),
    "kb_files":            [],
    "chat_history":        [],       # [{role, content, ...}]
    "last_result":         None,
    "last_question":       None,
    "uploader_key":        0,
    "web_fallback":        True,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Cached singletons ─────────────────────────────────────────────────────────
@st.cache_resource
def get_processor():
    return MultimodalProcessor(chunk_size=500, chunk_overlap=50)

@st.cache_resource
def get_qa_engine():
    return QAEngine(model_name="llama3.2:3b")

@st.cache_resource
def get_reranker():
    """Load BAAI/bge-reranker-base CrossEncoder. Returns None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    except Exception:
        return None

@st.cache_resource
def get_web_searcher():
    return WebSearcher()

@st.cache_resource
def get_evaluator():
    return Evaluator()

# ── Helper maps ────────────────────────────────────────────────────────────────
MODALITY_BADGE = {
    "pdf":   ("📄", "badge-green"),
    "text":  ("📝", "badge-gray"),
    "image": ("🖼️", "badge-blue"),
    "audio": ("🎵", "badge-orange"),
    "video": ("🎬", "badge-red"),
}

SOURCE_META = {
    SOURCE_KB:   ("🗄️", "src-kb",   "Knowledge Base"),
    SOURCE_WEB:  ("🌐", "src-web",  "Internet"),
    SOURCE_BOTH: ("🔀", "src-both", "Knowledge Base + Internet"),
}

CONF_CSS = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}

def score_color(s: float) -> str:
    if s >= 0.85: return "#34d399"
    if s >= 0.70: return "#fbbf24"
    if s >= 0.50: return "#f97316"
    return "#fb7185"

def pct(v: float) -> str:
    return f"{v*100:.1f}%"

def fmt_size(b: int) -> str:
    if b < 1024: return f"{b}B"
    if b < 1024*1024: return f"{b/1024:.1f}KB"
    return f"{b/(1024*1024):.1f}MB"

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔮 NexusRAG")
    st.markdown(
        '<span style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:var(--muted)">'
        'PDF · Image · Audio · Video → grounded, cited answers, with web fallback.'
        '</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Models section ─────────────────────────────────────────────────────────
    st.markdown("### Models")
    for label, val in [
        ("LLM",        "qwen2.5:3b"),
        ("Embeddings", "nomic-embed-text"),
        ("Whisper",    "base"),
        ("Reranker",   "on" if get_reranker() else "off"),
    ]:
        st.markdown(
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:var(--muted)">'
            f'{label}:</span> '
            f'<code style="font-size:0.7rem;color:var(--text)">{val}</code>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Web fallback toggle ────────────────────────────────────────────────────
    web_toggle = st.toggle(
        "🌐 Web fallback (Tavily)",
        value=st.session_state.web_fallback,
        key="web_toggle",
    )
    st.session_state.web_fallback = web_toggle

    st.markdown("---")

    # ── Tavily API key ─────────────────────────────────────────────────────────
    st.markdown("**🔑 Tavily API Key** *(optional)*")
    tavily_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-…  (leave blank for DuckDuckGo)",
        label_visibility="collapsed",
    )
    if tavily_key.strip():
        os.environ["TAVILY_API_KEY"] = tavily_key.strip()
    elif "TAVILY_API_KEY" not in os.environ:
        os.environ.pop("TAVILY_API_KEY", None)

    st.markdown("---")

    # ── Knowledge Base info ────────────────────────────────────────────────────
    if st.session_state.kb_files:
        total_chunks = sum(f["chunks"] for f in st.session_state.kb_files)
        st.markdown("**Chunks indexed**")
        st.markdown(f"""
        <div class="stat-card">
            <div class="num">{total_chunks}</div>
            <div class="label">Total chunks</div>
        </div>""", unsafe_allow_html=True)

        with st.expander(f"▸ Sources ({len(st.session_state.kb_files)})"):
            for f in st.session_state.kb_files:
                icon, badge_cls = MODALITY_BADGE.get(f["modality"], ("📄", "badge-gray"))
                st.markdown(
                    f'<div class="kb-file-row">'
                    f'<span class="badge {badge_cls}">{icon} {f["modality"].upper()}</span>'
                    f'<span style="font-size:0.76rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                    f'{f["name"][:28]}{"…" if len(f["name"])>28 else ""}'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if st.button("🗑 Clear knowledge base & chat"):
            for k in ["vector_store", "uploaded_file_names", "kb_files",
                      "chat_history", "last_result", "last_question"]:
                st.session_state[k] = (
                    set() if k == "uploaded_file_names" else
                    []    if k in ("kb_files", "chat_history") else
                    None
                )
            st.session_state["uploader_key"] += 1
            st.rerun()
    else:
        st.markdown(
            '<span class="badge badge-purple">⊘ No files uploaded</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Pipeline description ───────────────────────────────────────────────────
    st.markdown("### 🤖 Pipeline")
    for step in [
        ("1️⃣", "Query KB — ChromaDB + BM25 RRF"),
        ("2️⃣", "Evaluate cosine distance (θ = 0.92)"),
        ("3a", "✅ Sufficient → rerank → generate"),
        ("3b", "⚠️ Insufficient → web search"),
        ("3c", "🔀 Rerank with bge-reranker-base"),
        ("4️⃣", "llama3.2:3b generates → cite source"),
    ]:
        st.markdown(f"`{step[0]}` {step[1]}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nexus-header">
    <h1>🔮 NexusRAG</h1>
    <p>Ask your multimodal knowledge base</p>
</div>
""", unsafe_allow_html=True)

# ── File upload ────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-label">Upload documents, images, audio, or video</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload files",
    type=["pdf", "txt", "docx", "doc",
          "jpg", "jpeg", "png",
          "mp3", "wav", "m4a", "flac", "ogg",
          "mp4", "avi", "mkv", "mov", "webm"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    key=f"uploader_{st.session_state.uploader_key}",
)

processor = get_processor()

# ── Detect removed files and clean up their data ──────────────────────────────
current_upload_names = {f.name for f in (uploaded_files or [])}
previously_indexed   = set(st.session_state.uploaded_file_names)
removed_files        = previously_indexed - current_upload_names

if removed_files and st.session_state.vector_store is not None:
    for removed_name in removed_files:
        # Delete chunks from the vector store (ChromaDB + BM25 + local dict)
        st.session_state.vector_store.delete_by_source(removed_name)
        st.session_state.uploaded_file_names.discard(removed_name)
        # Remove from the KB file list
        st.session_state.kb_files = [
            f for f in st.session_state.kb_files if f["name"] != removed_name
        ]
        st.toast(f"🗑️ Removed **{removed_name}** from knowledge base", icon="🗑️")

    # If vector store is now empty, reset it to None
    if st.session_state.vector_store.count() == 0:
        st.session_state.vector_store = None

# ── Process newly uploaded files ──────────────────────────────────────────────
for uploaded in (uploaded_files or []):
    if uploaded.name in st.session_state.uploaded_file_names:
        continue

    if st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStore()

    suffix = Path(uploaded.name).suffix.lower()
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        _modality_hint = {
            ".pdf": "pdf", ".txt": "text", ".docx": "text", ".doc": "text",
            ".jpg": "image", ".jpeg": "image", ".png": "image",
            ".bmp": "image", ".tiff": "image", ".webp": "image",
            ".mp3": "audio", ".wav": "audio", ".m4a": "audio",
            ".flac": "audio", ".ogg": "audio", ".opus": "audio",
            ".mp4": "video", ".avi": "video", ".mkv": "video",
            ".mov": "video", ".webm": "video",
        }.get(suffix, "pdf")
        _model_label = MODALITY_MODEL.get(_modality_hint, "llama3.2:3b")

        with st.spinner(f"Processing **{uploaded.name}** using `{_model_label}`…"):
            chunks, full_text, modality = processor.process(tmp_path)
        if not chunks:
            raise RuntimeError(
                "No searchable text was extracted. For images, install/use "
                "llama3.2-vision or RapidOCR; for videos, make sure the video "
                "has speech/audio or readable frames."
            )
        with st.spinner(f"Embedding {len(chunks)} chunks via nomic-embed-text…"):
            st.session_state.vector_store.add_documents(
                chunks,
                source_file=uploaded.name,
                modality=modality,
            )
        st.session_state.kb_files.append({
            "name":     uploaded.name,
            "modality": modality,
            "chunks":   len(chunks),
            "chars":    len(full_text),
            "model":    _model_label,
            "size":     uploaded.size,
        })
        st.session_state.uploaded_file_names.add(uploaded.name)
        icon, _ = MODALITY_BADGE.get(modality, ("📄", ""))
        st.toast(f"{icon} Indexed **{uploaded.name}** ({len(chunks)} chunks)", icon="✅")
    except Exception as exc:
        st.error(f"❌ Failed to process {uploaded.name}: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

# ── File chips display ─────────────────────────────────────────────────────────
if st.session_state.kb_files:
    chips_html = '<div class="file-chips">'
    for f in st.session_state.kb_files:
        icon, badge_cls = MODALITY_BADGE.get(f["modality"], ("📄", "badge-gray"))
        size_str = fmt_size(f.get("size", 0))
        chips_html += (
            f'<div class="file-chip">'
            f'<span class="badge {badge_cls}" style="padding:0.1rem 0.35rem;font-size:0.6rem">{icon}</span>'
            f'{f["name"][:22]}{"…" if len(f["name"])>22 else ""}'
            f'<span class="chip-size">{size_str}</span>'
            f'</div>'
        )
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT HISTORY — render previous messages
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-q">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        result = msg.get("result")
        if result:
            # Source + Confidence + Latency row
            src_icon, src_css, src_label = SOURCE_META.get(result.source, ("❓", "src-kb", "Unknown"))
            conf_css = CONF_CSS.get(result.confidence, "conf-medium")
            st.markdown(f"""
            <div class="chat-a">{result.answer}</div>
            <div class="answer-meta">
                <div class="meta-item">
                    <span class="meta-label">Source Type</span>
                    <span class="src-badge {src_css}">{src_icon} {src_label}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Confidence</span>
                    <span class="{conf_css}">{result.confidence}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Latency</span>
                    <span class="latency">{result.latency:.2f}s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Sources expander
            total_sources = len(result.kb_chunks or []) + len(result.web_results or [])
            if total_sources > 0:
                with st.expander(f"▸ Sources ({total_sources})"):
                    if result.kb_chunks:
                        for i, chunk in enumerate(result.kb_chunks):
                            dist = result.kb_distances[i] if i < len(result.kb_distances) else None
                            meta = result.kb_metadatas[i] if i < len(result.kb_metadatas) else {}
                            sim_str = f"  ·  sim {1-dist:.2f}" if dist is not None else ""
                            src_str = (meta or {}).get("source", "")
                            mod_str = (meta or {}).get("modality", "")
                            icon = MODALITY_ICON.get(mod_str, "📄")
                            label = f"Chunk {i+1}{sim_str}"
                            if src_str:
                                label += f"  ·  {icon} {src_str}"
                            st.markdown(f'<div class="chunk-label">{label}</div>', unsafe_allow_html=True)
                            st.markdown(
                                f'<div class="chunk-card">{chunk[:400]}{"…" if len(chunk)>400 else ""}</div>',
                                unsafe_allow_html=True,
                            )
                    if result.web_results:
                        for r in result.web_results:
                            st.markdown(f"""
                            <div class="web-card">
                                <div class="web-title">{r.get('title','')}</div>
                                <div class="web-url">{r.get('url','')}</div>
                                <div class="web-snip">{r.get('snippet','')[:300]}</div>
                            </div>""", unsafe_allow_html=True)

            # Agent trace expander
            if result.trace:
                with st.expander("▸ Agent trace"):
                    for step in result.trace:
                        st.markdown(f'<div class="trace-step">{step}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION INPUT
# ══════════════════════════════════════════════════════════════════════════════
question = st.chat_input("Ask a question…")

if question and question.strip():
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": question.strip()})

    qa       = get_qa_engine()
    web      = get_web_searcher()
    reranker = get_reranker()

    # If web fallback is disabled, pass a no-op web searcher
    if not st.session_state.web_fallback:
        class _NoOpWeb:
            def search(self, *a, **kw): return []
            def as_context(self, *a, **kw): return ""
        web = _NoOpWeb()

    agent = RAGAgent(
        vector_store=st.session_state.vector_store,
        qa_engine=qa,
        web_searcher=web,
        reranker=reranker,
    )

    with st.spinner("Agent thinking…"):
        result = agent.run(question.strip(), top_k=3)

    st.session_state.last_result   = result
    st.session_state.last_question = question.strip()
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.answer,
        "result": result,
    })
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION SECTION (below chat)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.last_result:
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    with st.expander("📊 Evaluation — compare against ground truth"):
        gt_input = st.text_area(
            "Paste your ground-truth answer here",
            placeholder="Type or paste the correct / expected answer…",
            height=100,
            label_visibility="visible",
        )

        eval_btn = st.button("Evaluate ↗", disabled=not gt_input.strip())

        if eval_btn and gt_input.strip():
            evaluator = get_evaluator()
            pred = st.session_state.last_result.answer
            with st.spinner("Computing metrics…"):
                result_eval = evaluator.evaluate(
                    prediction=pred,
                    ground_truth=gt_input.strip(),
                )

            st.markdown("""
            <div class="gt-box">
                <strong style="color:var(--accent2);font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;">
                Your Ground Truth</strong><br><br>""" +
                gt_input.strip() + "</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            sc = score_color(result_eval.overall_score)
            em_cls   = "em-yes" if result_eval.exact_match else "em-no"
            em_label = "✓ Exact Match" if result_eval.exact_match else "✗ No Exact Match"

            st.markdown(f"""
            <div class="eval-box">
              <div class="eval-title">📊 Evaluation Metrics</div>
              <div class="metric-grid">
                <div class="metric-cell">
                  <div class="val" style="color:{score_color(result_eval.token_f1)}">{pct(result_eval.token_f1)}</div>
                  <div class="lbl">Token F1</div>
                </div>
                <div class="metric-cell">
                  <div class="val" style="color:{score_color(result_eval.rouge1)}">{pct(result_eval.rouge1)}</div>
                  <div class="lbl">ROUGE-1</div>
                </div>
                <div class="metric-cell">
                  <div class="val" style="color:{score_color(result_eval.rouge2)}">{pct(result_eval.rouge2)}</div>
                  <div class="lbl">ROUGE-2</div>
                </div>
                <div class="metric-cell">
                  <div class="val" style="color:{score_color(result_eval.rougeL)}">{pct(result_eval.rougeL)}</div>
                  <div class="lbl">ROUGE-L</div>
                </div>
                <div class="metric-cell">
                  <div class="val" style="color:{score_color(result_eval.token_precision)}">{pct(result_eval.token_precision)}</div>
                  <div class="lbl">Precision</div>
                </div>
                <div class="metric-cell">
                  <div class="val" style="color:{score_color(result_eval.token_recall)}">{pct(result_eval.token_recall)}</div>
                  <div class="lbl">Recall</div>
                </div>
                <div class="metric-cell">
                  <div class="val" style="color:{score_color(result_eval.semantic_similarity)}">{pct(result_eval.semantic_similarity)}</div>
                  <div class="lbl">Semantic Sim</div>
                </div>
                <div class="metric-cell">
                  <div class="val" style="color:{sc}">{pct(result_eval.overall_score)}</div>
                  <div class="lbl">Overall</div>
                </div>
              </div>
              <div class="verdict-row">
                <div class="verdict-label">Verdict</div>
                <div>{result_eval.verdict}</div>
                <div class="score-bar-bg">
                  <div class="score-bar-fill"
                       style="width:{result_eval.overall_score*100:.1f}%;background:{sc}"></div>
                </div>
                <span class="em-badge {em_cls}">{em_label}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("ℹ️ How metrics are computed"):
                st.markdown("""
| Metric | What it measures | Weight |
|---|---|---|
| **Token F1** | Word-overlap harmonic mean of precision & recall | 25% |
| **ROUGE-1** | Unigram overlap (F-measure) | 15% |
| **ROUGE-2** | Bigram overlap (F-measure) | — |
| **ROUGE-L** | Longest common subsequence overlap | 20% |
| **Semantic Similarity** | Cosine of MiniLM-L6-v2 sentence embeddings | 35% |
| **Exact Match** | Normalised string equality | 5% bonus |
| **Overall Score** | Weighted composite of the above | — |

**Verdict thresholds:**
- 🟢 Excellent ≥ 85%
- 🟡 Good ≥ 70%
- 🟠 Partial ≥ 50%
- 🔴 Weak ≥ 30%
- ⛔ Poor < 30%
                """)

elif not st.session_state.chat_history:
    if not st.session_state.kb_files:
        st.info("⬆️ Upload files to build a knowledge base, or just ask — the agent will search the web.")
    else:
        st.info("Type a question below. The agent will search your KB first.")
