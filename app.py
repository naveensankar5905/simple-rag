import streamlit as st
from pathlib import Path
from document_processor import DocumentProcessor
from vector_store import VectorStore
from qa_engine import QAEngine
from evaluator import Evaluator

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – AI Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --bg:       #0d0f14;
    --surface:  #13161e;
    --border:   #1e2330;
    --accent:   #00e5c0;
    --accent2:  #7c6af7;
    --text:     #e8eaf2;
    --muted:    #6b7280;
    --warn:     #f59e0b;
    --danger:   #f87171;
    --green:    #34d399;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
.main .block-container { padding-top: 1.5rem; max-width: 1100px; }

.docmind-header {
    text-align: center;
    padding: 2rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.docmind-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.docmind-header p {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    margin-top: 0.3rem;
}

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.stTextInput > div > div > input,
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,192,0.12) !important;
}

.answer-box {
    background: var(--surface);
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-top: 0.8rem;
    font-size: 0.93rem;
    line-height: 1.75;
}
.gt-box {
    background: var(--surface);
    border: 1px solid var(--accent2);
    border-left: 4px solid var(--accent2);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-top: 0.5rem;
    font-size: 0.93rem;
    line-height: 1.75;
}

/* ── Evaluation matrix ── */
.eval-wrapper {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}
.eval-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    margin-bottom: 1rem;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.7rem;
    margin-bottom: 1rem;
}
.metric-cell {
    background: #0d0f14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 0.8rem;
    text-align: center;
}
.metric-cell .val {
    font-size: 1.35rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
}
.metric-cell .lbl {
    font-size: 0.62rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.15rem;
}
.verdict-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 0.8rem;
    background: #0d0f14;
    border-radius: 8px;
    border: 1px solid var(--border);
}
.verdict-label {
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    white-space: nowrap;
}
.verdict-text {
    font-size: 1rem;
    font-weight: 700;
}
.score-bar-bg {
    flex: 1;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}
.em-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    font-weight: 600;
}
.em-yes { background: rgba(52,211,153,0.15); color: var(--green); border: 1px solid rgba(52,211,153,0.3); }
.em-no  { background: rgba(248,113,113,0.12); color: var(--danger); border: 1px solid rgba(248,113,113,0.25); }

/* ── Q-type badge ── */
.qtype-badge {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 20px;
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: rgba(124,106,247,0.15);
    color: var(--accent2);
    border: 1px solid rgba(124,106,247,0.3);
    margin-left: 0.5rem;
}

.source-chunk {
    background: #0d0f14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-top: 0.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.76rem;
    color: var(--muted);
    line-height: 1.6;
}
.source-label {
    font-size: 0.62rem;
    color: var(--accent2);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.25rem;
    font-weight: 600;
}
.section-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.4rem;
}
.badge { display:inline-block; padding:0.18rem 0.55rem; border-radius:20px; font-size:0.68rem; font-family:'DM Mono',monospace; font-weight:500; }
.badge-green { background:rgba(0,229,192,0.12); color:var(--accent); border:1px solid rgba(0,229,192,0.25); }
.badge-purple { background:rgba(124,106,247,0.12); color:var(--accent2); border:1px solid rgba(124,106,247,0.25); }
.stat-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:0.9rem 1.1rem; text-align:center; }
.stat-card .num { font-size:1.5rem; font-weight:700; color:var(--accent); }
.stat-card .label { font-size:0.65rem; color:var(--muted); font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:1px; }
.history-item { border-bottom:1px solid var(--border); padding:0.55rem 0; font-size:0.82rem; }
.history-item:last-child { border-bottom:none; }
.history-q { color:var(--accent); font-weight:600; }
.history-a { color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:0.15rem; }
[data-testid="stExpander"] { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:8px !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "vector_store": None,
    "doc_stats": None,
    "qa_history": [],
    "doc_name": None,
    "last_answer": None,
    "last_qtype": None,
    "last_sources": None,
    "last_question": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Cached singletons ──────────────────────────────────────────────────────────
@st.cache_resource
def get_processor():
    return DocumentProcessor(chunk_size=500, chunk_overlap=50)

@st.cache_resource
def get_qa_engine():
    return QAEngine(model_name="llama3.2:3b")

@st.cache_resource
def get_evaluator():
    return Evaluator()

# ── Helpers ────────────────────────────────────────────────────────────────────
QTYPE_EMOJI = {
    "WHAT": "❓", "WHEN": "📅", "WHERE": "📍", "WHO": "👤",
    "WHY": "🔍", "HOW": "⚙️", "YES_NO": "✅", "LIST": "📋",
    "COMPARE": "⚖️", "DEFINE": "📖", "SUMMARY": "📝",
}

def score_color(s: float) -> str:
    if s >= 0.85: return "#34d399"
    if s >= 0.70: return "#fbbf24"
    if s >= 0.50: return "#f97316"
    return "#f87171"

def pct(v: float) -> str:
    return f"{v*100:.1f}%"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="docmind-header">
    <h1>🧠 DocMind</h1>
    <p>RAG · llama3.2:3b · ChromaDB · Evaluation Matrix</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 3)
    st.markdown("---")
    st.markdown("### 📂 Document")
    if st.session_state.doc_name:
        st.markdown(f'<span class="badge badge-green">✓ {st.session_state.doc_name}</span>', unsafe_allow_html=True)
        s = st.session_state.doc_stats
        if s:
            st.markdown(f"""
            <div style="margin-top:0.7rem">
              <div class="stat-card" style="margin-bottom:0.4rem">
                <div class="num">{s['chunks']}</div><div class="label">Chunks indexed</div>
              </div>
              <div class="stat-card">
                <div class="num">{s['chars']:,}</div><div class="label">Characters</div>
              </div>
            </div>""", unsafe_allow_html=True)
        if st.button("🗑 Clear document"):
            for k in ["vector_store","doc_stats","qa_history","doc_name",
                      "last_answer","last_qtype","last_sources","last_question"]:
                st.session_state[k] = [] if k == "qa_history" else None
            st.rerun()
    else:
        st.markdown('<span class="badge badge-purple">⊘ No document loaded</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.qa_history:
        st.markdown("### 🕘 History")
        for item in reversed(st.session_state.qa_history[-5:]):
            st.markdown(f"""
            <div class="history-item">
              <div class="history-q">Q: {item['question'][:55]}{'…' if len(item['question'])>55 else ''}</div>
              <div class="history-a">{item['answer'][:70]}…</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT  — three columns: Upload | Q&A | Evaluation
# ══════════════════════════════════════════════════════════════════════════════
col_up, col_qa, col_eval = st.columns([0.9, 1.2, 1.1], gap="large")

# ── UPLOAD ────────────────────────────────────────────────────────────────────
with col_up:
    st.markdown('<div class="section-label">Step 1 — Upload document</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("PDF or TXT", type=["pdf","txt"], label_visibility="collapsed")

    if uploaded and uploaded.name != st.session_state.doc_name:
        processor = get_processor()
        tmp = Path("/tmp") / uploaded.name
        tmp.write_bytes(uploaded.read())
        with st.spinner("Extracting & chunking…"):
            chunks, full_text = processor.process(str(tmp))
        with st.spinner(f"Embedding {len(chunks)} chunks…"):
            vs = VectorStore()
            vs.add_documents(chunks)
        st.session_state.vector_store = vs
        st.session_state.doc_name = uploaded.name
        st.session_state.doc_stats = {"chunks": len(chunks), "chars": len(full_text)}
        st.session_state.qa_history = []
        for k in ["last_answer","last_qtype","last_sources","last_question"]:
            st.session_state[k] = None
        st.success(f"✓ Indexed **{len(chunks)} chunks**")
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-label">Pipeline</div>', unsafe_allow_html=True)
    for icon, label in [
        ("📄","Upload PDF / TXT"), ("✂️","Chunk (500w / 50w overlap)"),
        ("🔢","MiniLM-L6-v2 embeddings"), ("🗃️","ChromaDB vector store"),
        ("🤖","llama3.2:3b answers"), ("📊","Evaluation matrix"),
    ]:
        st.markdown(f"`{icon}` {label}")

    st.markdown("---")
    st.markdown('<div class="section-label">Question types detected</div>', unsafe_allow_html=True)
    for qtype, emoji in QTYPE_EMOJI.items():
        st.markdown(f"{emoji} `{qtype}`")

# ── Q&A ───────────────────────────────────────────────────────────────────────
with col_qa:
    st.markdown('<div class="section-label">Step 2 — Ask a question</div>', unsafe_allow_html=True)

    question = st.text_input(
        "Question",
        placeholder="e.g. What are the main findings?",
        label_visibility="collapsed",
        disabled=st.session_state.vector_store is None,
    )
    ask_btn = st.button(
        "Ask →",
        disabled=st.session_state.vector_store is None or not question.strip(),
    )

    if st.session_state.vector_store is None:
        st.info("⬅️  Upload a document first.")

    if ask_btn and question.strip() and st.session_state.vector_store:
        qa  = get_qa_engine()
        vs  = st.session_state.vector_store

        with st.spinner("Retrieving context…"):
            results = vs.query(question, n_results=top_k)

        with st.spinner("Generating answer…"):
            answer, qtype = qa.answer(question, results["documents"][0])

        st.session_state.last_answer   = answer
        st.session_state.last_qtype    = qtype
        st.session_state.last_sources  = results
        st.session_state.last_question = question
        st.session_state.qa_history.append({"question": question, "answer": answer})

    # Display answer
    if st.session_state.last_answer:
        qtype = st.session_state.last_qtype or "WHAT"
        emoji = QTYPE_EMOJI.get(qtype, "❓")
        st.markdown(
            f'<span style="color:var(--muted);font-size:0.72rem;font-family:DM Mono,monospace;">'
            f'Question type detected:</span>'
            f'<span class="qtype-badge">{emoji} {qtype}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div class="answer-box">
            <strong style="color:var(--accent);font-size:0.7rem;letter-spacing:1px;text-transform:uppercase;">
            AI Answer</strong><br><br>
            {st.session_state.last_answer}
        </div>""", unsafe_allow_html=True)

        if st.session_state.last_sources:
            res = st.session_state.last_sources
            with st.expander(f"📎 {len(res['documents'][0])} source chunks"):
                for i, chunk in enumerate(res["documents"][0]):
                    dist = res["distances"][0][i] if "distances" in res else None
                    score_str = f"  ·  sim {1-dist:.2f}" if dist is not None else ""
                    st.markdown(f'<div class="source-label">Chunk {i+1}{score_str}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="source-chunk">{chunk[:380]}{"…" if len(chunk)>380 else ""}</div>', unsafe_allow_html=True)

# ── EVALUATION ────────────────────────────────────────────────────────────────
with col_eval:
    st.markdown('<div class="section-label">Step 3 — Evaluation matrix</div>', unsafe_allow_html=True)

    gt_input = st.text_area(
        "Paste your ground-truth answer here",
        placeholder="Type or paste the correct / expected answer…",
        height=130,
        label_visibility="visible",
        disabled=st.session_state.last_answer is None,
    )

    eval_btn = st.button(
        "Evaluate ↗",
        disabled=(st.session_state.last_answer is None or not gt_input.strip()),
    )

    if st.session_state.last_answer is None:
        st.info("Ask a question first, then paste your ground truth here.")

    if eval_btn and gt_input.strip() and st.session_state.last_answer:
        evaluator = get_evaluator()
        with st.spinner("Computing metrics…"):
            result = evaluator.evaluate(
                prediction=st.session_state.last_answer,
                ground_truth=gt_input.strip(),
            )

        # ── Ground-truth echo ─────────────────────────────────
        st.markdown("""
        <div class="gt-box">
            <strong style="color:var(--accent2);font-size:0.7rem;letter-spacing:1px;text-transform:uppercase;">
            Your Ground Truth</strong><br><br>""" +
            gt_input.strip() + "</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metric grid ───────────────────────────────────────
        sc = score_color(result.overall_score)
        em_cls   = "em-yes" if result.exact_match else "em-no"
        em_label = "✓ Exact Match" if result.exact_match else "✗ No Exact Match"

        st.markdown(f"""
        <div class="eval-wrapper">
          <div class="eval-title">📊 Evaluation Metrics</div>

          <div class="metric-grid">
            <div class="metric-cell">
              <div class="val" style="color:{score_color(result.token_f1)}">{pct(result.token_f1)}</div>
              <div class="lbl">Token F1</div>
            </div>
            <div class="metric-cell">
              <div class="val" style="color:{score_color(result.rouge1)}">{pct(result.rouge1)}</div>
              <div class="lbl">ROUGE-1</div>
            </div>
            <div class="metric-cell">
              <div class="val" style="color:{score_color(result.rouge2)}">{pct(result.rouge2)}</div>
              <div class="lbl">ROUGE-2</div>
            </div>
            <div class="metric-cell">
              <div class="val" style="color:{score_color(result.rougeL)}">{pct(result.rougeL)}</div>
              <div class="lbl">ROUGE-L</div>
            </div>
            <div class="metric-cell">
              <div class="val" style="color:{score_color(result.token_precision)}">{pct(result.token_precision)}</div>
              <div class="lbl">Precision</div>
            </div>
            <div class="metric-cell">
              <div class="val" style="color:{score_color(result.token_recall)}">{pct(result.token_recall)}</div>
              <div class="lbl">Recall</div>
            </div>
            <div class="metric-cell">
              <div class="val" style="color:{score_color(result.semantic_similarity)}">{pct(result.semantic_similarity)}</div>
              <div class="lbl">Semantic Sim</div>
            </div>
            <div class="metric-cell">
              <div class="val" style="color:{sc}">{pct(result.overall_score)}</div>
              <div class="lbl">Overall</div>
            </div>
          </div>

          <div class="verdict-row">
            <div class="verdict-label">Verdict</div>
            <div class="verdict-text">{result.verdict}</div>
            <div class="score-bar-bg">
              <div class="score-bar-fill"
                   style="width:{result.overall_score*100:.1f}%;background:{sc}"></div>
            </div>
            <span class="em-badge {em_cls}">{em_label}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric explanation ─────────────────────────────────
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
- 🟢 Excellent ≥ 85% — answer is nearly perfect
- 🟡 Good ≥ 70% — answer covers most key content  
- 🟠 Partial ≥ 50% — partially correct, missing detail
- 🔴 Weak ≥ 30% — significant gaps
- ⛔ Poor < 30% — answer does not match ground truth
            """)
