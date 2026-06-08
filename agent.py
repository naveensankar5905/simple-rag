"""
agent.py
────────
Agentic RAG pipeline orchestrated with LangGraph StateGraph.

Graph topology:
  retrieve_kb
      │
      ├─ sufficient  ──────────────→ rerank → generate
      │
      └─ insufficient / no_kb  → retrieve_web → rerank → generate

Reranker: BAAI/bge-reranker-base (cross-encoder, sentence-transformers).
Falls back gracefully if the model is not downloaded.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

SOURCE_KB   = "knowledge_base"
SOURCE_WEB  = "internet"
SOURCE_BOTH = "both"

_SUFFICIENCY_THRESHOLD = 0.92   # cosine distance below this -> KB is sufficient
_LOCAL_MODALITIES = {"pdf", "text", "image", "audio", "video"}


# ── Public result dataclass (same interface as before) ─────────────────────────

@dataclass
class AgentResult:
    answer:        str
    source:        str
    kb_chunks:     list[str]   = field(default_factory=list)
    kb_distances:  list[float] = field(default_factory=list)
    kb_metadatas:  list[dict]  = field(default_factory=list)
    web_results:   list[dict]  = field(default_factory=list)
    question_type: str         = "WHAT"
    reasoning:     str         = ""
    confidence:    str         = "Medium"
    trace:         list[str]   = field(default_factory=list)
    latency:       float       = 0.0


# ── LangGraph state schema ─────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:       str
    top_k:          int
    kb_chunks:      list[str]
    kb_distances:   list[float]
    kb_metadatas:   list[dict]
    web_results:    list[dict]
    context:        list[str]
    answer:         str
    source:         str
    question_type:  str
    reasoning:      str
    confidence:     str
    trace:          list[str]


# ── RAGAgent ───────────────────────────────────────────────────────────────────

class RAGAgent:
    """
    Wraps a compiled LangGraph pipeline.

    Parameters
    ----------
    vector_store : VectorStore | None
    qa_engine    : QAEngine
    web_searcher : WebSearcher
    reranker     : CrossEncoder | None   (BAAI/bge-reranker-base)
    """

    def __init__(self, vector_store, qa_engine, web_searcher, reranker=None):
        self.vs       = vector_store
        self.qa       = qa_engine
        self.web      = web_searcher
        self.reranker = reranker
        self._app     = self._build_graph()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, question: str, top_k: int = 3) -> AgentResult:
        t0 = time.perf_counter()
        state: AgentState = {
            "question":      question,
            "top_k":         top_k,
            "kb_chunks":     [],
            "kb_distances":  [],
            "kb_metadatas":  [],
            "web_results":   [],
            "context":       [],
            "answer":        "",
            "source":        SOURCE_KB,
            "question_type": "WHAT",
            "reasoning":     "",
            "confidence":    "Medium",
            "trace":         [],
        }
        final = self._app.invoke(state)
        elapsed = round(time.perf_counter() - t0, 2)
        return AgentResult(
            answer=final["answer"],
            source=final["source"],
            kb_chunks=final["kb_chunks"],
            kb_distances=final["kb_distances"],
            kb_metadatas=final["kb_metadatas"],
            web_results=final["web_results"],
            question_type=final["question_type"],
            reasoning=final["reasoning"],
            confidence=final.get("confidence", "Medium"),
            trace=final.get("trace", []),
            latency=elapsed,
        )

    # ── Graph construction ─────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("retrieve_kb",  self._node_retrieve_kb)
        g.add_node("retrieve_web", self._node_retrieve_web)
        g.add_node("rerank",       self._node_rerank)
        g.add_node("generate",     self._node_generate)

        g.set_entry_point("retrieve_kb")
        g.add_conditional_edges(
            "retrieve_kb",
            self._route_after_kb,
            {
                "sufficient":   "rerank",
                "insufficient": "retrieve_web",
                "no_kb":        "retrieve_web",
            },
        )
        g.add_edge("retrieve_web", "rerank")
        g.add_edge("rerank",       "generate")
        g.add_edge("generate",     END)

        return g.compile()

    # ── Graph nodes ────────────────────────────────────────────────────────────

    def _node_retrieve_kb(self, state: AgentState) -> dict:
        trace = list(state.get("trace", []))
        has_kb = self.vs is not None and self.vs.count() > 0
        if not has_kb:
            trace.append("⊘ No knowledge base found — will search the web")
            return {"kb_chunks": [], "kb_distances": [], "kb_metadatas": [], "trace": trace}

        trace.append(f"🔍 Querying knowledge base ({self.vs.count()} chunks indexed, top-{state['top_k']})")
        res = self.vs.query(state["question"], n_results=state["top_k"])
        chunks = res.get("documents", [[]])[0]
        dists  = res.get("distances", [[]])[0]
        best   = min(dists) if dists else 1.0
        trace.append(f"📊 Retrieved {len(chunks)} chunk(s), best distance = {best:.3f}")
        return {
            "kb_chunks":    chunks,
            "kb_distances": dists,
            "kb_metadatas": res.get("metadatas",  [[None]])[0] or [],
            "trace":        trace,
        }

    def _route_after_kb(
        self, state: AgentState
    ) -> Literal["sufficient", "insufficient", "no_kb"]:
        has_kb = self.vs is not None and self.vs.count() > 0
        if not has_kb:
            return "no_kb"

        chunks = state.get("kb_chunks", [])
        dists  = state.get("kb_distances", [])

        # No chunks returned at all → web fallback
        if not chunks:
            return "no_kb"

        best = min(dists) if dists else 1.0

        if self._has_uploaded_context(state):
            return "sufficient"

        # KB returned chunks. Only fall back to web if the KB is clearly irrelevant.
        if best < _SUFFICIENCY_THRESHOLD:
            return "sufficient"

        # KB has chunks but they're not very close — still use KB (avoids going
        # to web for image/video questions where the model can't know the content)
        # but also fetch web to supplement.
        return "insufficient"

    def _has_uploaded_context(self, state: AgentState) -> bool:
        """True when retrieval returned chunks from the user's uploaded files."""
        return any(
            (meta or {}).get("modality") in _LOCAL_MODALITIES
            for meta in state.get("kb_metadatas", [])
        )

    def _node_retrieve_web(self, state: AgentState) -> dict:
        trace = list(state.get("trace", []))
        trace.append("🌐 Performing web search (Tavily / DuckDuckGo)")
        results = self.web.search(state["question"])
        trace.append(f"📥 Retrieved {len(results)} web result(s)")
        return {"web_results": results, "trace": trace}

    def _node_rerank(self, state: AgentState) -> dict:
        trace = list(state.get("trace", []))
        question   = state["question"]
        candidates = list(state.get("kb_chunks", []))

        if state.get("web_results"):
            candidates.append(self.web.as_context(state["web_results"]))

        if not candidates:
            trace.append("⚠️ No candidates to rerank")
            return {"context": [], "trace": trace}

        # Cross-encoder reranking (BAAI/bge-reranker-base)
        if self.reranker and len(candidates) > 1:
            try:
                trace.append(f"🔀 Reranking {len(candidates)} candidates with bge-reranker-base")
                pairs  = [(question, doc) for doc in candidates]
                scores = self.reranker.predict(pairs)
                ranked = sorted(
                    zip(candidates, scores), key=lambda x: x[1], reverse=True
                )
                candidates = [doc for doc, _ in ranked]
            except Exception:
                trace.append("⚠️ Reranker failed — using original order")
        else:
            trace.append(f"📋 Passing {len(candidates)} candidate(s) to generator (no reranker)")

        return {"context": candidates, "trace": trace}

    @staticmethod
    def _compute_confidence(kb_dists: list[float], has_kb: bool, has_web: bool) -> str:
        """Derive a human-readable confidence level from retrieval signals."""
        if not has_kb and not has_web:
            return "Low"
        if not has_kb:
            return "Medium"  # web-only answers are decent but unverified
        best = min(kb_dists) if kb_dists else 1.0
        if best < 0.50:
            return "High"
        if best < 0.80:
            return "Medium"
        return "Low"

    def _node_generate(self, state: AgentState) -> dict:
        trace = list(state.get("trace", []))
        question    = state["question"]
        context     = state.get("context", [])
        kb_chunks   = state.get("kb_chunks", [])
        web_results = state.get("web_results", [])
        kb_dists    = state.get("kb_distances", [])

        has_kb  = bool(kb_chunks)
        has_web = bool(web_results)

        uploaded_context = self._has_uploaded_context(state)

        if has_kb and not has_web:
            source    = SOURCE_KB
            if uploaded_context:
                modalities = sorted({
                    (meta or {}).get("modality", "unknown")
                    for meta in state.get("kb_metadatas", [])
                })
                reasoning = (
                    f"Uploaded {', '.join(modalities)} context returned "
                    f"{len(kb_chunks)} chunk(s). Answered from local KB only."
                )
            else:
                reasoning = (
                    f"KB returned {len(kb_chunks)} chunk(s) "
                    f"(best distance: {min(kb_dists):.3f}). "
                    "Reranked with BAAI/bge-reranker-base. No web search needed."
                )
        elif has_web and not has_kb:
            source    = SOURCE_WEB
            reasoning = "No KB or insufficient relevance. Answered from Tavily web search."
        elif has_kb and has_web:
            source    = SOURCE_BOTH
            reasoning = (
                f"KB distance {min(kb_dists):.3f} ≥ {_SUFFICIENCY_THRESHOLD}. "
                "Supplemented with Tavily web search. Results reranked."
            )
        else:
            source    = SOURCE_KB
            reasoning = "No context found in KB or web."

        confidence = self._compute_confidence(kb_dists, has_kb, has_web)
        trace.append(f"🧠 Generating answer with llama3.2:3b (confidence: {confidence})")

        answer, qtype = self.qa.answer(question, context)
        trace.append("✅ Answer generated")

        return {
            "answer":        answer,
            "source":        source,
            "question_type": qtype,
            "reasoning":     reasoning,
            "confidence":    confidence,
            "trace":         trace,
        }
