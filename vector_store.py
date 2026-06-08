"""
vector_store.py
───────────────
Hybrid retrieval:
  • Dense  — ChromaDB with Ollama nomic-embed-text embeddings
  • Sparse — SQLite FTS5 (BM25 lexical search)
  • Fusion — Reciprocal Rank Fusion (RRF, k=60)

Falls back to sentence-transformers (all-MiniLM-L6-v2) if Ollama is unavailable.
"""
from __future__ import annotations
import json
import re
import sqlite3
import uuid
import urllib.request
import urllib.error
from typing import Any

OLLAMA_EMBED_URL  = "http://localhost:11434/api/embed"
OLLAMA_MODEL      = "nomic-embed-text"
FALLBACK_MODEL    = "all-MiniLM-L6-v2"
RRF_K             = 60


# ── Embedding functions ────────────────────────────────────────────────────────

class _OllamaEF:
    """Calls Ollama nomic-embed-text; raises on failure so fallback can catch."""
    def __call__(self, input: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": OLLAMA_MODEL, "input": input}).encode()
        req = urllib.request.Request(
            OLLAMA_EMBED_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["embeddings"]

    def name(self) -> str:
        return "ollama-nomic-embed-text"


class _SentenceTransformerEF:
    """Fallback: sentence-transformers all-MiniLM-L6-v2 (device=cpu)."""
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(FALLBACK_MODEL, device="cpu")

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._model.encode(input, convert_to_numpy=True).tolist()

    def name(self) -> str:
        return "sentence-transformer-minilm"


def _build_ef():
    """Return Ollama EF if reachable, else sentence-transformers EF."""
    try:
        ef = _OllamaEF()
        ef(["test"])          # probe
        return ef, "ollama"
    except Exception:
        return _SentenceTransformerEF(), "sentence-transformers"


# ── VectorStore ────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Hybrid in-memory store:
      dense  → ChromaDB + Ollama nomic-embed-text (or MiniLM fallback)
      sparse → SQLite FTS5 BM25
      fusion → Reciprocal Rank Fusion
    """
    EMBEDDING_MODEL = FALLBACK_MODEL
    COLLECTION_NAME = "rag_docs"

    def __init__(self):
        import chromadb
        from chromadb import EmbeddingFunction, Embeddings

        self._ef_impl, self.embed_backend = _build_ef()

        # Wrap in ChromaDB-compatible class
        _impl = self._ef_impl
        class _ChromaEF(EmbeddingFunction):
            def __call__(self, input: list[str]) -> Embeddings:
                return _impl(input)
            def name(self) -> str:
                return _impl.name()

        self._client = chromadb.Client()
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=_ChromaEF(),
            metadata={"hnsw:space": "cosine"},
        )

        # SQLite FTS5 for BM25 lexical search
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.execute(
            "CREATE VIRTUAL TABLE docs USING fts5(id UNINDEXED, text, source UNINDEXED, modality UNINDEXED)"
        )
        self._db.commit()

        # Local doc store for fast ID → document lookup after RRF
        self._store: dict[str, dict] = {}

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_documents(
        self,
        chunks:      list[str],
        source_file: str = "",
        modality:    str = "pdf",
    ) -> None:
        if not chunks:
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"source": source_file, "modality": modality, "chunk_idx": i}
            for i in range(len(chunks))
        ]

        # Dense store
        self._col.add(documents=chunks, ids=ids, metadatas=metadatas)

        # Sparse store
        self._db.executemany(
            "INSERT INTO docs(id, text, source, modality) VALUES(?,?,?,?)",
            [(ids[i], chunks[i], source_file, modality) for i in range(len(chunks))],
        )
        self._db.commit()

        # Local lookup
        for i, id_ in enumerate(ids):
            self._store[id_] = {
                "text": chunks[i], "source": source_file,
                "modality": modality, "chunk_idx": i,
            }

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def query(self, question: str, n_results: int = 3) -> dict[str, Any]:
        total = self._col.count()
        if total == 0:
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

        fetch = min(n_results * 3, total)

        # Dense retrieval
        dense = self._col.query(
            query_texts=[question], n_results=fetch,
            include=["documents", "distances", "metadatas"],
        )
        dense_docs  = dense.get("documents", [[]])[0]
        dense_dists_list = dense.get("distances", [[]])[0]
        dense_metas = dense.get("metadatas", [[]])[0] or []
        dense_ids   = dense.get("ids", [[]])[0]
        if not dense_ids:
            dense_ids = [f"__dense_{i}" for i in range(len(dense_docs))]
            for i, id_ in enumerate(dense_ids):
                meta = dense_metas[i] if i < len(dense_metas) else {}
                self._store[id_] = {
                    "text": dense_docs[i],
                    "source": meta.get("source", ""),
                    "modality": meta.get("modality", ""),
                    "chunk_idx": meta.get("chunk_idx", i),
                }
        dense_dists = {id_: d for id_, d in zip(dense_ids, dense_dists_list)}

        # Sparse BM25 retrieval
        sparse_ids = self._bm25_query(question, fetch)

        # RRF fusion
        fused_ids = self._rrf(dense_ids, sparse_ids)[:n_results]

        dense_doc_by_id = {id_: doc for id_, doc in zip(dense_ids, dense_docs)}
        dense_meta_by_id = {id_: meta for id_, meta in zip(dense_ids, dense_metas)}

        # Build result
        docs, dists, metas = [], [], []
        for id_ in fused_ids:
            rec = self._store.get(id_, {})
            fallback_meta = dense_meta_by_id.get(id_, {}) or {}
            docs.append(rec.get("text", dense_doc_by_id.get(id_, "")))
            dists.append(dense_dists.get(id_, 1.0))
            metas.append({
                "source":    rec.get("source", fallback_meta.get("source", "")),
                "modality":  rec.get("modality", fallback_meta.get("modality", "")),
                "chunk_idx": rec.get("chunk_idx", fallback_meta.get("chunk_idx", 0)),
            })

        return {"documents": [docs], "distances": [dists], "metadatas": [metas]}

    # ── BM25 helpers ───────────────────────────────────────────────────────────

    def _bm25_query(self, question: str, n: int) -> list[str]:
        words = re.findall(r"\w+", question)
        if not words:
            return []
        fts_q = " OR ".join(words)
        try:
            cur = self._db.execute(
                "SELECT id FROM docs WHERE text MATCH ? ORDER BY rank LIMIT ?",
                (fts_q, n),
            )
            return [row[0] for row in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

    # ── RRF ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf(dense_ids: list[str], sparse_ids: list[str], k: int = RRF_K) -> list[str]:
        scores: dict[str, float] = {}
        for rank, id_ in enumerate(dense_ids):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank + 1)
        for rank, id_ in enumerate(sparse_ids):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    # ── Deletion ────────────────────────────────────────────────────────────────

    def delete_by_source(self, source_file: str) -> int:
        """Remove all chunks that belong to *source_file* from every store.

        Returns the number of chunks deleted.
        """
        # 1. Find IDs from the local store (fastest lookup)
        ids_to_delete = [
            id_ for id_, rec in self._store.items()
            if rec.get("source") == source_file
        ]

        if not ids_to_delete:
            # Fallback: query ChromaDB metadata directly
            try:
                res = self._col.get(
                    where={"source": source_file},
                    include=[],
                )
                ids_to_delete = res.get("ids", [])
            except Exception:
                pass

        if not ids_to_delete:
            return 0

        # 2. ChromaDB — delete by IDs
        try:
            self._col.delete(ids=ids_to_delete)
        except Exception:
            pass

        # 3. SQLite FTS5 — delete by source
        try:
            self._db.execute(
                "DELETE FROM docs WHERE source = ?", (source_file,)
            )
            self._db.commit()
        except Exception:
            pass

        # 4. Local lookup dict
        for id_ in ids_to_delete:
            self._store.pop(id_, None)

        return len(ids_to_delete)

    # ── Metadata ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._col.count()
