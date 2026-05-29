"""
vector_store.py
───────────────
Wraps ChromaDB with a sentence-transformer embedding function.
Each VectorStore instance uses an in-memory Chroma collection.
"""

from __future__ import annotations
import uuid
from typing import Any


class VectorStore:
    """Lightweight ChromaDB wrapper for document chunks."""

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, 384-dim, MIT licence
    COLLECTION_NAME = "rag_docs"

    def __init__(self):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._client = chromadb.Client()          # in-memory (ephemeral)
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=self.EMBEDDING_MODEL
        )
        # Fresh collection for every document session
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[str]) -> None:
        """Embed and store a list of text chunks."""
        if not chunks:
            return
        ids = [str(uuid.uuid4()) for _ in chunks]
        self._col.add(documents=chunks, ids=ids)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def query(self, question: str, n_results: int = 3) -> dict[str, Any]:
        """Return the top-k most relevant chunks for the query."""
        n = min(n_results, self._col.count())
        if n == 0:
            return {"documents": [[]], "distances": [[]]}
        return self._col.query(query_texts=[question], n_results=n)

    # ── Metadata ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._col.count()
