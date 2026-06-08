"""
test_vector_store.py
────────────────────
Unit tests for the VectorStore module.
All ChromaDB and SentenceTransformer interactions are mocked
to avoid heavyweight dependencies during testing.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import uuid


# ══════════════════════════════════════════════════════════════════════════════
#  INITIALISATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreInit:
    """Tests for VectorStore.__init__."""

    @patch("vector_store._build_ef")
    @patch("chromadb.Client")
    def test_init_creates_client_and_collection(self, mock_client_cls, mock_build_ef):
        """VectorStore creates a ChromaDB client and a collection."""
        mock_build_ef.return_value = (MagicMock(), "sentence-transformers")
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        from vector_store import VectorStore
        vs = VectorStore()

        mock_client_cls.assert_called_once()
        mock_build_ef.assert_called_once()
        assert vs.embed_backend == "sentence-transformers"
        call = mock_client.get_or_create_collection.call_args.kwargs
        assert call["name"] == "rag_docs"
        assert call["metadata"] == {"hnsw:space": "cosine"}
        assert call["embedding_function"] is not None

    def test_class_constants(self):
        """Verify class-level constants."""
        from vector_store import VectorStore
        assert VectorStore.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert VectorStore.COLLECTION_NAME == "rag_docs"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — CREATE A MOCKED VECTORSTORE
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_vs():
    """Create a VectorStore with all external deps mocked."""
    with patch("chromadb.Client") as mock_client_cls, \
         patch("vector_store._build_ef") as mock_build_ef:
        mock_build_ef.return_value = (MagicMock(), "sentence-transformers")
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        from vector_store import VectorStore
        vs = VectorStore()
        return vs, mock_collection


# ══════════════════════════════════════════════════════════════════════════════
#  ADD_DOCUMENTS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAddDocuments:
    """Tests for VectorStore.add_documents."""

    def test_add_documents_empty_list(self):
        """Empty chunk list is a no-op (no call to _col.add)."""
        vs, mock_col = _make_mock_vs()
        vs.add_documents([])
        mock_col.add.assert_not_called()

    def test_add_documents_stores_chunks(self):
        """Chunks are passed to _col.add with generated UUIDs."""
        vs, mock_col = _make_mock_vs()
        chunks = ["chunk one", "chunk two", "chunk three"]
        vs.add_documents(chunks)

        mock_col.add.assert_called_once()
        call_kwargs = mock_col.add.call_args
        assert call_kwargs[1]["documents"] == chunks if call_kwargs[1] else call_kwargs[0][0] == chunks
        # Verify IDs are UUID strings
        ids = call_kwargs[1].get("ids") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1]["ids"]
        assert len(ids) == 3
        for id_str in ids:
            uuid.UUID(id_str)  # Raises ValueError if not valid UUID

    def test_add_documents_single_chunk(self):
        """Single chunk is stored correctly."""
        vs, mock_col = _make_mock_vs()
        vs.add_documents(["only one chunk"])
        mock_col.add.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
#  QUERY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestQuery:
    """Tests for VectorStore.query."""

    def test_query_returns_results(self):
        """Query returns mocked documents and distances."""
        vs, mock_col = _make_mock_vs()
        mock_col.count.return_value = 5
        mock_col.query.return_value = {
            "ids": [["id-a", "id-b"]],
            "documents": [["chunk A", "chunk B"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"source": "a.txt", "modality": "text", "chunk_idx": 0},
                           {"source": "b.txt", "modality": "text", "chunk_idx": 1}]],
        }

        result = vs.query("test question", n_results=2)

        assert result["documents"] == [["chunk A", "chunk B"]]
        assert result["distances"] == [[0.1, 0.3]]
        assert result["metadatas"][0][0]["source"] == "a.txt"
        mock_col.query.assert_called_once_with(
            query_texts=["test question"],
            n_results=5,
            include=["documents", "distances", "metadatas"],
        )

    def test_query_empty_store(self):
        """Query on empty store returns empty documents and distances."""
        vs, mock_col = _make_mock_vs()
        mock_col.count.return_value = 0

        result = vs.query("any question")

        assert result == {"documents": [[]], "distances": [[]], "metadatas": [[]]}
        mock_col.query.assert_not_called()

    def test_query_n_results_capped_at_count(self):
        """n_results is capped to collection count."""
        vs, mock_col = _make_mock_vs()
        mock_col.count.return_value = 2
        mock_col.query.return_value = {
            "ids": [["id-a", "id-b"]],
            "documents": [["a", "b"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{}, {}]],
        }

        vs.query("question", n_results=10)

        mock_col.query.assert_called_once_with(
            query_texts=["question"],
            n_results=2,
            include=["documents", "distances", "metadatas"],
        )

    def test_query_default_n_results(self):
        """Default n_results is 3."""
        vs, mock_col = _make_mock_vs()
        mock_col.count.return_value = 10
        mock_col.query.return_value = {
            "ids": [["id-a", "id-b", "id-c"]],
            "documents": [["a", "b", "c"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [[{}, {}, {}]],
        }

        vs.query("question")

        mock_col.query.assert_called_once_with(
            query_texts=["question"],
            n_results=9,
            include=["documents", "distances", "metadatas"],
        )


# ══════════════════════════════════════════════════════════════════════════════
#  COUNT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCount:
    """Tests for VectorStore.count."""

    def test_count_returns_collection_count(self):
        """count() delegates to _col.count()."""
        vs, mock_col = _make_mock_vs()
        mock_col.count.return_value = 42

        assert vs.count() == 42
        mock_col.count.assert_called()

    def test_count_empty_collection(self):
        """Empty collection returns 0."""
        vs, mock_col = _make_mock_vs()
        mock_col.count.return_value = 0

        assert vs.count() == 0
