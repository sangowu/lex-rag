"""Unit tests for VectorStore's pure-logic pieces (RRF fusion, parent expansion).

VectorStore.__init__ opens a real Postgres connection, so these tests build the
instance via object.__new__ (skipping __init__) and mock out the DB-touching
methods, mirroring the pattern used in rag_demo's test_vector_store_dense_embeddings.py.
"""

from unittest.mock import MagicMock

from lex_rag.chunking import ChunkWindow
from lex_rag.store import VectorStore


def _make_store() -> VectorStore:
    return object.__new__(VectorStore)


def _chunk(chunk_id: str, start: int) -> ChunkWindow:
    return ChunkWindow(chunk_id=chunk_id, doc_id="doc1", text=chunk_id, start=start, end=start + 1)


def test_search_hybrid_merges_and_ranks_by_rrf_score():
    store = _make_store()
    store.search_vector = MagicMock(return_value=[_chunk("a", 0), _chunk("b", 1)])
    store.search_bm25 = MagicMock(return_value=[_chunk("b", 1), _chunk("c", 2)])

    result = store.search_hybrid("query", [0.1, 0.2], k=3)

    # "b" ranks in both lists (rank1 in vector, rank0 in bm25) -> highest combined RRF score
    assert [c.chunk_id for c in result] == ["b", "a", "c"]


def test_search_hybrid_respects_k_limit():
    store = _make_store()
    store.search_vector = MagicMock(return_value=[_chunk(f"v{i}", i) for i in range(5)])
    store.search_bm25 = MagicMock(return_value=[])

    result = store.search_hybrid("query", [0.1], k=2)

    assert [c.chunk_id for c in result] == ["v0", "v1"]


def test_search_hybrid_empty_results_from_both_sources():
    store = _make_store()
    store.search_vector = MagicMock(return_value=[])
    store.search_bm25 = MagicMock(return_value=[])

    assert store.search_hybrid("query", [0.1], k=5) == []


def test_expand_to_parent_empty_children_returns_empty_list():
    store = _make_store()
    assert store.expand_to_parent([]) == []
