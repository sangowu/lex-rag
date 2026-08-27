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


# ── 连接容错：一次 SQL 错误不能报废整条连接 ────────────────────

def test_cursor_rolls_back_on_error_so_the_connection_stays_usable():
    """psycopg 在事务里出错后，该连接上的后续语句一律报 InFailedSqlTransaction，
    直到显式 rollback。不回滚的话一次瞬时错误（死锁、超时）会**永久**报废连接。

    实测：4 个 worker 并发构造 VectorStore 触发一次 DDL 死锁，同一批 20 条里
    随后 15 条全部是 InFailedSqlTransaction——真正的死锁只发生过一次，其余都是
    连接已废的连锁反应。这类症状是"从此全部失败"，很容易把排查引到错误的方向。
    """
    store = _make_store()
    store.conn = MagicMock()
    store.conn.cursor.return_value.__enter__.side_effect = RuntimeError("deadlock detected")

    try:
        with store._cursor():
            pass
    except RuntimeError:
        pass
    else:
        raise AssertionError("异常应当继续向上抛，调用方需要知道这一次失败了")

    store.conn.rollback.assert_called_once()


def test_cursor_does_not_roll_back_on_success():
    store = _make_store()
    store.conn = MagicMock()
    with store._cursor() as cur:
        assert cur is store.conn.cursor.return_value.__enter__.return_value
    assert not store.conn.rollback.called


def test_rollback_failure_does_not_mask_the_original_error():
    """连接彻底断了时 rollback 本身也会抛。原始异常才是有信息量的那个。"""
    store = _make_store()
    store.conn = MagicMock()
    store.conn.cursor.return_value.__enter__.side_effect = RuntimeError("original")
    store.conn.rollback.side_effect = RuntimeError("connection already closed")

    try:
        with store._cursor():
            pass
    except RuntimeError as e:
        assert "original" in str(e)
    else:
        raise AssertionError("应当抛出原始异常")
