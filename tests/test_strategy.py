"""RetrievalStrategy 与 _query_impl 重构的单元测试（不碰网络与数据库）。

重点不是覆盖率，而是**与改造前的行为等价**：这次重构把五个检索开关从构造期
搬到了运行期，如果哪个默认值悄悄变了，后面所有"失败分析"都会是在分析重构
自己引入的 bug——而那时已经分不清是谁的锅了。
"""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from lex_rag.strategy import RetrievalStrategy


class _Cfg:
    """最小可用的 AppConfig 替身。"""

    def __init__(self, mode="hybrid", top_k=10, rerank_top_k=60, rerank=True,
                 hyde=False, mq=False, mq_n=3):
        self.retrieval = MagicMock(mode=mode, top_k=top_k, rerank_top_k=rerank_top_k)
        self.reranker = MagicMock(enabled=rerank)
        self.hyde_enabled = hyde
        self.multi_query_enabled = mq
        self.multi_query_n = mq_n


# ── from_config：默认策略必须复刻改造前的取值 ──────────────────────

def test_from_config_mirrors_pre_refactor_defaults():
    st = RetrievalStrategy.from_config(_Cfg())
    assert (st.mode, st.top_k, st.fetch_k, st.rerank) == ("hybrid", 10, 60, True)
    assert st.use_hyde is False and st.use_multi_query is False


def test_fetch_k_falls_back_to_top_k_when_rerank_disabled():
    """改造前是 `fetch_k = rerank_top_k if reranker else k`，不能改语义。"""
    st = RetrievalStrategy.from_config(_Cfg(rerank=False, top_k=7, rerank_top_k=60))
    assert st.fetch_k == 7 and st.rerank is False


# ── with_top_k：调用方显式传 k 时的覆盖规则 ───────────────────────

def test_with_top_k_keeps_fetch_k_when_reranking():
    """开 rerank 时候选池由 rerank_top_k 决定，与 top_k 无关。"""
    st = RetrievalStrategy.from_config(_Cfg()).with_top_k(3)
    assert st.top_k == 3 and st.fetch_k == 60


def test_with_top_k_moves_fetch_k_when_not_reranking():
    """不开 rerank 时候选池就是返回集，两者必须一起动。"""
    st = RetrievalStrategy.from_config(_Cfg(rerank=False)).with_top_k(25)
    assert st.top_k == 25 and st.fetch_k == 25


def test_with_top_k_none_is_identity():
    st = RetrievalStrategy.from_config(_Cfg())
    assert st.with_top_k(None) is st


# ── key()：防重复的基础 ────────────────────────────────────────────

def test_key_distinguishes_strategies_that_differ():
    base = RetrievalStrategy()
    assert base.key() != replace(base, mode="bm25").key()
    assert base.key() != replace(base, use_hyde=True).key()
    assert base.key() != replace(base, query_text="rewritten").key()


def test_key_is_stable_for_identical_strategies():
    """同一策略两次构造必须得到同一个 key，否则防重复形同虚设。"""
    a = RetrievalStrategy(mode="bm25", query_text="notice period")
    b = RetrievalStrategy(mode="bm25", query_text="notice period")
    assert a.key() == b.key()


def test_frozen_dataclass_cannot_be_mutated():
    """不可变是刻意的：策略要被记进 trace，事后不能被人改掉。"""
    with pytest.raises(Exception):
        RetrievalStrategy().mode = "bm25"       # type: ignore[misc]


# ── _query_impl：策略真的被用上了吗 ────────────────────────────────

def _pipeline(mode="hybrid", rerank=True):
    """构造一个所有外部依赖都被替换掉的 RAGPipeline。"""
    from lex_rag.pipeline import RAGPipeline

    p = RAGPipeline.__new__(RAGPipeline)          # 跳过 __init__ 的真实连接
    p.cfg = _Cfg(mode=mode, rerank=rerank)
    p.cfg.database = MagicMock(table="chunks")
    p.embedder = MagicMock()
    p.embedder.embed_text.return_value = [0.1] * 8
    p.store = MagicMock()
    p.store.search_hybrid.return_value = []
    p.store.search_vector.return_value = []
    p.store.search_bm25.return_value = []
    p._chunk_mode_cache = "standard"
    p._reranker = MagicMock()
    p._reranker.rerank.return_value = []
    p._hyde = MagicMock()
    p._hyde.generate.return_value = "hypothetical clause"
    p._expander = None
    p._contextualizer = None
    p._meta_extractor = None
    return p


def test_default_strategy_uses_hybrid_and_reranks():
    p = _pipeline()
    p._query_impl("governing law?")
    assert p.store.search_hybrid.called
    assert p._reranker.rerank.called


def test_bm25_strategy_skips_embedding_entirely():
    """BM25 不需要向量——多一次 embedding 调用就是白花钱和白等。"""
    p = _pipeline()
    p._query_impl("penalty amount", strategy=RetrievalStrategy(mode="bm25", rerank=False))
    assert p.store.search_bm25.called
    assert not p.embedder.embed_text.called


def test_hyde_only_affects_the_vector_side_of_hybrid():
    """HyDE 生成的假设条款用于向量检索；BM25 那一路仍用原文。

    把假设文本喂给关键词匹配只会引入噪声——它是模型编出来的，不是用户问的。
    """
    p = _pipeline()
    p._query_impl("termination?", strategy=RetrievalStrategy(use_hyde=True, rerank=False))
    p.embedder.embed_text.assert_called_once_with("hypothetical clause")
    assert p.store.search_hybrid.call_args[0][0] == "termination?"


def test_rewritten_query_is_used_for_search_but_rerank_scores_the_original():
    """重写是为了改善召回；相关性判断应当对着用户真正问的问题。"""
    p = _pipeline()
    st = RetrievalStrategy(query_text="notice period termination renewal")
    p._query_impl("How long is the notice period?", strategy=st)

    assert p.store.search_hybrid.call_args[0][0] == "notice period termination renewal"
    assert p._reranker.rerank.call_args[0][0] == "How long is the notice period?"


def test_expand_parent_is_ignored_on_standard_tables():
    """standard 表里没有 parent 行，强行展开只会查空。"""
    p = _pipeline()
    p._query_impl("q", strategy=RetrievalStrategy(expand_parent=True, rerank=False))
    assert not p.store.expand_to_parent.called


def test_rerank_disabled_truncates_to_top_k_itself():
    """不走 reranker 时，截断得自己做——改造前是靠 fetch_k==top_k 隐式实现的。"""
    p = _pipeline()
    p.store.search_hybrid.return_value = [f"c{i}" for i in range(10)]
    out = p._query_impl("q", strategy=RetrievalStrategy(rerank=False, top_k=3, fetch_k=10))
    assert out == ["c0", "c1", "c2"]
    assert not p._reranker.rerank.called
