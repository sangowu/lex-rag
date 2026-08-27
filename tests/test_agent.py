"""AgenticPipeline 与 StrategySelector 的单元测试（不碰网络与数据库）。

规格第 3 节列了六类新失败，其中能在单测层面钉住的是**策略震荡**（防重复失效）和
**终止原因错乱**。判定器假阴/假阳需要真实语料，那是 2.6 的事。

另一条重点是"trace 必须真的落盘"：主循环的 with 块在生成器里，最终结果如果在 with
内部 yield，调用方拿到结果就丢掉生成器，GeneratorExit 会让整条 trace 消失——这个
坑不写测试根本发现不了。
"""

from unittest.mock import MagicMock

import pytest

from lex_rag.agent import _ACTIONS, AgenticPipeline, StrategySelector
from lex_rag.chunking import ChunkWindow
from lex_rag.config import ContextualConfig
from lex_rag.strategy import RetrievalStrategy
from lex_rag.sufficiency import Verdict
from lex_rag.trace_sink import TraceSink, read_traces


def _cfg() -> ContextualConfig:
    return ContextualConfig(enabled=True, model="m", api_key="k", rpm_limit=60,
                            max_retries=0, retry_backoff_sec=0.0)


def _chunks(ids, score=0.5):
    return [ChunkWindow(chunk_id=i, doc_id="d", text="x" * 50, start=0, end=50,
                        score=score, score_kind="rrf") for i in ids]


def _pipeline(results=None):
    """RAGPipeline 替身：每轮 _query_impl 返回 results 里的下一批。"""
    p = MagicMock()
    p.cfg.reranker.enabled = False
    p.cfg.retrieval.mode = "hybrid"
    p.cfg.retrieval.top_k = 10
    p.cfg.retrieval.rerank_top_k = 60
    p.cfg.hyde_enabled = False
    p.cfg.multi_query_enabled = False
    p.cfg.multi_query_n = 3
    seq = list(results or [_chunks(["a", "b"])])
    p._query_impl.side_effect = lambda *a, **kw: seq.pop(0) if seq else []
    return p


def _agent(pipeline, verdicts, actions=None, **kw):
    """judge 按 verdicts 顺序返回；selector 按 actions 顺序返回。"""
    judge = MagicMock()
    judge.judge.side_effect = list(verdicts)
    sel = MagicMock()
    if actions is not None:
        sel.select.side_effect = [
            (st, f"选了 {name}", "P", "R") for name, st in actions
        ]
    return AgenticPipeline(pipeline, _cfg(), judge=judge, selector=sel, **kw)


# ── 三种终止原因都要能稳定复现（规格完成标准）───────────────────

def test_terminates_as_sufficient_on_first_round():
    p = _pipeline()
    a = _agent(p, [Verdict(sufficient=True)])
    chunks, trace = a.query("q")
    assert [c.chunk_id for c in chunks] == ["a", "b"]
    assert trace[-1] == "terminated_by=sufficient"
    assert p._query_impl.call_count == 1          # 够了就不该再检索


def test_terminates_as_refused_when_judge_says_out_of_scope():
    p = _pipeline()
    a = _agent(p, [Verdict(out_of_scope=True, missing="合同无此条款")])
    _, trace = a.query("q")
    assert trace[-1] == "terminated_by=refused"


def test_terminates_as_max_rounds_and_still_returns_what_it_has():
    """跑满仍不充分时不能返回空——手上这些片段仍是目前最好的。"""
    p = _pipeline([_chunks(["a"]), _chunks(["b"]), _chunks(["c"])])
    a = _agent(p, [Verdict(sufficient=False)] * 3,
               actions=[("bm25", RetrievalStrategy(mode="bm25")),
                        ("hyde", RetrievalStrategy(use_hyde=True))],
               max_iterations=3)
    chunks, trace = a.query("q")
    assert trace[-1] == "terminated_by=max_rounds"
    assert {c.chunk_id for c in chunks} == {"a", "b", "c"}


# ── 防重复必须在执行层拦住 ────────────────────────────────────

def test_repeated_strategy_is_blocked_at_the_execution_layer():
    """选择器坚持给同一个策略时，执行层要拦住——不能只靠 prompt 里写"别重复"。"""
    same = RetrievalStrategy(mode="bm25")
    p = _pipeline([_chunks(["a"]), _chunks(["b"])])
    a = _agent(p, [Verdict(sufficient=False), Verdict(sufficient=False)],
               actions=[("bm25", same), ("bm25", same)], max_iterations=3)
    _, trace = a.query("q")
    assert trace[-1] == "terminated_by=repeat_blocked"
    # 第 2 轮的检索不该发生：拦截要在执行之前
    assert p._query_impl.call_count == 2


def test_no_strategy_left_terminates_cleanly():
    p = _pipeline()
    a = _agent(p, [Verdict(sufficient=False)], actions=[(None, None)], max_iterations=3)
    a.selector.select.side_effect = [(None, "回落失败", "P", "R")]
    _, trace = a.query("q")
    assert trace[-1] == "terminated_by=no_strategy_left"


def test_retrieval_error_does_not_crash_the_loop():
    p = _pipeline()
    p._query_impl.side_effect = RuntimeError("db down")
    a = _agent(p, [Verdict(sufficient=True)])
    chunks, trace = a.query("q")
    assert chunks == [] and trace[-1] == "terminated_by=error"


# ── 累积与去重 ────────────────────────────────────────────────

def test_chunks_accumulate_across_rounds_without_duplicates():
    p = _pipeline([_chunks(["a", "b"]), _chunks(["b", "c"])])
    a = _agent(p, [Verdict(sufficient=False), Verdict(sufficient=True)],
               actions=[("bm25", RetrievalStrategy(mode="bm25"))], max_iterations=3)
    chunks, _ = a.query("q")
    assert [c.chunk_id for c in chunks] == ["a", "b", "c"]


def test_first_round_uses_the_default_strategy_without_calling_the_selector():
    """多数问题一轮就够；为它们各花一次 LLM 调用去挑一个多半还是默认值的策略不划算。"""
    p = _pipeline()
    a = _agent(p, [Verdict(sufficient=True)])
    a.query("q")
    assert not a.selector.select.called


def test_select_first_round_flag_turns_that_off():
    p = _pipeline()
    a = _agent(p, [Verdict(sufficient=True)],
               actions=[("bm25", RetrievalStrategy(mode="bm25"))],
               select_first_round=True)
    a.query("q")
    assert a.selector.select.called


# ── trace 落盘 ────────────────────────────────────────────────

def test_trace_is_written_even_though_query_abandons_the_generator(tmp_path):
    """最终结果必须在 with 块之外 yield。

    否则调用方拿到结果就丢掉生成器，GeneratorExit 从 with 里穿出去——它继承自
    BaseException，sink 的 except Exception 接不住，整条 trace 就没了。
    """
    p = _pipeline([_chunks(["a"]), _chunks(["b"])])
    sink = TraceSink(tmp_path / "t.jsonl", run_id="R")
    a = _agent(p, [Verdict(sufficient=False, missing="缺金额",
                           missing_kind="exact_term"),
                   Verdict(sufficient=True)],
               actions=[("bm25", RetrievalStrategy(mode="bm25"))],
               sink=sink, max_iterations=3)
    a.query("q", doc_id="D")           # 用 query() 而不是消费完整个 stream
    sink.close()

    traces = read_traces(tmp_path / "t.jsonl")
    assert len(traces) == 1
    t = traces[0]
    assert t["terminated_by"] == "sufficient" and t["doc_id"] == "D"
    assert t["n_rounds"] == 2
    assert t["rounds"][0]["verdict"]["missing_kind"] == "exact_term"
    assert t["rounds"][1]["strategy"]["mode"] == "bm25"
    assert t["rounds"][1]["selector_reason"] == "选了 bm25"
    assert [s["name"] for s in t["rounds"][0]["steps"]] == ["retrieval", "judge"]
    assert t["tried_strategies"]


def test_blocked_repeat_is_visible_in_the_trace(tmp_path):
    """策略震荡是规格列的失败类之一，必须能从 trace 里数出来。"""
    same = RetrievalStrategy(mode="bm25")
    p = _pipeline([_chunks(["a"]), _chunks(["b"])])
    sink = TraceSink(tmp_path / "t.jsonl", run_id="R")
    a = _agent(p, [Verdict(sufficient=False)] * 2,
               actions=[("bm25", same), ("bm25", same)], sink=sink, max_iterations=3)
    a.query("q")
    sink.close()

    t = read_traces(tmp_path / "t.jsonl")[0]
    assert t["terminated_by"] == "repeat_blocked"
    assert t["rounds"][-1]["rejected_repeat"] is True


def test_works_without_a_sink():
    p = _pipeline()
    a = _agent(p, [Verdict(sufficient=True)])
    chunks, _ = a.query("q")
    assert len(chunks) == 2


# ── StrategySelector ──────────────────────────────────────────

def _selector(reply):
    s = StrategySelector(_cfg())
    s._chat = MagicMock()
    s._chat.complete_json.return_value = reply
    return s


def test_selector_applies_the_chosen_action():
    s = _selector({"action": "bm25", "query_rewrite": None, "reason": "精确术语"})
    st, reason, _, _ = s.select("q", RetrievalStrategy(), [], "缺金额", "bm25")
    assert st.mode == "bm25" and "精确术语" in reason


def test_selector_applies_a_query_rewrite():
    s = _selector({"action": "rewrite", "query_rewrite": "notice period",
                   "reason": "换合同用语"})
    st, _, _, _ = s.select("q", RetrievalStrategy(), [], "", "")
    assert st.query_text == "notice period"


def test_selector_falls_back_when_the_llm_raises():
    """选择器是链路上又一个 LLM 调用，它挂了不该让整轮循环塌掉。"""
    s = _selector({})
    s._chat.complete_json.side_effect = RuntimeError("429")
    st, reason, _, _ = s.select("q", RetrievalStrategy(), [], "缺金额", "bm25")
    assert st is not None and st.mode == "bm25" and "回落" in reason


def test_selector_falls_back_on_an_unknown_action():
    s = _selector({"action": "teleport", "query_rewrite": None, "reason": "?"})
    st, reason, _, _ = s.select("q", RetrievalStrategy(), [], "", "")
    assert st is not None and "回落" in reason


def test_selector_retries_once_when_it_picks_something_already_tried():
    """撞重复时把"这个试过了"回灌回去重选，而不是直接放弃。"""
    s = StrategySelector(_cfg())
    s._chat = MagicMock()
    s._chat.complete_json.side_effect = [
        {"action": "bm25", "query_rewrite": None, "reason": "第一次"},
        {"action": "hyde", "query_rewrite": None, "reason": "换一个"},
    ]
    tried = [(RetrievalStrategy(mode="bm25").key(), "缺金额")]
    st, reason, prompt, _ = s.select("q", RetrievalStrategy(), tried, "缺金额", "bm25")
    assert st.use_hyde is True and "重选" in reason
    assert "already tried" in prompt


def test_selector_gives_up_to_the_deterministic_ladder_after_two_repeats():
    s = StrategySelector(_cfg())
    s._chat = MagicMock()
    s._chat.complete_json.return_value = {"action": "bm25", "query_rewrite": None,
                                          "reason": "还是它"}
    tried = [(RetrievalStrategy(mode="bm25").key(), "m")]
    st, reason, _, _ = s.select("q", RetrievalStrategy(), tried, "m", "bm25")
    assert st is not None and st.key() not in {k for k, _ in tried}
    assert "回落" in reason


def test_selector_returns_none_when_every_action_is_exhausted():
    s = _selector({"action": "bm25", "query_rewrite": None, "reason": "r"})
    base = RetrievalStrategy()
    tried = [(mut(base).key(), "m") for name, (_, mut) in _ACTIONS.items()
             if mut is not None]
    st, reason, _, _ = s.select("q", base, tried, "m", "bm25")
    assert st is None and "所有动作都已试过" in reason


@pytest.mark.parametrize("hint,expected_mode", [("bm25", "bm25"), ("parent", "bm25")])
def test_fallback_prefers_the_judges_hint(hint, expected_mode):
    """judge 的 missing_kind 是有信息量的，回落时先听它的再按固定顺序。"""
    s = StrategySelector(_cfg())
    st, _ = s.fallback(RetrievalStrategy(), set(), hint)
    assert st.mode == expected_mode


# ── 累积池必须真的被重排（否则多轮是空操作）─────────────────────
# 上一版 `_rank_pool` 开头是 `if len(pool) <= cap: return pool`，而 cap=30、
# contract scope 下 pool 实测最大 17 —— 分支每次都命中，reranker 一次没调过，
# 新片段永远落在第 11 位往后，而 judge 和调用方只看前 10 个。1000 条语料里
# 288 条多轮查询的最终 top-10 与第 0 轮**逐位相同**。见 docs/experiments.md。

def _reranking_pipeline(results, order):
    """带 reranker 的替身：rerank 按 `order` 里的 chunk_id 顺序重排。"""
    p = _pipeline(results)
    p.cfg.reranker.enabled = True
    p.reranker.rerank.side_effect = lambda q, chunks, top_k: sorted(
        chunks, key=lambda c: order.index(c.chunk_id))[:top_k]
    return p


def test_accumulated_pool_is_reranked_even_when_it_fits_under_cap():
    """池子没超上限也要重排——排序和截断是两件事。"""
    p = _reranking_pipeline(
        [_chunks(["a", "b"]), _chunks(["c", "d"])],
        order=["c", "a", "d", "b"],          # 第二轮的 c 应该被排到最前
    )
    a = _agent(p, [Verdict(sufficient=False), Verdict(sufficient=True)],
               actions=[("bm25", RetrievalStrategy(mode="bm25"))], max_iterations=3)
    chunks, _ = a.query("q", k=4)

    assert [c.chunk_id for c in chunks] == ["c", "a", "d", "b"]
    p.reranker.rerank.assert_called_once()


def test_new_chunks_can_reach_the_top_k_the_judge_sees():
    """第二轮捞到的片段必须能进入前 k —— 否则多轮对输出毫无影响。"""
    p = _reranking_pipeline(
        [_chunks(["a", "b"]), _chunks(["new"])],
        order=["new", "a", "b"],
    )
    judge = MagicMock()
    judge.judge.side_effect = [Verdict(sufficient=False), Verdict(sufficient=True)]
    sel = MagicMock()
    sel.select.side_effect = [(RetrievalStrategy(mode="bm25"), "换 bm25", "P", "R")]
    a = AgenticPipeline(p, _cfg(), judge=judge, selector=sel, max_iterations=3)
    a.query("q", k=2)

    seen = [c.chunk_id for c in judge.judge.call_args_list[-1].args[1]]
    assert "new" in seen


def test_first_round_is_not_reranked_twice():
    """第 0 轮的结果已被检索层排好，再 rerank 一次是白花一次 API 往返。"""
    p = _reranking_pipeline([_chunks(["a", "b"])], order=["a", "b"])
    a = _agent(p, [Verdict(sufficient=True)])
    a.query("q")
    p.reranker.rerank.assert_not_called()


def test_round_that_finds_nothing_new_is_not_reranked():
    """去重后没长出新片段时重排也是同样的输入同样的输出。"""
    p = _reranking_pipeline(
        [_chunks(["a", "b"]), _chunks(["a", "b"])],   # 第二轮全是重复
        order=["a", "b"],
    )
    a = _agent(p, [Verdict(sufficient=False), Verdict(sufficient=True)],
               actions=[("bm25", RetrievalStrategy(mode="bm25"))], max_iterations=3)
    a.query("q")
    p.reranker.rerank.assert_not_called()


def test_rerank_failure_falls_back_to_plain_truncation():
    """重排失败不该让整轮循环挂掉。"""
    p = _pipeline([_chunks(["a", "b"]), _chunks(["c"])])
    p.cfg.reranker.enabled = True
    p.reranker.rerank.side_effect = RuntimeError("rerank down")
    a = _agent(p, [Verdict(sufficient=False), Verdict(sufficient=True)],
               actions=[("bm25", RetrievalStrategy(mode="bm25"))], max_iterations=3)
    chunks, trace = a.query("q", k=3)

    assert [c.chunk_id for c in chunks] == ["a", "b", "c"]
    assert trace[-1] == "terminated_by=sufficient"


def test_default_max_iterations_is_two():
    """第 2 轮从未救回过任何一条（两轮 1000 条语料一致），却贡献 33~38 次白烧。

    默认值是行为约定，改它要有语料支持——钉在这里免得被顺手改回去。
    见 docs/experiments.md「白烧率怎么控制」。
    """
    a = AgenticPipeline(_pipeline(), _cfg(), judge=MagicMock(), selector=MagicMock())
    assert a.max_iterations == 2


def test_loop_stops_after_two_rounds_by_default():
    p = _pipeline([_chunks(["a"]), _chunks(["b"]), _chunks(["c"])])
    a = _agent(p, [Verdict(sufficient=False), Verdict(sufficient=False)],
               actions=[("bm25", RetrievalStrategy(mode="bm25"))])
    _, trace = a.query("q")

    assert p._query_impl.call_count == 2
    assert trace[-1] == "terminated_by=max_rounds"
