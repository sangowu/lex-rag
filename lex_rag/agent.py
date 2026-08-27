"""
Agentic 检索循环 —— 规格 `docs/agentic_loop_upgrade.md` 第 2.4 节。

## 这个类为什么被整个重写

旧实现的重试条件是 `if chunks:` 就停，也就是**只有检索返回空列表才会有第二轮**。
而 hybrid 检索（向量 + BM25 融合）几乎永远返回非空结果，所以那个循环从来没跑过
第二轮——它是死代码。动作空间也只有"重写查询"一个，而检索失败的原因有好几种，
重写查询只能救其中一种。

真正的问题从来不是"没检索到"，而是"**检索到了但不对 / 不全**"。所以新循环的
继续条件不是"空结果"，而是 `SufficiencyJudge` 判定上下文不足。

## 循环

    for round in range(max_iterations):
        strategy = 选策略（首轮用默认，之后由 LLM 选，禁止重复已试过的 key）
        chunks   = pipeline.query(question, doc_id, strategy)
        累积     = 去重合并(累积, chunks)，再用 reranker 重排成可比的一池
        verdict  = judge(question, 累积)
        out_of_scope → terminated_by="refused"；sufficient → "sufficient"
    else: terminated_by="max_rounds"

## 三个刻意的设计选择

**防重复在执行层拦截，不在 prompt 里。** 规格明确要求，而且 #25 的冒烟里朴素映射
连续两轮选了 bm25，确实会撞上。选择器选中已试过的 key 时直接拒绝并把"这个试过了、
结果是……"回灌给它重选一次；再撞就退回固定顺序里第一个没试过的。

**判定器的 confidence 不进控制流。** #24 实测它在两个分支上重叠（sufficient 时
0.95~1.00，否则 0.80~1.00），按它设阈值等于按噪声决策。字段照常落 trace。

**累积上下文要重排，不能直接拼。** 不同轮的分数 kind 不同（bm25 / rrf / cosine
互不可比），直接按分数排是拿苹果比橘子，按轮次拼接又会让第一轮的尾巴压过第二轮的
头部。所以每轮累积后用 reranker 对**整池**重打一次分——这也正好是规格第 3 节
"累积上下文污染"那个失败类的抓手。没开 reranker 时退化为按轮次顺序拼接。

## 向后兼容

`query()` / `query_stream()` 的签名和返回值不变（`serve.py` 与 `scripts/eval.py`
都依赖），`query_trace` 仍是 `list[str]`，第 0 项仍是原始问题。区别是后续各项从
"重写后的查询"变成"本轮用了什么策略"。

> ⚠️ `scripts/eval.py` 的 `run_agentic_eval` 是按旧前提写的（Pass 1 挑
> `chunks == []` 的样本给 Pass 2 恢复）。在新循环下那个前提不成立——空结果本来
> 就几乎不发生。它还能跑，但测的不是这个循环的能力。真正的评估见规格 2.6。
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Iterator

from lex_rag.chunking import ChunkWindow
from lex_rag.config import ContextualConfig
from lex_rag.llm import ChatClient
from lex_rag.pipeline import RAGPipeline
from lex_rag.strategy import RetrievalStrategy
from lex_rag.sufficiency import SufficiencyJudge, Verdict

# 动作 → 怎么改策略。**顺序即退化顺序**：选择器两次都撞重复时，按这个顺序取第一个
# 没试过的。排在前面的是实测最可能救回来的（#24 的 missing_kind 分布里
# concept_mismatch 与 exact_term 占了绝大多数）。
_ACTIONS: dict[str, tuple[str, Any]] = {
    "bm25": (
        "exact keyword matching; use when the answer hinges on a specific figure, "
        "date, section number or defined term",
        lambda st: replace(st, mode="bm25"),
    ),
    "hyde": (
        "generate a hypothetical clause and search with it; use when the contract "
        "likely words the concept differently than the question does",
        lambda st: replace(st, mode="hybrid", use_hyde=True),
    ),
    "multi_query": (
        "fan out into several sub-queries and merge; use when the question asks "
        "about several things at once",
        lambda st: replace(st, mode="hybrid", use_multi_query=True),
    ),
    "vector": (
        "pure semantic search; use when keyword matching is pulling in the wrong "
        "sections",
        lambda st: replace(st, mode="vector"),
    ),
    "rewrite": (
        "keep the current retrieval mode but restate the query in contract language",
        None,          # 需要 query_rewrite 字段，单独处理
    ),
}

# judge 的 missing_kind → 建议动作。选择器拿它当提示，但不被它绑死。
_HINT_TO_ACTION = {"bm25": "bm25", "hyde": "hyde",
                   "multi_query": "multi_query", "parent": "bm25"}

_SELECT_PROMPT = """\
You are choosing the next retrieval strategy for a legal contract question.
The previous attempts did not retrieve enough context to answer it.

Question: {question}

Attempts so far:
{tried}

The context auditor said what is still missing: {missing}
Its suggested action: {hint}

Available actions:
{actions}

Rules:
- Do NOT pick an action that was already tried, unless you also supply a
  materially different "query_rewrite".
- Pick the action that addresses what is *missing*, not the one that sounds
  most powerful.

Reply with JSON only:
{{"action": "one of the action names above",
  "query_rewrite": "a restated query, or null to keep the original",
  "reason": "one sentence explaining the choice"}}"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(_ACTIONS)},
        "query_rewrite": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "query_rewrite", "reason"],
    "additionalProperties": False,
}


class StrategySelector:
    """选下一轮的检索策略。LLM 决策，失败时回落到 missing_kind 的映射表。

    回落不是可有可无的兜底：选择器是链路上又一个 LLM 调用，而它失败（限流、
    解析不出）时循环不该整个塌掉——退回一个确定性的选择，比放弃这一轮好。
    """

    def __init__(self, cfg: ContextualConfig, actions: list[str] | None = None) -> None:
        self.cfg = cfg
        self.actions = actions or list(_ACTIONS)
        self._chat = ChatClient.from_config(cfg)

    def _apply(self, action: str, base: RetrievalStrategy,
               rewrite: str | None) -> RetrievalStrategy:
        st = base
        _, mutate = _ACTIONS.get(action, (None, None))
        if mutate is not None:
            st = mutate(st)
        if rewrite:
            st = replace(st, query_text=rewrite)
        return st

    def fallback(self, base: RetrievalStrategy, tried: set[str],
                 hint: str) -> tuple[RetrievalStrategy | None, str]:
        """确定性回落：先听 judge 的建议，再按 `_ACTIONS` 顺序取第一个没试过的。"""
        order = [a for a in (_HINT_TO_ACTION.get(hint),) if a] + list(self.actions)
        for action in order:
            if action == "rewrite":          # 没有重写文本时它等于什么都不做
                continue
            st = self._apply(action, base, None)
            if st.key() not in tried:
                return st, f"回落：{action}（选择器不可用或重复）"
        return None, "回落失败：所有动作都已试过"

    def select(self, question: str, base: RetrievalStrategy,
               tried: list[tuple[str, str]], missing: str, hint: str,
               ) -> tuple[RetrievalStrategy | None, str, str, str]:
        """返回 (策略, 理由, prompt, 原始响应)。策略为 None 表示无路可走。"""
        tried_keys = {k for k, _ in tried}
        tried_txt = "\n".join(f"- {k}\n  → still missing: {m}" for k, m in tried) or "- (none)"
        actions_txt = "\n".join(f'- "{a}": {_ACTIONS[a][0]}' for a in self.actions)
        prompt = _SELECT_PROMPT.format(
            question=question, tried=tried_txt, missing=missing or "(unspecified)",
            hint=_HINT_TO_ACTION.get(hint, "(none)"), actions=actions_txt,
        )
        try:
            data = self._chat.complete_json(prompt, schema=_SCHEMA,
                                            trace_name="agent.select_strategy")
        except Exception as e:
            st, why = self.fallback(base, tried_keys, hint)
            return st, f"{why}（选择器异常 {type(e).__name__}）", prompt, ""

        raw = str(data)
        action = str(data.get("action", "")).strip().lower()
        rewrite = data.get("query_rewrite") or None
        reason = str(data.get("reason", "") or "").strip()
        if action not in self.actions:
            st, why = self.fallback(base, tried_keys, hint)
            return st, f"{why}（选择器给了未知动作 {action!r}）", prompt, raw

        st = self._apply(action, base, rewrite)
        if st.key() in tried_keys:
            # 执行层拦截：把"这个试过了"回灌给选择器，只重选一次。
            retry_prompt = prompt + (
                f'\n\nYou picked "{action}", which was already tried and did not '
                f"help. Pick a different action, or supply a materially different "
                f"query_rewrite."
            )
            try:
                data2 = self._chat.complete_json(retry_prompt, schema=_SCHEMA,
                                                 trace_name="agent.select_strategy.retry")
                a2 = str(data2.get("action", "")).strip().lower()
                if a2 in self.actions:
                    st2 = self._apply(a2, base, data2.get("query_rewrite") or None)
                    if st2.key() not in tried_keys:
                        return (st2, f"重选：{a2}（{data2.get('reason', '')}）",
                                retry_prompt, str(data2))
            except Exception:
                pass
            st, why = self.fallback(base, tried_keys, hint)
            return st, f"{why}（选择器两次都选了已试过的 {action}）", retry_prompt, raw

        return st, f"{action}：{reason}", prompt, raw


def _dedupe(existing: list[ChunkWindow], incoming: list[ChunkWindow]) -> list[ChunkWindow]:
    seen = {c.chunk_id for c in existing}
    return existing + [c for c in incoming if c.chunk_id not in seen]


class AgenticPipeline:
    """多轮检索：选策略 → 检索 → 累积 → 判定是否够了。

    `sink` 传入 `TraceSink` 时，每轮的策略、选择器理由、chunk 分数、判定结果、
    累积量与 `terminated_by` 全部落盘（规格 2.5）。不传则只是不落盘，逻辑不变。
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        cfg: ContextualConfig,
        max_iterations: int = 3,
        *,
        sink: Any = None,
        select_first_round: bool = False,
        judge: SufficiencyJudge | None = None,
        selector: StrategySelector | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.cfg = cfg
        self.max_iterations = max_iterations
        self.sink = sink
        # 首轮是否也调选择器。默认 False：多数问题一轮就够，为它们各花一次 LLM
        # 调用去挑一个多半还是默认值的策略并不划算。开着能让首轮也因题而异，
        # 代价是 100% 的查询都多一次调用——两种配置的取舍留给规格 2.6 去测。
        self.select_first_round = select_first_round
        # judge 与选择器都不开 thinking：它们在拒答路径上，而 #24 实测 thinking
        # 在 200 条上买不到可测的质量，只买到 3 倍延迟。
        fast = replace(cfg, thinking=False)
        self.judge = judge or SufficiencyJudge(fast)
        self.selector = selector or StrategySelector(fast)

    def _rank_pool(self, question: str, pool: list[ChunkWindow],
                   cap: int) -> list[ChunkWindow]:
        """把累积池重排成一个可比的排序，并截到 cap。

        跨轮的分数 kind 不同（bm25 / rrf / cosine 互不可比），直接按分数排是拿苹果
        比橘子；按轮次拼接则会让第一轮的尾巴压过第二轮的头部。用 reranker 对整池
        重打一次分是唯一能让它们同尺的做法。

        **cap 必须显著大于 k。** 早先这里传的是 `max(k, 10)`，等于每轮把池子截回
        一轮的容量——刚检索到的新片段立刻被扔掉，累积形同虚设（实测平均累积
        chunk 10.2，而第一轮就有 10 个）。截断应该发生在喂给 judge 和生成器的
        时候，而不是发生在池子上。
        """
        if len(pool) <= cap:
            return pool
        try:
            if self.pipeline.cfg.reranker.enabled:
                return self.pipeline.reranker.rerank(question, pool, top_k=cap)
        except Exception:
            pass
        return pool[:cap]

    # ── 主循环 ──────────────────────────────────────────────────
    def query_stream(
        self,
        question: str,
        doc_id: str | None = None,
        k: int = 10,
        meta: dict | None = None,
    ) -> Iterator[str | tuple[list[ChunkWindow], list[str]]]:
        """先 yield str 状态消息，最后 yield (chunks, query_trace)。

        `meta` 原样写进 trace 的 meta 字段。跑语料时用它带上样本 id 与 gold 标签，
        让语料自包含——下游分析不必再回头 join qa 文件。
        """
        base = RetrievalStrategy.from_config(self.pipeline.cfg).with_top_k(k)
        strategy = base
        tried: list[tuple[str, str]] = []
        pool: list[ChunkWindow] = []
        trace_lines: list[str] = [question]
        terminated_by = "max_rounds"
        verdict: Verdict | None = None

        qctx = (self.sink.query(question, doc_id=doc_id, meta={"k": k, **(meta or {})})
                if self.sink is not None else _NullCtx())
        with qctx as qt:
            for rnd in range(self.max_iterations):
                # ── 选策略 ──────────────────────────────────────
                reason, sel_prompt, sel_raw = "默认策略（首轮不调用选择器）", None, None
                if rnd > 0 or self.select_first_round:
                    hint = verdict.strategy_hint if verdict else ""
                    missing = verdict.missing if verdict else ""
                    strategy, reason, sel_prompt, sel_raw = self.selector.select(
                        question, base, tried, missing, hint)
                    if strategy is None:
                        yield f"⚠️ 没有未试过的策略了（已试 {len(tried)} 种）"
                        terminated_by = "no_strategy_left"
                        break

                rctx = qt.round() if qt is not None else _NullCtx()
                with rctx as rt:
                    if rt is not None:
                        rt.strategy(strategy)
                        rt.selector(reason=reason, prompt=sel_prompt, raw=sel_raw)

                    # 执行层的最后一道防重复：选择器已经重选过，这里仍然要拦。
                    if strategy.key() in {kk for kk, _ in tried}:
                        if rt is not None:
                            rt.rejected_repeat()
                        yield f"⚠️ 策略重复，已拦截：{reason}"
                        terminated_by = "repeat_blocked"
                        break

                    yield f"⏳ 第{rnd + 1}轮 · {reason}"

                    t0 = time.perf_counter()
                    try:
                        got = self.pipeline._query_impl(
                            question, doc_id=doc_id, strategy=strategy)
                    except Exception as e:
                        if rt is not None:
                            rt.step("retrieval", input=strategy.query_text or question,
                                    error=f"{type(e).__name__}: {e}",
                                    duration_ms=(time.perf_counter() - t0) * 1000)
                        yield f"⚠️ 检索失败：{type(e).__name__}"
                        terminated_by = "error"
                        break
                    if rt is not None:
                        rt.step("retrieval", input=strategy.query_text or question,
                                output=[c.chunk_id for c in got],
                                duration_ms=(time.perf_counter() - t0) * 1000)

                    # 池子按 k × 轮数 上限累积；judge 和调用方各自只看前 k 个。
                    # 上限是为了压住规格第 3 节的"累积上下文污染"，但它必须大于 k，
                    # 否则累积会被自己截没。
                    cap = max(k * self.max_iterations, k + 10)
                    pool = self._rank_pool(question, _dedupe(pool, got), cap)
                    if rt is not None:
                        rt.chunks(pool)

                    # ── 判定够不够 ──────────────────────────────
                    t1 = time.perf_counter()
                    verdict = self.judge.judge(question, pool[:k])
                    if rt is not None:
                        rt.step("judge", input=f"{len(pool[:k])} chunks",
                                output=verdict.to_dict(),
                                duration_ms=(time.perf_counter() - t1) * 1000)
                        rt.verdict(verdict)

                    trace_lines.append(f"[{rnd + 1}] {reason} → {len(pool)} chunks, "
                                       f"sufficient={verdict.sufficient}")
                    tried.append((strategy.key(), verdict.missing or "(未说明)"))

                    if verdict.out_of_scope:
                        yield f"⚠️ 合同中似乎没有该条款（{verdict.missing[:60]}）"
                        terminated_by = "refused"
                        break
                    if verdict.sufficient:
                        yield f"✅ 上下文已充分，共 {len(pool)} 个片段，开始生成..."
                        terminated_by = "sufficient"
                        break
                    yield f"🔄 还缺：{(verdict.missing or '未说明')[:60]}"

            if terminated_by == "max_rounds":
                yield f"⚠️ 跑满 {self.max_iterations} 轮仍判定不充分，用现有 {len(pool)} 个片段生成"
            if qt is not None:
                qt.terminate(terminated_by)
                qt.set("n_final_chunks", len(pool[:k]))
                qt.set("tried_strategies", [kk for kk, _ in tried])

        # 这一句**必须在 `with qctx` 之外**。generator 的 with 块要等生成器被消费到
        # 这里才会退出；如果最终结果在 with 内部 yield，调用方拿到结果就丢掉生成器，
        # GeneratorExit 会从 with 里穿出去——而它继承自 BaseException，sink 的
        # `except Exception` 接不住，那条 trace 就没了。放在外面才保证先落盘再交付。
        trace_lines.append(f"terminated_by={terminated_by}")
        yield pool[:k], trace_lines

    def query(
        self,
        question: str,
        doc_id: str | None = None,
        k: int = 10,
        meta: dict | None = None,
    ) -> tuple[list[ChunkWindow], list[str]]:
        """普通调用（API / eval 用），不产生状态消息。"""
        for event in self.query_stream(question, doc_id=doc_id, k=k, meta=meta):
            if not isinstance(event, str):
                return event
        return [], [question]


class _NullCtx:
    """sink 未接入时的占位，让主循环不必到处写 if sink。"""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None
