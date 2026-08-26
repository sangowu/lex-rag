"""
sufficiency_judge —— 回答"当前这堆 chunk 够不够回答这个问题"。

对应 `docs/agentic_loop_upgrade.md` 第 2.3 节。消费者是 2.4 的**策略选择器**，
它读 `missing_kind` 决定下一轮换什么检索策略。

曾经还有第二个候选消费者：生成层的两段式校验（读 `out_of_scope` /
`answer_supported`）。`scripts/ab_sufficiency.py` 的 200 条 A/B 把它否掉了，
见 `docs/experiments.md`。两条实测结论：

* 校验环节 200 条里只干预 3 次且全错，多一次调用换来每个指标都变差；
* **给判定器看草稿会让它被锚定**——无答案样本的 `out_of_scope` 召回从 0.733
  掉到 0.433。它倾向于相信一份言之凿凿的草稿。

所以生产路径只用 `mode="sufficiency"`。`mode="unified"` 保留下来是为了让那次
A/B 还能复跑（删了实验就不可复现），**不要接进生成层**。两种 prompt 共用同一套
解析与容错——当初这么写是为了让比较只落在 prompt 和用法上，否则结论不成立。

**`missing_kind` 是枚举而不是自由文本**，因为它的读者是代码不是人。让选择器
去正则匹配一句英文描述，等于把决策质量押在措辞上；自由文本仍然保留在
`missing` 字段里，供 trace 和人工复盘用。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Literal

from lex_rag.chunking import ChunkWindow
from lex_rag.config import ContextualConfig
from lex_rag.llm import ChatClient

# 缺失类型 → 下一轮该换什么策略（规格 2.3 的映射表）。
# 选择器直接查这张表，不解析自然语言。
MissingKind = Literal[
    "exact_term",        # 缺具体金额 / 条款编号 → BM25 精确匹配
    "clause_context",    # 命中了条款但上下文被截断 → parent 粒度
    "concept_mismatch",  # 问题的说法与合同用词对不上 → HyDE
    "multi_aspect",      # 问题涉及多个方面，一路检索覆盖不全 → multi-query
    "none",              # 不缺
]

STRATEGY_HINT: dict[str, str] = {
    "exact_term":       "bm25",
    "clause_context":   "parent",
    "concept_mismatch": "hyde",
    "multi_aspect":     "multi_query",
    "none":             "",
}

_VALID_KINDS = set(STRATEGY_HINT)

_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient":   {"type": "boolean"},
        "missing":      {"type": "string"},
        "missing_kind": {"type": "string", "enum": sorted(_VALID_KINDS)},
        "out_of_scope": {"type": "boolean"},
        "confidence":   {"type": "number"},
    },
    "required": ["sufficient", "missing", "missing_kind", "out_of_scope", "confidence"],
    "additionalProperties": False,
}

_UNIFIED_SCHEMA = {
    "type": "object",
    "properties": {
        **_SCHEMA["properties"],
        "answer_supported": {"type": "boolean"},
    },
    "required": _SCHEMA["required"] + ["answer_supported"],
    "additionalProperties": False,
}

_KIND_TABLE = """\
- "exact_term"       the answer hinges on a specific figure, date, section number or defined
                     term that does not appear in any excerpt
- "clause_context"   a relevant clause IS present but is cut off mid-sentence, or refers to
                     another section ("as defined in Section 8") that is not included
- "concept_mismatch" the excerpts are about other topics entirely; the contract likely words
                     this concept differently than the question does
- "multi_aspect"     the question asks about several things at once and the excerpts cover
                     only some of them
- "none"             nothing is missing"""

_SUFFICIENCY_PROMPT = """\
You are auditing whether a set of contract excerpts is sufficient to answer a question.
You are NOT answering the question. Judge only what is present in the excerpts.

Question: {question}

Contract excerpts:
{context}

Decide three things:

1. "sufficient" — can the question be answered using ONLY these excerpts, quoting them
   verbatim? True only if the text that answers it is literally present.
2. "out_of_scope" — is this a question about a provision that this contract simply does not
   contain? Distinguish carefully from (1): "the excerpts do not include it" means insufficient
   context, while "this contract has no such provision at all" means out of scope. Set true only
   when the excerpts cover the relevant part of the contract and the provision is still absent.
3. "missing_kind" — if not sufficient, which of these best describes the gap:
{kinds}

Reply with JSON only:
{{"sufficient": bool, "missing": "one sentence, empty if sufficient",
  "missing_kind": "one of the labels above", "out_of_scope": bool,
  "confidence": 0.0-1.0}}"""

_UNIFIED_PROMPT = """\
You are auditing a draft answer produced from contract excerpts.
You are NOT rewriting the answer. Judge only what is present in the excerpts.

Question: {question}

Contract excerpts:
{context}

Draft answer: {draft}

Decide four things:

1. "answer_supported" — is every claim in the draft answer quoted from, or directly stated by,
   the excerpts? A draft that quotes a related-but-different clause is NOT supported. A draft
   that refuses is trivially supported (set true).
2. "sufficient" — could the question be answered using ONLY these excerpts, quoting them
   verbatim? Answer this independently of what the draft did — a draft that refused may well
   have been wrong to refuse.
3. "out_of_scope" — is this a question about a provision that this contract simply does not
   contain? Set true only when the excerpts cover the relevant part of the contract and the
   provision is still absent. An unsupported draft answer is evidence for this.
4. "missing_kind" — if not sufficient, which of these best describes the gap:
{kinds}

Reply with JSON only:
{{"answer_supported": bool, "sufficient": bool,
  "missing": "one sentence, empty if sufficient",
  "missing_kind": "one of the labels above", "out_of_scope": bool,
  "confidence": 0.0-1.0}}"""


@dataclass(frozen=True)
class Verdict:
    """一次充分性判定。frozen —— 它要被写进 trace，事后不该被改。"""

    sufficient: bool = False
    missing: str = ""
    missing_kind: str = "none"
    out_of_scope: bool = False
    confidence: float = 0.0
    answer_supported: bool | None = None   # 仅 unified 模式有值
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def strategy_hint(self) -> str:
        """下一轮该换什么。空字符串 = 没有建议。"""
        return STRATEGY_HINT.get(self.missing_kind, "")

    def to_dict(self) -> dict:
        return {
            "sufficient": self.sufficient,
            "missing": self.missing,
            "missing_kind": self.missing_kind,
            "strategy_hint": self.strategy_hint,
            "out_of_scope": self.out_of_scope,
            "confidence": self.confidence,
            "answer_supported": self.answer_supported,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


def _coerce(data: dict, unified: bool) -> Verdict:
    """把模型输出收敛成 Verdict。

    json_object 模式只保证语法合法、不保证字段齐全，所以每个字段都要有缺省。
    缺省方向是刻意选的：**缺字段时按"不充分"处理**——判成不够最多白跑一轮，
    判成够了会直接停止检索、拿着残缺上下文去生成，后者的代价大得多。
    """
    kind = str(data.get("missing_kind", "none")).strip().lower()
    if kind not in _VALID_KINDS:
        kind = "none"

    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(max(conf, 0.0), 1.0)

    supported = None
    if unified:
        # 同样按危险方向缺省：字段缺失时不认为"已被支持"。
        supported = bool(data.get("answer_supported", False))

    return Verdict(
        sufficient=bool(data.get("sufficient", False)),
        missing=str(data.get("missing", "") or "").strip(),
        missing_kind=kind,
        out_of_scope=bool(data.get("out_of_scope", False)),
        confidence=conf,
        answer_supported=supported,
    )


class SufficiencyJudge:
    """上下文充分性判定器。

    mode="sufficiency" —— 只看问题 + 上下文（规格 2.3 的本体，A 臂）
    mode="unified"     —— 额外看一份草稿答案，兼做生成层校验（B 臂）
    """

    def __init__(self, cfg: ContextualConfig, *, mode: str = "sufficiency",
                 max_context_chars: int = 12000) -> None:
        if mode not in ("sufficiency", "unified"):
            raise ValueError(f"未知 mode: {mode!r}")
        self.cfg = cfg
        self.mode = mode
        self.max_context_chars = max_context_chars
        self._chat = ChatClient.from_config(cfg)

    @property
    def unified(self) -> bool:
        return self.mode == "unified"

    def _context(self, chunks: list[ChunkWindow]) -> str:
        parts = [f"[{i}] {c.text}" for i, c in enumerate(chunks, 1)]
        text = "\n\n".join(parts)
        # 截断是为了控成本；从尾部截掉，靠前的是 reranker 认为最相关的。
        return text[: self.max_context_chars]

    def _prompt(self, question: str, chunks: list[ChunkWindow],
                draft_answer: str | None) -> tuple[str, dict]:
        if self.unified:
            draft = (draft_answer or "").strip() or "(the model refused to answer)"
            return _UNIFIED_PROMPT.format(
                question=question, context=self._context(chunks),
                draft=draft, kinds=_KIND_TABLE,
            ), _UNIFIED_SCHEMA
        return _SUFFICIENCY_PROMPT.format(
            question=question, context=self._context(chunks), kinds=_KIND_TABLE,
        ), _SCHEMA

    def judge(self, question: str, chunks: list[ChunkWindow],
              draft_answer: str | None = None) -> Verdict:
        """返回 Verdict。**不抛异常**——判定器崩了不该让整轮循环中断。"""
        if not chunks:
            return Verdict(sufficient=False, missing="no chunks retrieved",
                           missing_kind="concept_mismatch", confidence=1.0)

        prompt, schema = self._prompt(question, chunks, draft_answer)

        t0 = time.perf_counter()
        try:
            data = self._chat.complete_json(
                prompt, schema=schema, trace_name=f"judge.{self.mode}",
            )
        except Exception as e:
            # 调用失败按"不充分且无建议"返回，让上层照常继续；error 落进 trace。
            return Verdict(missing=f"judge failed: {type(e).__name__}",
                           latency_ms=(time.perf_counter() - t0) * 1000,
                           error=str(e))

        return replace(_coerce(data, unified=self.unified),
                       latency_ms=(time.perf_counter() - t0) * 1000)
