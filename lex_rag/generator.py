"""
Generation 层：将检索到的 chunks 合成为带引用的自然语言答案。

用法：
    from lex_rag.generator import LegalGenerator
    gen = LegalGenerator(cfg.contextual)
    result = gen.generate(question, chunks)
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

from lex_rag.chunking import ChunkWindow
from lex_rag.config import ContextualConfig
from lex_rag.llm import ChatClient

# 供 structured_output="json_schema" 时做服务端强制约束用。
# json_object 模式下不发送——那时结构只靠 prompt 约束，_parse_response 的字段
# 缺省逻辑是唯一防线。
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "refused": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["refused", "answer"],
    "additionalProperties": False,
}

_GENERATE_PROMPT = """\
You are a legal contract analysis assistant. Answer questions based ONLY on the contract excerpts provided.

IMPORTANT — when to refuse:
Set "refused": true (and "answer": "") if:
- The exact information is not present in any of the excerpts
- You would need to infer or fabricate to answer
- The question asks whether a specific type of provision exists (non-compete, non-disparagement,
  termination for convenience, ...) and no excerpt contains such a provision. Absence of a
  provision is a refusal, NOT an answer of "No" — only answer "No" when an excerpt explicitly
  states the negative (e.g. "Distributor shall NOT be entitled to ...").
- A related-but-different clause is present. Quoting a clause that merely resembles the one
  asked about is a wrong answer, not a partial one.
When in doubt, refuse.

Only set "refused": false when the answer is explicitly stated in the excerpts.

Respond in JSON with exactly two fields:
- "refused": true or false
- "answer": your answer when not refused; empty string when refused

When "refused" is false:
- Your answer MUST consist only of verbatim quotes from the excerpts, cited with [N]
- Do NOT paraphrase, summarize, or add any words not present in the excerpts
- Yes/No questions: start with "Yes" or "No", then immediately quote the exact clause.
  Reminder: if neither the provision nor an explicit negative statement is in the excerpts,
  refuse instead of answering "No"
- Factual questions: quote the exact sentence(s) that contain the answer
- One paragraph maximum; no bullet lists
{multi_doc_note}
Examples:
Q: Does this contract contain a non-disparagement clause?
A: {{"refused": true, "answer": ""}}

Q: Does this contract allow termination for convenience?
   (excerpts only contain: "Either party may terminate this Agreement upon 30 days prior written
   notice upon the occurrence of any event of default")
A: {{"refused": true, "answer": ""}}

Q: What is the governing law of this contract?
A: {{"refused": false, "answer": "Illinois. \\"This Agreement is to be construed according to the laws of the State of Illinois\\" [1]."}}

Q: Does the contract include an exclusivity provision?
A: {{"refused": false, "answer": "Yes. \\"Company hereby appoints Distributor as its exclusive distributor in the Territory\\" [2]."}}

Contract excerpts:
{context}

Question: {question}"""

_MULTI_DOC_NOTE = """\
- Excerpts come from MULTIPLE contracts; cite each quote with [N] AND mention the contract name inline, e.g. "quote" [1] (CONTRACT_NAME)\
"""

@dataclass
class Citation:
    doc_id: str
    chunk_id: str
    start: int | None
    end: int | None
    excerpt: str          # chunk 文本前 120 字，方便展示
    num: int = 0          # 模型在答案中使用的引用编号 [N]


@dataclass
class GenerationResult:
    question: str
    answer: str                          # LLM 生成的答案；空字符串表示主动拒答
    citations: list[Citation] = field(default_factory=list)
    is_refused: bool = False             # True = 模型判断合同中无相关信息
    latency_ms: float = 0.0
    error: str | None = None             # 非 None 表示调用失败


class LegalGenerator:
    def __init__(self, cfg: ContextualConfig):
        self.cfg = cfg
        self._chat = ChatClient.from_config(cfg)

    def _meta_block(self, doc_id: str, meta: dict) -> str:
        lines = [f"[Contract: {doc_id}]"]
        for key, label in [
            ("contract_type", "Contract Type"),
            ("party_a",       "Party A"),
            ("party_b",       "Party B"),
            ("effective_date","Effective Date"),
            ("governing_law", "Governing Law"),
        ]:
            if meta.get(key):
                lines.append(f"{label}: {meta[key]}")
        if meta.get("key_clauses"):
            lines.append(f"Key Clauses: {', '.join(meta['key_clauses'])}")
        return "\n".join(lines)

    def _build_context(self, chunks: list[ChunkWindow],
                       meta: dict | None = None,
                       metas: dict[str, dict] | None = None) -> str:
        """
        metas 不为 None → 多文档模式：按 doc_id 分组，每组前置该合同 meta。
        meta  不为 None → 单文档模式（向后兼容）。
        """
        if metas is not None:
            doc_chunks: dict[str, list[tuple[int, ChunkWindow]]] = defaultdict(list)
            for i, chunk in enumerate(chunks, 1):
                doc_chunks[chunk.doc_id].append((i, chunk))
            parts = []
            for doc_id, indexed in doc_chunks.items():
                doc_meta = metas.get(doc_id)
                parts.append(self._meta_block(doc_id, doc_meta) if doc_meta
                              else f"[Contract: {doc_id}]")
                for i, chunk in indexed:
                    parts.append(f"[{i}] (pos: {chunk.start}-{chunk.end})\n{chunk.text}")
            return "\n\n".join(parts)

        # 单文档模式
        parts = []
        if meta:
            parts.append(self._meta_block(chunks[0].doc_id if chunks else "", meta))
        for i, chunk in enumerate(chunks, 1):
            header = f"[{i}] (doc: {chunk.doc_id}, pos: {chunk.start}-{chunk.end})"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n".join(parts)

    def _parse_response(self, data: dict, chunks: list[ChunkWindow]) -> tuple[str, bool, list[Citation]]:
        is_refused = bool(data.get("refused", False))
        if is_refused:
            return "", True, []

        answer = (data.get("answer") or "").strip()
        if not answer:
            return "", True, []

        citations: list[Citation] = []

        # 数字引用 [N]（单文档模式）
        nums = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
        for num in dict.fromkeys(nums):
            idx = num - 1
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                citations.append(Citation(
                    doc_id=chunk.doc_id, chunk_id=chunk.chunk_id,
                    start=chunk.start, end=chunk.end, excerpt=chunk.text[:120],
                    num=num,
                ))

        # 合同名引用 [Contract: DOC_ID]（多文档模式）
        if not citations:
            cited_docs = re.findall(r"\[Contract:\s*([^\]]+)\]", answer)
            chunk_by_doc: dict[str, ChunkWindow] = {}
            for c in chunks:
                chunk_by_doc.setdefault(c.doc_id, c)
            for doc_id in dict.fromkeys(d.strip() for d in cited_docs):
                if doc_id in chunk_by_doc:
                    chunk = chunk_by_doc[doc_id]
                    citations.append(Citation(
                        doc_id=chunk.doc_id, chunk_id=chunk.chunk_id,
                        start=chunk.start, end=chunk.end, excerpt=chunk.text[:120],
                    ))

        return answer, False, citations

    def _call_llm(self, prompt: str) -> dict:
        """JSON mode 调用，返回解析后的 dict。重试与 tracing 都在 ChatClient 里。"""
        return self._chat.complete_json(prompt, schema=_RESPONSE_SCHEMA,
                                        trace_name="generator.generate")

    def generate_stream(
        self,
        question: str,
        chunks: list[ChunkWindow],
        meta: dict | None = None,
        metas: dict[str, dict] | None = None,
    ) -> Iterator[str | GenerationResult]:
        """
        流式生成。先 yield str（partial answer token），最后 yield GenerationResult。
        调用方检查 isinstance(item, GenerationResult) 判断结束。

        实现：用 JSON mode streaming，状态机从流式 JSON token 中提取 answer 字段内容。
        """
        if not chunks:
            yield GenerationResult(
                question=question, answer="", is_refused=True,
                error="no chunks retrieved",
            )
            return

        is_multi = metas is not None and len({c.doc_id for c in chunks}) > 1
        context = self._build_context(chunks, meta=meta, metas=metas)
        multi_doc_note = _MULTI_DOC_NOTE + "\n" if is_multi else ""
        prompt = _GENERATE_PROMPT.format(
            context=context, question=question, multi_doc_note=multi_doc_note
        )

        t0 = time.perf_counter()
        full_text = ""

        # 状态机状态
        # SCAN    — 未找到 "answer": " 前缀
        # STREAM  — 正在输出 answer 内容
        # DONE    — 遇到非转义引号，answer 结束
        state = "SCAN"
        answer_prefix = '"answer": "'
        scan_buf = ""        # 用于在流式 token 中匹配前缀
        escaped = False      # 上一个字符是否为反斜杠

        try:
            for token in self._chat.stream(prompt, json_mode=True, schema=_RESPONSE_SCHEMA,
                                           trace_name="generator.generate_stream"):
                full_text += token

                if state == "DONE":
                    continue

                if state == "SCAN":
                    scan_buf += token
                    # 检查 scan_buf 是否包含 answer 字段开头
                    idx = scan_buf.find(answer_prefix)
                    if idx != -1:
                        # answer 内容从 idx + len(answer_prefix) 开始
                        remaining = scan_buf[idx + len(answer_prefix):]
                        state = "STREAM"
                        scan_buf = ""
                        # 处理剩余部分
                        token = remaining
                        # fall through to STREAM handling below

                if state == "STREAM":
                    out = []
                    for ch in token:
                        if escaped:
                            # 转义字符：输出实际字符（去掉反斜杠转义）
                            out.append(ch)
                            escaped = False
                        elif ch == "\\":
                            escaped = True
                        elif ch == '"':
                            # answer 字段结束
                            state = "DONE"
                            break
                        else:
                            out.append(ch)
                    if out:
                        yield "".join(out)

        except Exception as e:
            yield GenerationResult(
                question=question, answer="", is_refused=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=str(e),
            )
            return

        latency_ms = (time.perf_counter() - t0) * 1000

        # 用完整响应文本做最终解析（citations、refused 判断）
        try:
            data = json.loads(full_text or "{}")
        except json.JSONDecodeError:
            data = {}

        answer, is_refused, citations = self._parse_response(data, chunks)
        yield GenerationResult(
            question=question,
            answer=answer,
            citations=citations,
            is_refused=is_refused,
            latency_ms=latency_ms,
        )

    def generate(self, question: str, chunks: list[ChunkWindow],
                 meta: dict | None = None,
                 metas: dict[str, dict] | None = None) -> GenerationResult:
        """
        question + top-k chunks → GenerationResult。
        meta  — 单文档 doc_meta（doc_id 已知时使用）。
        metas — 多文档 {doc_id: meta}（corpus 查询时使用，优先于 meta）。
        """
        if not chunks:
            return GenerationResult(
                question=question,
                answer="",
                is_refused=True,
                error="no chunks retrieved",
            )

        is_multi = metas is not None and len({c.doc_id for c in chunks}) > 1
        context = self._build_context(chunks, meta=meta, metas=metas)
        multi_doc_note = _MULTI_DOC_NOTE + "\n" if is_multi else ""
        prompt = _GENERATE_PROMPT.format(
            context=context, question=question, multi_doc_note=multi_doc_note
        )

        t0 = time.perf_counter()
        try:
            data = self._call_llm(prompt)
        except Exception as e:
            return GenerationResult(
                question=question,
                answer="",
                is_refused=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=str(e),
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        answer, is_refused, citations = self._parse_response(data, chunks)

        return GenerationResult(
            question=question,
            answer=answer,
            citations=citations,
            is_refused=is_refused,
            latency_ms=latency_ms,
        )
