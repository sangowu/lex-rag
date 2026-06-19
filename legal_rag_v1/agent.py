"""
Agentic 迭代检索：当初次检索无结果时，用 LLM 重写查询并重试。

用法：
    from legal_rag_v1.agent import AgenticPipeline
    agent = AgenticPipeline(pipeline, cfg.contextual)

    # 带实时状态的流式调用（UI 用）
    for event in agent.query_stream("What is the governing law?", doc_id="..."):
        if isinstance(event, str):
            print(event)          # 状态消息
        else:
            chunks, trace = event # 最终结果

    # 普通调用（API / eval 用）
    chunks, trace = agent.query("What is the governing law?", doc_id="...")
"""
from __future__ import annotations

import time
from typing import Iterator

from legal_rag_v1.chunking import ChunkWindow
from legal_rag_v1.config import ContextualConfig
from legal_rag_v1.pipeline import RAGPipeline

_REWRITE_PROMPT = """\
A semantic search over legal contract text failed to return relevant results for the following query.

Original question: {question}
Search query used: {query}

Rewrite the search query to improve recall. Rules:
- Use legal synonyms or alternative phrasing for the key concept
- Prefer short, keyword-focused phrases over full sentences
- Do NOT add qualifiers like "in the contract" or "according to the agreement"
- Return ONLY the rewritten query, no explanation

Rewritten query:"""


class AgenticPipeline:
    """
    在 RAGPipeline 外包一层迭代检索逻辑。

    query_stream() 流程：
      1. yield 状态消息（str）供 UI 实时展示
      2. 用原始问题检索；若无结果，LLM 重写查询并重试
      3. 最多重试 max_iterations 次
      4. 最后 yield (chunks, query_trace) 作为最终结果
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        cfg: ContextualConfig,
        max_iterations: int = 2,
    ) -> None:
        self.pipeline = pipeline
        self.cfg = cfg
        self.max_iterations = max_iterations
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.cfg.api_key)
        return self._client

    def _rewrite_query(self, question: str, failed_query: str) -> str:
        prompt = _REWRITE_PROMPT.format(question=question, query=failed_query)
        try:
            client = self._get_client()
            resp = client.models.generate_content(model=self.cfg.model, contents=prompt)
            rewritten = (resp.text or "").strip()
            return rewritten if rewritten and rewritten != failed_query else failed_query
        except Exception:
            return failed_query

    def query_stream(
        self,
        question: str,
        doc_id: str | None = None,
        k: int = 10,
    ) -> Iterator[str | tuple[list[ChunkWindow], list[str]]]:
        """
        生成器：先 yield str 状态消息，最后 yield (chunks, query_trace)。
        调用方用 isinstance(event, str) 区分状态消息和最终结果。
        """
        seen_ids: set[str] = set()
        all_chunks: list[ChunkWindow] = []
        query_trace: list[str] = [question]

        for iteration in range(self.max_iterations + 1):
            current_query = query_trace[-1]
            label = f"第{iteration + 1}次" if self.max_iterations > 0 else ""
            yield f"⏳ {label}检索：{current_query}"

            chunks = self.pipeline.query(current_query, doc_id=doc_id, k=k)

            for c in chunks:
                if c.chunk_id not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c.chunk_id)

            if chunks:
                yield f"✅ 找到 {len(all_chunks)} 个相关片段，开始生成..."
                break

            if iteration == self.max_iterations:
                yield "⚠️ 所有查询均无结果"
                break

            yield "🔄 无结果，重写查询中..."
            new_query = self._rewrite_query(question, current_query)
            if new_query == current_query:
                yield "⚠️ 查询无法进一步优化"
                break
            query_trace.append(new_query)
            time.sleep(0.1)

        yield all_chunks[:k], query_trace

    def query(
        self,
        question: str,
        doc_id: str | None = None,
        k: int = 10,
    ) -> tuple[list[ChunkWindow], list[str]]:
        """普通调用（API / eval 脚本用），不产生状态消息。"""
        for event in self.query_stream(question, doc_id=doc_id, k=k):
            if not isinstance(event, str):
                return event
        return [], [question]
