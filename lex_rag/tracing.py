"""Langfuse 可观测性封装（Langfuse Cloud，Python SDK v4）。

设计原则 —— 失败安全（与 reranker fallback 同哲学）：
- 未配置 ``LANGFUSE_PUBLIC_KEY`` 时完全 no-op，零开销、零外部依赖，不影响主流程与 CI；
- 任何 Langfuse 调用异常都被吞掉，绝不因追踪问题中断真实 LLM 调用；
- 通过 ``atexit`` 在进程退出前 flush，保证 CLI / eval 等短命进程数据不丢失。

用法（在 generator 内）::

    gen = start_generation("gemini.generate", model, prompt)
    resp = ...                       # 真实 LLM 调用，永远执行，不受追踪影响
    end_generation(gen, output=text, input_tokens=..., output_tokens=...)

父级 span（在 serve/pipeline 内）把一次查询的检索 + 生成聚合成一棵 trace 树::

    with trace_span("lex_rag.query", question):
        chunks = pipeline.query(question)      # 内部 trace_span("lex_rag.retrieval")
        result = generator.generate(...)       # 内部 start/end_generation

延迟（latency）由 Langfuse 依据 generation 的 start/end 时间自动计算，无需手动记录。
"""

from __future__ import annotations

import atexit
import contextlib
import os
from collections.abc import Iterator
from typing import Any

# 惰性启用：不在 import 时快照 env，而在首次调用时读取。
# 原因：本项目 load_dotenv() 由 load_config() 在运行时调用，晚于本模块 import；
# 若 import 时快照，则 .env 中的 key 尚未加载，会误判为禁用。
# _client 三态：None=未初始化 / False=已确定禁用 / 其他=Langfuse client 实例。
_client: Any = None
_flush_registered = False


def _get_client() -> Any:
    """惰性获取 Langfuse client；未配置/初始化失败时返回 None（并记住禁用状态）。"""
    global _client, _flush_registered
    if _client is False:
        return None
    if _client is not None:
        return _client
    # key 尚未出现：**不** latch 禁用——它可能在稍后的 load_dotenv() 后才可见
    # （如 serve 的首个请求：trace_span 早于请求内的 load_config）。下次调用会重试。
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    try:
        from langfuse import get_client

        _client = get_client()
        if not _flush_registered:
            atexit.register(flush)
            _flush_registered = True
    except Exception:
        _client = False  # 仅真正的初始化失败才 latch 禁用，避免反复重试坏 client
        return None
    return _client


def start_generation(name: str, model: str, input_data: Any) -> Any:
    """开始一次 LLM 调用追踪，返回 generation 句柄；未启用/出错时返回 None。"""
    client = _get_client()
    if client is None:
        return None
    try:
        return client.start_observation(
            as_type="generation", name=name, model=model, input=input_data
        )
    except Exception:
        return None


def end_generation(
    gen: Any,
    output: Any = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """结束追踪并记录 output 与 token 用量。gen 为 None 时静默跳过。"""
    if gen is None:
        return
    try:
        usage: dict[str, int] = {}
        if input_tokens is not None:
            usage["input_tokens"] = input_tokens
        if output_tokens is not None:
            usage["output_tokens"] = output_tokens
        gen.update(output=output, usage_details=usage or None)
        gen.end()
    except Exception:
        pass


@contextlib.contextmanager
def trace_span(name: str, input_data: Any = None) -> Iterator[None]:
    """父级 span context manager：包裹一段工作，其内部的 generation 会自动嵌套成 trace 树。

    失败安全：未启用或出错时退化为无操作的 with 块，不影响被包裹的业务逻辑。
    """
    client = _get_client()
    span = None
    if client is not None:
        try:
            span = client.start_as_current_observation(
                as_type="span", name=name, input=input_data
            )
            span.__enter__()
        except Exception:
            span = None
    try:
        yield
    finally:
        if span is not None:
            with contextlib.suppress(Exception):
                span.__exit__(None, None, None)


def genai_usage(resp: Any) -> tuple[int | None, int | None]:
    """从 google-genai 响应中安全提取 (prompt_tokens, output_tokens)。

    字段缺失或结构异常时返回 (None, None)，绝不抛异常。
    """
    try:
        meta = getattr(resp, "usage_metadata", None)
        if meta is None:
            return None, None
        return (
            getattr(meta, "prompt_token_count", None),
            getattr(meta, "candidates_token_count", None),
        )
    except Exception:
        return None, None


def flush() -> None:
    """刷新缓冲。atexit 自动注册；也可在进程收尾时手动调用。"""
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
