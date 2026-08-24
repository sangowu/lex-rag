"""
OpenAI 兼容的对话补全封装 —— 项目里所有 LLM 调用的唯一入口。

当前指向 Z.ai 的 GLM（`https://api.z.ai/api/paas/v4/`，key 取自 `GENERATE_MODEL_API`）。
换服务商只需改 `config.yaml` 的 `contextual.base_url` / `contextual.model`，
不需要动任何调用方——前提是对方兼容 OpenAI 的 `/chat/completions`。

从 Gemini 迁移过来时有两点行为差异，调用方需要知道：

1. **结构化输出的强制力变弱了。** Gemini 用 `response_schema` 在服务端强制 JSON
   结构；OpenAI 风格只有 `response_format={"type": "json_object"}`，只保证语法是
   合法 JSON，不保证字段齐全。所以 `complete_json()` 解析失败时返回 `{}` 而不是抛，
   由调用方的字段缺省逻辑兜底（`generator._parse_response` 本来就按缺省处理）。
   注意 json_object 模式要求 prompt 里出现 "json" 字样，本项目的 prompt 都满足。

2. **流式 usage 需要显式索取。** OpenAI 风格默认不在流里带 token 统计，要靠
   `stream_options={"include_usage": True}`；服务端不支持时静默拿不到，
   tracing 侧按 None 处理，不影响主流程。

用法::

    chat = ChatClient.from_config(cfg.contextual)
    text = chat.complete("...")                       # 纯文本
    data = chat.complete_json("...")                  # JSON mode → dict
    for token in chat.stream("...", json_mode=True):  # 流式
        ...
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from lex_rag import tracing

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4/"
DEFAULT_MODEL = "glm-4.7-flash"


class LLMError(RuntimeError):
    """重试耗尽后仍失败。"""


class ChatClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
        timeout: float = 120.0,
        thinking: bool | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.timeout = timeout
        # None = 不发这个字段（用服务端默认，也兼容不认识它的其他服务商）
        # False = 关闭思考链；True = 显式开启
        self.thinking = thinking
        self._client: Any = None      # 懒加载，import lex_rag.llm 本身不该有副作用

    @classmethod
    def from_config(cls, cfg: Any, *, max_retries: int | None = None) -> "ChatClient":
        """从 ContextualConfig / RagasConfig 构造（两者字段不完全一致，缺的用默认值）。

        ``max_retries=0`` 用于调用方自己已经有重试循环的场景（contextualizer 的几个
        类都是），避免内外两层重试相乘成 (n+1)² 次请求。
        """
        return cls(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=getattr(cfg, "base_url", DEFAULT_BASE_URL),
            max_retries=getattr(cfg, "max_retries", 3) if max_retries is None else max_retries,
            retry_backoff_sec=getattr(cfg, "retry_backoff_sec", 2.0),
            thinking=getattr(cfg, "thinking", None),
        )

    def _extra_body(self) -> dict:
        """Z.ai 的 thinking 开关。GLM-4.5+ 默认 enabled，实测同一个问题
        开着要 376 个输出 token（其中 1366 字符是 reasoning_content），
        关掉只要 28 个——对本项目的抽取式问答，思考链纯属浪费。

        thinking 为 None 时整个字段不发送，保证换到不认识它的服务商也不会 400。
        """
        if self.thinking is None:
            return {}
        return {"extra_body": {"thinking": {"type": "enabled" if self.thinking else "disabled"}}}

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
        return self._client

    # ── 非流式 ──────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        trace_name: str = "llm.generate",
    ) -> str:
        """一次补全，返回文本。重试耗尽后抛 LLMError。"""
        gen = tracing.start_generation(trace_name, self.model, prompt)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **({"response_format": {"type": "json_object"}} if json_mode else {}),
                    **self._extra_body(),
                )
                text = (resp.choices[0].message.content or "").strip()
                in_tok, out_tok = tracing.chat_usage(resp)
                tracing.end_generation(gen, output=text, input_tokens=in_tok, output_tokens=out_tok)
                return text
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_sec * (2 ** attempt))
        tracing.end_generation(gen, output="<error>")
        raise LLMError(f"{self.model} 调用重试 {self.max_retries} 次后仍失败：{last_error}") from last_error

    def complete_json(self, prompt: str, *, trace_name: str = "llm.generate") -> dict:
        """JSON mode 补全。**解析失败返回 {}**，由调用方的字段缺省逻辑兜底。

        不抛异常是有意的：json_object 模式只保证语法合法、不保证字段齐全，
        把"模型少给了个字段"升级成异常会让整批评测中断，得不偿失。
        """
        return _loads_or_empty(self.complete(prompt, json_mode=True, trace_name=trace_name))

    # ── 流式 ────────────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        trace_name: str = "llm.stream",
    ) -> Iterator[str]:
        """流式补全，逐个 yield token 文本。

        全程只调一次，不重试：token 已经吐给调用方了，重试会让前半段重复出现。
        结束时把完整文本与 token 用量写进 tracing。
        """
        gen = tracing.start_generation(trace_name, self.model, prompt)
        full_text = ""
        in_tok: int | None = None
        out_tok: int | None = None
        try:
            stream = self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
                **self._extra_body(),
            )
            for chunk in stream:
                # 带 usage 的那个 chunk 通常没有 choices，先取 usage 再取 delta
                _in, _out = tracing.chat_usage(chunk)
                if _in is not None:
                    in_tok = _in
                if _out is not None:
                    out_tok = _out
                if not getattr(chunk, "choices", None):
                    continue
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_text += token
                    yield token
        except Exception:
            tracing.end_generation(gen, output="<error>")
            raise
        tracing.end_generation(gen, output=full_text, input_tokens=in_tok, output_tokens=out_tok)


def _loads_or_empty(raw: str) -> dict:
    """解析模型返回的 JSON；带 ``` 围栏时先剥掉。失败返回 {}。"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
