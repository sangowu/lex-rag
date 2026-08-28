"""
OpenAI 兼容的对话补全封装 —— 项目里所有 LLM 调用的唯一入口。

实际用哪个模型由 `config.yaml` 的 `contextual.base_url` / `contextual.model` 决定
（当前是 DashScope 的 qwen3.7-flash），key 取自 `GENERATE_MODEL_API`。换服务商只改
配置，不需要动任何调用方——前提是对方兼容 OpenAI 的 `/chat/completions`。

下面两条差异是从 Gemini 迁移时发现的，对所有 OpenAI 风格的服务商都成立：

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
from dataclasses import dataclass
from typing import Any

from lex_rag import tracing

# 仅当调用方没给 base_url/model 时兜底。正常路径都从 config.yaml 来，
# 这里保持与当前配置一致，避免"默认值和实际配置矛盾"这种排查陷阱。
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-flash"


class LLMError(RuntimeError):
    """重试耗尽后仍失败。"""


@dataclass(frozen=True)
class Usage:
    """一次调用的 token 用量。字段缺失时一律是 0，不是 None。

    用 0 而不是 None 是为了让求和不必到处判空——评测里 usage 缺失和用量为 0 的
    区别不重要，"能不能直接加起来"才重要。要区分"服务端没给"的场景看 `reported`。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    # thinking 模型把思考链单独计在 completion_tokens_details.reasoning_tokens 里。
    # 它**已经包含在** completion_tokens 内，是其中的一部分，不要再加一遍。
    reasoning_tokens: int = 0
    reported: bool = False          # 服务端到底给没给 usage

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
            self.reported or other.reported,
        )


def _usage_of(resp: Any) -> Usage:
    """从 OpenAI 风格响应里安全提取 usage。字段缺失是常态，绝不抛。"""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return Usage()
    def _int(obj: Any, name: str) -> int:
        v = getattr(obj, name, None)
        return int(v) if isinstance(v, (int, float)) else 0
    details = getattr(usage, "completion_tokens_details", None)
    return Usage(
        prompt_tokens=_int(usage, "prompt_tokens"),
        completion_tokens=_int(usage, "completion_tokens"),
        reasoning_tokens=_int(details, "reasoning_tokens") if details is not None else 0,
        reported=True,
    )


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
        thinking_style: str = "auto",
        structured_output: str = "json_object",
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
        # 各家的传参形式不同，见 _extra_body()。auto = 按 base_url 推断
        self.thinking_style = thinking_style
        # json_object = 只保证语法合法；json_schema = 服务端强制结构（支持的模型有限）
        self.structured_output = structured_output
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
            thinking_style=getattr(cfg, "thinking_style", "auto"),
            structured_output=getattr(cfg, "structured_output", "json_object"),
        )

    def _backoff_sec(self, error: Exception, attempt: int) -> float:
        """限流类错误退避得更狠：普通错误 2^n，429 用 4^n（上限 60s）。

        Z.ai 的 429 有两种：1302 配额限流、1305 服务过载。两者都是"等一会儿就好"，
        但 2/4/8 秒这种退避对 1305 明显不够——实测评测跑到 judge 阶段仍被打断。
        """
        msg = str(error)
        is_rate_limited = (
            type(error).__name__ == "RateLimitError" or "429" in msg or "rate limit" in msg.lower()
        )
        if is_rate_limited:
            return min(self.retry_backoff_sec * (4 ** attempt), 60.0)
        return self.retry_backoff_sec * (2 ** attempt)

    def _resolve_thinking_style(self) -> str:
        """按 base_url 推断思考开关的传参形式。显式配置优先。"""
        if self.thinking_style != "auto":
            return self.thinking_style
        host = self.base_url.lower()
        if "dashscope" in host or "aliyuncs" in host:
            return "dashscope"
        if "z.ai" in host or "bigmodel" in host:
            return "zai"
        return "none"

    def _extra_body(self) -> dict:
        """思考链开关。各家参数名不同，语义相同：

          Z.ai       thinking={"type": "enabled"|"disabled"}
          DashScope  enable_thinking=true|false

        两家都默认开启，而本项目是抽取式问答（逐字引用合同条款）——实测同一
        问题开着要 376 个输出 token（其中 1366 字符是 reasoning_content），
        关掉只要 28 个。但它也不是纯浪费：开着能显著改善"条款不存在"的判断，
        代价是误拒上升，取舍见 docs/experiments.md。

        thinking 为 None 时整个字段不发送，保证换到不认识它的服务商也不会 400。
        """
        if self.thinking is None:
            return {}
        style = self._resolve_thinking_style()
        if style == "dashscope":
            return {"extra_body": {"enable_thinking": bool(self.thinking)}}
        if style == "zai":
            return {"extra_body": {"thinking": {"type": "enabled" if self.thinking else "disabled"}}}
        return {}

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
        return self._client

    # ── 非流式 ──────────────────────────────────────────────────

    def _response_format(self, json_mode: bool, schema: dict | None) -> dict:
        """构造 response_format。

        json_object 只保证语法是合法 JSON，不保证字段齐全；json_schema + strict
        由服务端强制结构，是最接近 Gemini response_schema 的东西——但支持的模型
        有限（DashScope 上仅 Qwen3.7/3.8 系列），所以由配置显式选择，不做自动降级：
        静默降级会让人以为拿到了强约束，实际没有。
        """
        if not json_mode:
            return {}
        if self.structured_output == "json_schema" and schema:
            return {"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "structured_answer", "strict": True, "schema": schema},
            }}
        return {"response_format": {"type": "json_object"}}

    def complete_with_usage(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        schema: dict | None = None,
        trace_name: str = "llm.generate",
    ) -> tuple[str, Usage]:
        """一次补全，返回 (文本, token 用量)。重试耗尽后抛 LLMError。

        usage 一直都取到了，但此前只喂给 Langfuse——没配 key 时 tracing 是 no-op，
        于是本地评测**完全看不到 token**。关掉 thinking 那次改动的收益里，
        "省了多少钱"就因此报不出来，只能靠延迟去猜。所以把它一路带回调用方。
        """
        gen = tracing.start_generation(trace_name, self.model, prompt)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **self._response_format(json_mode, schema),
                    **self._extra_body(),
                )
                text = (resp.choices[0].message.content or "").strip()
                in_tok, out_tok = tracing.chat_usage(resp)
                tracing.end_generation(gen, output=text, input_tokens=in_tok, output_tokens=out_tok)
                return text, _usage_of(resp)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self._backoff_sec(e, attempt))
        tracing.end_generation(gen, output="<error>")
        raise LLMError(f"{self.model} 调用重试 {self.max_retries} 次后仍失败：{last_error}") from last_error

    def complete(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        schema: dict | None = None,
        trace_name: str = "llm.generate",
    ) -> str:
        """一次补全，返回文本。重试耗尽后抛 LLMError。"""
        return self.complete_with_usage(
            prompt, json_mode=json_mode, schema=schema, trace_name=trace_name
        )[0]

    def complete_json_with_usage(self, prompt: str, *, schema: dict | None = None,
                                 trace_name: str = "llm.generate") -> tuple[dict, Usage]:
        """JSON mode 补全，附带 token 用量。解析规则同 `complete_json`。"""
        text, usage = self.complete_with_usage(
            prompt, json_mode=True, schema=schema, trace_name=trace_name
        )
        return _loads_or_empty(text), usage

    def complete_json(self, prompt: str, *, schema: dict | None = None,
                      trace_name: str = "llm.generate") -> dict:
        """JSON mode 补全。**解析失败返回 {}**，由调用方的字段缺省逻辑兜底。

        不抛异常是有意的：json_object 模式只保证语法合法、不保证字段齐全，
        把"模型少给了个字段"升级成异常会让整批评测中断，得不偿失。
        """
        return self.complete_json_with_usage(prompt, schema=schema, trace_name=trace_name)[0]

    # ── 流式 ────────────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        schema: dict | None = None,
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
                **self._response_format(json_mode, schema),
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
