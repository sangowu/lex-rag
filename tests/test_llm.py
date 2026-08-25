"""ChatClient 单元测试（网络调用 mock 掉）。

重点覆盖从 Gemini 迁移过来时行为发生变化的地方：JSON mode 的开关方式、
解析失败不抛异常、流式的 usage-only chunk、以及 from_config 的重试覆盖
（内外两层重试相乘是这次重构最容易埋下的性能坑）。
"""

from unittest.mock import MagicMock, patch

import pytest

from lex_rag.config import ContextualConfig
from lex_rag.llm import ChatClient, LLMError, _loads_or_empty


def _cfg(**kw) -> ContextualConfig:
    base = dict(
        enabled=True, model="glm-test", api_key="k", rpm_limit=60,
        max_retries=2, retry_backoff_sec=0.0,
    )
    base.update(kw)
    return ContextualConfig(**base)


def _client(**kw) -> ChatClient:
    c = ChatClient(model="glm-test", api_key="k", base_url="https://fake/v4/",
                   retry_backoff_sec=0.0, **kw)
    c._client = MagicMock()
    return c


def _resp(content: str, prompt_tokens=None, completion_tokens=None) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock(message=MagicMock(content=content))]
    if prompt_tokens is None and completion_tokens is None:
        r.usage = None
    else:
        r.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return r


def test_json_mode_sets_response_format_and_plain_mode_does_not():
    c = _client()
    c._client.chat.completions.create.return_value = _resp("{}")

    c.complete("p", json_mode=True)
    assert c._client.chat.completions.create.call_args[1]["response_format"] == {"type": "json_object"}

    c.complete("p")
    assert "response_format" not in c._client.chat.completions.create.call_args[1]


def test_complete_strips_whitespace_and_reports_usage():
    c = _client()
    c._client.chat.completions.create.return_value = _resp("  hello  ", 11, 7)

    with patch("lex_rag.llm.tracing.end_generation") as end:
        assert c.complete("p") == "hello"

    assert end.call_args[1]["input_tokens"] == 11
    assert end.call_args[1]["output_tokens"] == 7


def test_complete_retries_then_raises_llm_error():
    c = _client(max_retries=2)
    c._client.chat.completions.create.side_effect = RuntimeError("503")

    with pytest.raises(LLMError, match="503"):
        c.complete("p")

    assert c._client.chat.completions.create.call_count == 3   # 首次 + 2 次重试


def test_complete_json_returns_empty_dict_on_unparseable_output():
    """json_object 模式只保证语法合法，不保证字段齐全——解析失败不该中断整批评测。"""
    c = _client()
    c._client.chat.completions.create.return_value = _resp("not json at all")

    assert c.complete_json("p") == {}


def test_loads_or_empty_strips_markdown_fence():
    assert _loads_or_empty('```json\n{"score": 1}\n```') == {"score": 1}
    assert _loads_or_empty('{"score": 1}') == {"score": 1}
    assert _loads_or_empty("[1, 2]") == {}          # 顶层不是对象时按空处理


def test_stream_yields_tokens_and_skips_usage_only_chunk():
    """开了 include_usage 后，最后一个 chunk 只带 usage、没有 choices，不能崩。"""
    c = _client()

    def _chunk(text=None, usage=None):
        ch = MagicMock()
        ch.usage = usage
        ch.choices = [MagicMock(delta=MagicMock(content=text))] if text is not None else []
        return ch

    c._client.chat.completions.create.return_value = iter([
        _chunk("Hel"), _chunk("lo"),
        _chunk(usage=MagicMock(prompt_tokens=5, completion_tokens=2)),
    ])

    with patch("lex_rag.llm.tracing.end_generation") as end:
        assert list(c.stream("p")) == ["Hel", "lo"]

    assert end.call_args[1]["input_tokens"] == 5
    assert end.call_args[1]["output_tokens"] == 2


def test_from_config_can_override_max_retries_to_zero():
    """contextualizer 各类自带重试循环；不覆盖会变成 (n+1)² 次请求。"""
    assert ChatClient.from_config(_cfg(max_retries=3)).max_retries == 3
    assert ChatClient.from_config(_cfg(max_retries=3), max_retries=0).max_retries == 0


def test_from_config_reads_base_url_from_config():
    c = ChatClient.from_config(_cfg(base_url="https://example.com/v4/"))
    assert c.base_url == "https://example.com/v4/"
    assert c.model == "glm-test"


def test_thinking_field_is_omitted_by_default_and_sent_when_set():
    """thinking 是服务商专有字段：不配时整个字段不发，换服务商才不会 400。"""
    c = _client(thinking_style="zai")
    c._client.chat.completions.create.return_value = _resp("{}")

    c.complete("p")
    assert "extra_body" not in c._client.chat.completions.create.call_args[1]

    c.thinking = False
    c.complete("p")
    assert c._client.chat.completions.create.call_args[1]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }

    c.thinking = True
    c.complete("p")
    assert c._client.chat.completions.create.call_args[1]["extra_body"] == {
        "thinking": {"type": "enabled"}
    }


def test_from_config_picks_up_thinking():
    assert ChatClient.from_config(_cfg()).thinking is None
    assert ChatClient.from_config(_cfg(thinking=False)).thinking is False


def test_rate_limit_errors_back_off_harder_than_generic_errors():
    """429 是"等一会儿就好"，2/4/8 秒退避实测不够——限流走 4^n，上限 60s。"""
    c = _client()
    c.retry_backoff_sec = 2.0      # _client() 默认 0，会让两种退避都算成 0

    class RateLimitError(Exception):
        pass

    generic = RuntimeError("connection reset")
    limited = RateLimitError("Error code: 429 - service overloaded")

    # 第一次重试两者相同（base * x^0），从第二次起限流才拉开差距
    assert c._backoff_sec(generic, 0) == c._backoff_sec(limited, 0) == 2.0
    assert c._backoff_sec(generic, 1) == 4.0
    assert c._backoff_sec(limited, 1) == 8.0
    assert c._backoff_sec(limited, 2) > c._backoff_sec(generic, 2)
    assert c._backoff_sec(limited, 10) == 60.0        # 封顶，不会退避到天荒地老


# ── 多服务商的思考开关传参形式 ──────────────────────────────────

def test_thinking_wire_format_differs_per_provider():
    """同一个语义开关，Z.ai 与 DashScope 的参数名不同，必须按 base_url 分流。"""
    zai = ChatClient(base_url="https://api.z.ai/api/paas/v4/", thinking=False)
    dash = ChatClient(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", thinking=False)

    assert zai._extra_body() == {"extra_body": {"thinking": {"type": "disabled"}}}
    assert dash._extra_body() == {"extra_body": {"enable_thinking": False}}
    assert dash.__class__(base_url=dash.base_url, thinking=True)._extra_body() == {
        "extra_body": {"enable_thinking": True}
    }


def test_unknown_provider_sends_no_thinking_field():
    """认不出的服务商不发该字段，好过发一个对方不认识的参数换来 400。"""
    c = ChatClient(base_url="https://api.example.com/v1", thinking=False)
    assert c._extra_body() == {}


def test_explicit_thinking_style_overrides_inference():
    c = ChatClient(base_url="https://api.example.com/v1", thinking=True,
                   thinking_style="dashscope")
    assert c._extra_body() == {"extra_body": {"enable_thinking": True}}
