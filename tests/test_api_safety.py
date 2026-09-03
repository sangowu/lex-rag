"""API 边界：鉴权、限流、访问日志。

中间件直接按 ASGI 协议调用，不起服务器、不装 httpx——scope/receive/send 是纯
数据结构，手工构造比拉一个测试客户端更快，也更精确。
"""
from __future__ import annotations

import asyncio
import io
import json

import pytest

from lex_rag.api_safety import (
    ApiKeyRegistry,
    ApiSafetyMiddleware,
    RateLimiter,
    bind_log_fields,
    configure_logging,
    current_request_id,
    is_loopback,
)

KEY = "sk-test-primary"
OTHER = "sk-test-secondary"


# --- 一个最小的下游 app ----------------------------------------------------

def make_app(body: bytes = b'{"ok":true}', status: int = 200, chunks: int = 1):
    """回一个固定响应；chunks > 1 时分多个 body 事件发出，用来验证流式不被吞。"""
    async def app(scope, receive, send):
        bind_log_fields(handled=True)
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        for i in range(chunks):
            await send({"type": "http.response.body", "body": body,
                        "more_body": i < chunks - 1})
    return app


def call(mw, path="/query", headers=None, method="POST", client=("1.2.3.4", 5)):
    """把中间件当 ASGI app 调一次，收集它发出的所有消息。

    用 asyncio.run 而不是 pytest-asyncio：中间件是纯 ASGI，一次调用就是一个
    协程，为此多装一个测试插件（还得配 asyncio_mode）不划算。
    """
    scope = {"type": "http", "method": method, "path": path, "client": client,
             "headers": [(k.lower().encode(), v.encode())
                         for k, v in (headers or {}).items()]}
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(mw(scope, receive, send))
    return sent


def status_of(sent):
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def headers_of(sent):
    start = next(m for m in sent if m["type"] == "http.response.start")
    return {k.decode().lower(): v.decode() for k, v in start["headers"]}


def body_of(sent):
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


@pytest.fixture
def log_stream():
    return io.StringIO()


@pytest.fixture
def mw_factory(log_stream):
    def _make(*, keys=(), rpm=1000, burst=0, app=None, **kw):
        return ApiSafetyMiddleware(
            app or make_app(),
            keys=ApiKeyRegistry(list(keys)),
            limiter=RateLimiter(rpm, burst),
            logger=configure_logging("lex_rag.api.test", stream=log_stream),
            **kw,
        )
    return _make


def log_lines(log_stream):
    return [json.loads(ln) for ln in log_stream.getvalue().splitlines() if ln.strip()]


# --- 密钥表 ----------------------------------------------------------------

def test_no_keys_configured_means_auth_disabled():
    assert ApiKeyRegistry([]).enabled is False
    assert ApiKeyRegistry(["", "  "]).enabled is False


def test_from_env_splits_on_commas_and_trims(monkeypatch):
    monkeypatch.setenv("API_KEYS", f" {KEY} , {OTHER} ,")
    reg = ApiKeyRegistry.from_env()
    assert reg.keys == [KEY, OTHER]


def test_identify_accepts_a_configured_key_and_rejects_others():
    reg = ApiKeyRegistry([KEY])
    assert reg.identify(KEY) is not None
    assert reg.identify(OTHER) is None
    assert reg.identify(None) is None
    assert reg.identify("") is None


def test_key_id_never_leaks_the_key_itself():
    """日志里出现的必须是 key_id，不是密钥。这条要是坏了，日志就是凭据泄露面。"""
    reg = ApiKeyRegistry([KEY])
    kid = reg.identify(KEY)
    assert KEY not in kid
    assert len(kid) == 8


def test_distinct_keys_get_distinct_ids():
    reg = ApiKeyRegistry([KEY, OTHER])
    assert reg.identify(KEY) != reg.identify(OTHER)


# --- 鉴权 ------------------------------------------------------------------

def test_query_without_a_key_is_rejected(mw_factory):
    sent = call(mw_factory(keys=[KEY]))
    assert status_of(sent) == 401
    assert json.loads(body_of(sent))["error"] == "unauthorized"


def test_query_with_the_key_passes_through(mw_factory):
    sent = call(mw_factory(keys=[KEY]), headers={"X-API-Key": KEY})
    assert status_of(sent) == 200
    assert json.loads(body_of(sent))["ok"] is True


def test_bearer_token_is_accepted_too(mw_factory):
    sent = call(mw_factory(keys=[KEY]), headers={"Authorization": f"Bearer {KEY}"})
    assert status_of(sent) == 200


def test_a_wrong_key_is_rejected(mw_factory):
    sent = call(mw_factory(keys=[KEY]), headers={"X-API-Key": OTHER})
    assert status_of(sent) == 401


def test_health_never_needs_a_key(mw_factory):
    """健康检查护上了，ALB / k8s 会把服务判成永久不健康。"""
    sent = call(mw_factory(keys=[KEY]), path="/health", method="GET")
    assert status_of(sent) == 200


def test_ui_is_exempt_and_that_is_deliberate(mw_factory):
    """浏览器发不出 X-API-Key，所以 /ui 只靠网络位置保护。

    这个洞由 serve.bind_safety_error 的启动检查兜住——见下面那组测试。
    """
    sent = call(mw_factory(keys=[KEY]), path="/ui/", method="GET")
    assert status_of(sent) == 200


def test_a_path_merely_prefixed_like_an_exempt_one_is_still_protected(mw_factory):
    """/healthcheck 不是 /health 的子路径，不能靠前缀匹配蒙混过去。"""
    sent = call(mw_factory(keys=[KEY]), path="/healthz-secret", method="GET")
    assert status_of(sent) == 401


def test_no_keys_configured_lets_everything_through(mw_factory):
    sent = call(mw_factory(keys=[]))
    assert status_of(sent) == 200


# --- 限流 ------------------------------------------------------------------

def test_rate_limiter_allows_a_burst_then_refuses():
    lim = RateLimiter(rpm=60, burst=3)
    assert [lim.check("a")[0] for _ in range(4)] == [True, True, True, False]


def test_rate_limiter_reports_a_positive_retry_after():
    lim = RateLimiter(rpm=60, burst=1)
    lim.check("a")
    ok, retry = lim.check("a")
    assert ok is False and retry >= 1.0


def test_rate_limiter_keeps_callers_independent():
    lim = RateLimiter(rpm=60, burst=1)
    assert lim.check("a")[0] is True
    assert lim.check("b")[0] is True     # b 不该被 a 的用量牵连
    assert lim.check("a")[0] is False


def test_rate_limit_of_zero_disables_it():
    lim = RateLimiter(rpm=0)
    assert lim.enabled is False
    assert all(lim.check("a")[0] for _ in range(100))


def test_middleware_returns_429_with_retry_after(mw_factory):
    mw = mw_factory(keys=[KEY], rpm=60, burst=1)
    first = call(mw, headers={"X-API-Key": KEY})
    second = call(mw, headers={"X-API-Key": KEY})
    assert status_of(first) == 200
    assert status_of(second) == 429
    assert int(headers_of(second)["retry-after"]) >= 1


def test_rate_limit_is_per_key_not_global(mw_factory):
    mw = mw_factory(keys=[KEY, OTHER], rpm=60, burst=1)
    call(mw, headers={"X-API-Key": KEY})
    sent = call(mw, headers={"X-API-Key": OTHER})
    assert status_of(sent) == 200


def test_health_is_not_rate_limited(mw_factory):
    """豁免路径不消耗配额——否则健康检查会把调用方的额度吃光。"""
    mw = mw_factory(keys=[KEY], rpm=60, burst=1)
    for _ in range(5):
        assert status_of(call(mw, path="/health", method="GET")) == 200
    assert status_of(call(mw, headers={"X-API-Key": KEY})) == 200


# --- request id -------------------------------------------------------------

def test_a_request_id_is_generated_and_returned(mw_factory):
    sent = call(mw_factory())
    assert headers_of(sent)["x-request-id"]


def test_an_inbound_request_id_is_reused(mw_factory):
    sent = call(mw_factory(), headers={"X-Request-ID": "abc-123"})
    assert headers_of(sent)["x-request-id"] == "abc-123"


def test_the_request_id_is_on_the_401_too(mw_factory):
    """被拒的请求最需要能对上号，这里丢了就没法回答"你那次是哪一条"。"""
    sent = call(mw_factory(keys=[KEY]))
    assert headers_of(sent)["x-request-id"] == json.loads(body_of(sent))["request_id"]


def test_the_handler_can_read_the_current_request_id(mw_factory):
    seen = {}

    async def app(scope, receive, send):
        seen["rid"] = current_request_id()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    sent = call(mw_factory(app=app), headers={"X-Request-ID": "rid-9"})
    assert seen["rid"] == "rid-9" == headers_of(sent)["x-request-id"]


# --- 访问日志 ---------------------------------------------------------------

def test_every_request_logs_one_json_line(mw_factory, log_stream):
    call(mw_factory())
    rows = log_lines(log_stream)
    assert len(rows) == 1
    assert rows[0]["event"] == "request"
    assert rows[0]["status"] == 200
    assert rows[0]["path"] == "/query"
    assert rows[0]["latency_ms"] >= 0


def test_handler_bound_fields_reach_the_log(mw_factory, log_stream):
    call(mw_factory())
    assert log_lines(log_stream)[0]["handled"] is True


def test_the_log_never_contains_the_api_key(mw_factory, log_stream):
    """合同问答的访问日志会长期留存；密钥进去一次就等于泄露。"""
    call(mw_factory(keys=[KEY]), headers={"X-API-Key": KEY})
    raw = log_stream.getvalue()
    assert KEY not in raw
    assert log_lines(log_stream)[0]["key_id"]


def test_rejections_are_logged_with_their_reason(mw_factory, log_stream):
    mw = mw_factory(keys=[KEY], rpm=60, burst=1)
    call(mw, headers={"X-API-Key": KEY})
    call(mw, headers={"X-API-Key": KEY})
    call(mw)
    rows = log_lines(log_stream)
    assert [r["status"] for r in rows] == [200, 429, 401]
    assert rows[1]["error"] == "rate_limited"
    assert rows[2]["error"] == "unauthorized"


def test_an_exception_still_produces_a_log_line(mw_factory, log_stream):
    """否则 500 在访问日志里根本不存在，排查时看到的是"请求没来过"。"""
    async def boom(scope, receive, send):
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError):
        call(mw_factory(app=boom))
    row = log_lines(log_stream)[0]
    assert row["status"] == 500
    assert row["error"] == "RuntimeError"


# --- 流式 -------------------------------------------------------------------

def test_streaming_chunks_are_not_coalesced(mw_factory):
    """裸 ASGI 中间件的理由：BaseHTTPMiddleware 会把 SSE 收进内存再一次性吐出。

    功能上看不出来（内容一模一样），只是不流了。这里数 body 事件的**个数**。
    """
    mw = mw_factory(app=make_app(body=b"data: x\n\n", chunks=3))
    sent = call(mw)
    bodies = [m for m in sent if m["type"] == "http.response.body"]
    assert len(bodies) == 3
    assert body_of(sent) == b"data: x\n\n" * 3


# --- 启动检查（serve.py） ---------------------------------------------------

def test_loopback_is_always_fine():
    from scripts.serve import bind_safety_error
    for host in ("127.0.0.1", "localhost", "::1"):
        assert bind_safety_error(host, auth_enabled=False, ui_mounted=True,
                                 allow_public_ui=False) is None


def test_public_bind_without_keys_refuses_to_start():
    """"绑 0.0.0.0 但没配密钥"必须是启动失败，不是一条警告。

    这个仓库已经有三次"配置变了、读它的一侧没跟上，而且完全无声"的事故。
    功能正常但全世界能问你的合同库，属于同一类，所以让它吵。
    """
    from scripts.serve import bind_safety_error
    err = bind_safety_error("0.0.0.0", auth_enabled=False, ui_mounted=False,
                            allow_public_ui=False)
    assert err and "API_KEYS" in err


def test_public_bind_with_keys_but_public_ui_refuses_to_start():
    from scripts.serve import bind_safety_error
    err = bind_safety_error("0.0.0.0", auth_enabled=True, ui_mounted=True,
                            allow_public_ui=False)
    assert err and "--no-ui" in err


def test_public_bind_with_keys_and_no_ui_is_fine():
    from scripts.serve import bind_safety_error
    assert bind_safety_error("0.0.0.0", auth_enabled=True, ui_mounted=False,
                             allow_public_ui=False) is None


def test_public_ui_can_be_opted_into_explicitly():
    from scripts.serve import bind_safety_error
    assert bind_safety_error("0.0.0.0", auth_enabled=True, ui_mounted=True,
                             allow_public_ui=True) is None


def test_is_loopback_does_not_treat_a_public_host_as_local():
    assert is_loopback("0.0.0.0") is False
    assert is_loopback("10.0.0.5") is False
