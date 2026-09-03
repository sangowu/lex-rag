"""API 边界：鉴权、限流、结构化请求日志。

三件事放同一个模块，因为它们共用同一份请求上下文（`request_id` + 调用方身份），
拆开就得在 `serve.py` 里到处传参。

**为什么是裸 ASGI 中间件而不是 `BaseHTTPMiddleware`**：`/query` 支持 SSE 流式
返回，而 `BaseHTTPMiddleware` 会把响应体收进内存再转发，流式就退化成"等全部
生成完再一次性吐出"——功能看着正常，只是不流了。仓库里已有的
`RootPathMiddleware` 也是裸 ASGI，形态保持一致。

**绝不记录问题与答案原文。** 这是合同问答，请求体里就是法律文本。日志里只留
长度和结构化字段；要复现某次请求靠 `request_id` 去关联，不靠日志里存正文。
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid

__all__ = [
    "ApiKeyRegistry", "RateLimiter", "ApiSafetyMiddleware",
    "bind_log_fields", "current_request_id", "configure_logging",
    "is_loopback",
]

# ---------------------------------------------------------------------------
# 请求上下文
# ---------------------------------------------------------------------------

_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
_LOG_FIELDS: contextvars.ContextVar[dict | None] = contextvars.ContextVar("log_fields", default=None)


def current_request_id() -> str:
    """当前请求的 id；不在请求里时返回空串。"""
    return _REQUEST_ID.get()


def bind_log_fields(**fields) -> None:
    """给本次请求的访问日志补字段。

    ⚠️ 只能在请求所在的那个 asyncio task 里调用。`run_in_executor` 起的工作线程
    **不会**继承 contextvar，在那里调用会静默丢掉——所以 `serve.py` 是在
    executor 返回之后、在协程里绑定的。
    """
    d = _LOG_FIELDS.get()
    if d is not None:
        d.update(fields)


# ---------------------------------------------------------------------------
# 鉴权
# ---------------------------------------------------------------------------

def _key_id(key: str) -> str:
    """日志里代表某个 key 的短标识：sha256 前 8 位。

    绝不记录 key 本身。出了事要能回答"是哪个调用方"，不需要"是哪个密钥"。
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


class ApiKeyRegistry:
    """`API_KEYS` 里逗号分隔的密钥表；为空表示未启用鉴权。"""

    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = [k.strip() for k in (keys or []) if k.strip()]
        self._ids = {k: _key_id(k) for k in self.keys}

    @classmethod
    def from_env(cls, var: str = "API_KEYS") -> ApiKeyRegistry:
        return cls((os.environ.get(var) or "").split(","))

    @property
    def enabled(self) -> bool:
        return bool(self.keys)

    def identify(self, presented: str | None) -> str | None:
        """返回 key_id；密钥不匹配（或未提供）时返回 None。

        用 `compare_digest` 逐个比，不用集合查找——集合查找是内容相关的提前
        返回，正好是计时侧信道。这里的密钥数是个位数，全表比一遍不值一提。
        """
        if not presented:
            return None
        matched = None
        for k in self.keys:
            if secrets.compare_digest(presented, k):
                matched = self._ids[k]
        return matched


def _presented_key(headers: dict[bytes, bytes]) -> str | None:
    """`X-API-Key: <k>` 优先，其次 `Authorization: Bearer <k>`。"""
    raw = headers.get(b"x-api-key")
    if raw:
        return raw.decode("latin-1").strip()
    auth = headers.get(b"authorization")
    if auth:
        v = auth.decode("latin-1").strip()
        if v.lower().startswith("bearer "):
            return v[7:].strip()
    return None


# ---------------------------------------------------------------------------
# 限流
# ---------------------------------------------------------------------------

class RateLimiter:
    """按调用方计的请求漏桶。

    与 `reranker._TokenBucket` 的关键区别是**不睡等**：那个桶在客户端贴着服务商
    上限跑，等一会儿是对的；这里是服务端，等一会儿等于把队列堆在自己身上，正确
    的行为是立刻回 429 让调用方退避。所以 `check()` 返回布尔 + 建议重试秒数。
    """

    def __init__(self, rpm: float, burst: float = 0.0) -> None:
        self.rpm = float(rpm)
        self.capacity = float(burst) if burst else max(1.0, float(rpm))
        self.rate = self.rpm / 60.0
        self._state: dict[str, tuple[float, float]] = {}   # identity -> (tokens, last_ts)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rpm > 0

    def check(self, identity: str) -> tuple[bool, float]:
        """返回 (是否放行, 建议 Retry-After 秒数)。"""
        if not self.enabled:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            tokens, last = self._state.get(identity, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._state[identity] = (tokens - 1.0, now)
                return True, 0.0
            self._state[identity] = (tokens, now)
            return False, max(1.0, (1.0 - tokens) / self.rate)


# ---------------------------------------------------------------------------
# 结构化日志
# ---------------------------------------------------------------------------

class _JsonLineFormatter(logging.Formatter):
    """一行一个 JSON 对象，不加时间戳前缀——前缀会让 `jq` 直接读不了。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "payload", None)
        if payload is None:
            payload = {"event": "log", "level": record.levelname.lower(),
                       "message": record.getMessage()}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(name: str = "lex_rag.api", stream=None) -> logging.Logger:
    """给访问日志装一个只输出 JSON 行的 handler。

    `propagate=False`：uvicorn 会给 root logger 挂自己的格式化器，不切断的话
    每条访问日志会同时以两种格式出现两遍。
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    h = logging.StreamHandler(stream if stream is not None else sys.stdout)
    h.setFormatter(_JsonLineFormatter())
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 中间件
# ---------------------------------------------------------------------------

def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "")


async def _send_json(send, status: int, payload: dict, extra_headers=()) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = [(b"content-type", b"application/json"),
               (b"content-length", str(len(body)).encode())]
    headers += [(k, v) for k, v in extra_headers]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class ApiSafetyMiddleware:
    """鉴权 + 限流 + 访问日志，按这个顺序。

    `exempt` 是**不需要密钥**的路径前缀：
      - `/health` —— ALB / k8s 的健康检查不会带密钥，护上了等于服务永远不健康。
      - `/ui` 与 Gradio 自己的路由 —— 浏览器发不出自定义头。UI 因此**不受密钥
        保护**，只靠网络位置；`serve.py` 的启动检查会拦住"绑非回环地址 + 挂着
        UI"这个组合，除非显式加 `--allow-public-ui`。

    这几条都是有意的洞，写在这里是为了它们别变成无意的洞。
    """

    DEFAULT_EXEMPT = ("/health", "/ui", "/gradio_api", "/theme.css",
                      "/favicon.ico", "/info", "/")

    def __init__(self, app, *, keys: ApiKeyRegistry, limiter: RateLimiter,
                 logger: logging.Logger | None = None,
                 exempt: tuple[str, ...] | None = None) -> None:
        self.app = app
        self.keys = keys
        self.limiter = limiter
        self.logger = logger or configure_logging()
        self.exempt = self.DEFAULT_EXEMPT if exempt is None else exempt

    def _is_exempt(self, path: str) -> bool:
        for p in self.exempt:
            if p == "/":
                if path == "/":
                    return True
            elif path == p or path.startswith(p + "/"):
                return True
        return False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        path = scope.get("path", "")
        client = (scope.get("client") or ("-", 0))[0]

        # 入站带了 X-Request-ID 就沿用，方便跨服务串联；否则新发一个。
        request_id = (headers.get(b"x-request-id", b"").decode("latin-1").strip()
                      or uuid.uuid4().hex[:16])[:64]
        fields: dict = {}
        tok_id = _REQUEST_ID.set(request_id)
        tok_fields = _LOG_FIELDS.set(fields)

        status_holder = {"status": 0}
        rid_header = (b"x-request-id", request_id.encode("latin-1"))

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [rid_header]
            await send(message)

        def emit(status: int, **extra):
            payload = {
                "event": "request", "request_id": request_id,
                "method": scope.get("method", ""), "path": path,
                "status": status, "client": client,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "auth": "enabled" if self.keys.enabled else "disabled",
            }
            payload.update(extra)
            payload.update(fields)
            self.logger.info("", extra={"payload": payload})

        try:
            exempt = self._is_exempt(path)
            key_id = None

            if self.keys.enabled and not exempt:
                key_id = self.keys.identify(_presented_key(headers))
                if key_id is None:
                    emit(401, error="unauthorized")
                    await _send_json(send, 401, {
                        "error": "unauthorized",
                        "detail": "Provide a valid key in the X-API-Key header.",
                        "request_id": request_id,
                    }, [rid_header])
                    return

            if not exempt:
                # 没鉴权时按来源 IP 限流——总比不限强，但它挡不住换 IP 的调用方。
                identity = key_id or f"ip:{client}"
                ok, retry_after = self.limiter.check(identity)
                if not ok:
                    emit(429, error="rate_limited", key_id=key_id)
                    await _send_json(send, 429, {
                        "error": "rate_limited",
                        "detail": f"Too many requests; retry in {retry_after:.0f}s.",
                        "request_id": request_id,
                    }, [(b"retry-after", str(int(retry_after)).encode()), rid_header])
                    return

            if key_id:
                fields.setdefault("key_id", key_id)

            await self.app(scope, receive, send_wrapper)
            emit(status_holder["status"] or 200)
        except Exception as e:
            # 异常也要留一行。否则 500 在访问日志里完全不存在，
            # 排查时看到的是"请求没来过"。
            emit(500, error=type(e).__name__)
            raise
        finally:
            _REQUEST_ID.reset(tok_id)
            _LOG_FIELDS.reset(tok_fields)
