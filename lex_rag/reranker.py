"""
RerankClient: 对候选 chunk 列表重新打分排序。

支持三种后端 API 格式：
  provider="direct"/"ssh_tunnel" — TEI 及大多数云服务商的 OpenAI 风格 rerank：
    POST {base_url}/v1/rerank
    Body: {"model": ..., "query": str, "documents": [str, ...]}
    Response: {"results": [{"index": int, "score"|"relevance_score": float}, ...]}

  provider="bge_http" — 自定义 BGE reranker server:
    POST {base_url}/rerank
    Body: {"query": str, "texts": [str, ...]}
    Response: {"scores": [float, ...]}   # 与输入 texts 顺序一致

  provider="macrolens" — MacroLens cloud_server：
    POST {base_url}/rerank
    Body: {"query": str, "documents": [str, ...]}
    Response: {"scores": [float, ...]}

认证：``cfg.api_key`` 非空时以 ``Authorization: Bearer`` 发送。自建的 TEI /
llama.cpp 不校验，所以这里长期是空的也没出问题；换成云服务商后必须带，否则 401。

base_url 约定：**不要带 /v1 后缀**（本模块自己拼 `/v1/rerank`）。这与 embedding
那边正好相反——embedding 走 OpenAI SDK，base_url 必须带 `/v1`。同一服务商同时
提供两种服务时最容易在这里配错，所以 direct 这一路会容错地把结尾的 /v1 去掉。
"""
import random
import sys
import threading
import time
import requests
from lex_rag.config import RerankConfig
from lex_rag.chunking import ChunkWindow

# 实测：随机抽 32 条 chunk 走 embedding API（bge-m3，与 bge-reranker-v2-m3 同族
# 分词器），usage.total_tokens 反推得 4.19 字符/token。这里取 4.0 是**故意偏保守**
# ——低估字符/token 就是高估 token 数，宁可自己限得紧一点也别去撞服务端的 429。
_CHARS_PER_TOKEN = 4.0


class _TokenBucket:
    """按 token 计的漏桶，用来贴着服务商的 TPM 上限跑而不是去撞它。

    SiliconFlow 的 reranker 档位是扁平的 RPM 2000 / TPM 500,000（不随消费等级变）。
    实测 1000 条语料一轮：RPM 58（上限的 2.9%，完全不是瓶颈），**TPM 440,132
    （88%）**。而 88% 是均值——按合同算瞬时速率，25 份合同里 11 份在处理时超上限，
    最高 906K（181%）。语料按文档排序，于是那 11 份各有约 40 条连续查询持续跑在
    1.6~1.8 倍上限上，429 就是这么来的。

    退避能把 429 接住（实测 93 次限流只剩 1 条失败），但那是撞了再退；限速是先看
    表再走。两者不冲突：这里限速，退避留作兜底。
    """

    def __init__(self, tokens_per_min: float) -> None:
        self.capacity = float(tokens_per_min)
        self.rate = self.capacity / 60.0
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float) -> float:
        """等到桶里攒够 n 个 token 再放行，返回实际等待秒数。

        单次请求比整桶还大时不死等——那种情况下限速已经无意义，放行让它去撞
        服务端，至少错误消息能说明问题；死等只会让整轮循环挂住。
        """
        n = min(float(n), self.capacity)
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity,
                                   self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return waited
                sleep = (n - self._tokens) / self.rate
            time.sleep(sleep)
            waited += sleep


# 跑批时每个 worker 有自己的 RerankClient（psycopg 连接不能多线程共用），但 TPM
# 配额是**账号级**的，所以桶必须按 endpoint 共享，不能一人一个。
_BUCKETS: dict[str, _TokenBucket] = {}
_BUCKETS_LOCK = threading.Lock()


def _bucket_for(url: str, tpm_limit: float) -> _TokenBucket | None:
    if not tpm_limit or tpm_limit <= 0:
        return None
    with _BUCKETS_LOCK:
        b = _BUCKETS.get(url)
        if b is None or b.capacity != float(tpm_limit):
            b = _TokenBucket(tpm_limit)
            _BUCKETS[url] = b
        return b


def _describe(e: Exception) -> str:
    """把异常压成一行可诊断的文本，HTTP 错误要带上服务端响应体。

    限流、鉴权失败、payload 超限在服务端是三种完全不同的回复，而
    `requests.HTTPError` 的 str 只有状态码和 URL，正文全在 `e.response.text` 里。
    """
    resp = getattr(e, "response", None)
    text = getattr(resp, "text", None) if resp is not None else None
    if isinstance(text, str):
        return f"HTTP {resp.status_code}: {text[:300].replace(chr(10), ' ')}"
    return f"{type(e).__name__}: {e}"


def _backoff_sec(base: float, attempt: int) -> float:
    """指数退避 + 抖动，返回第 attempt 次失败后该睡多久。

    原来是固定 `base` 秒，两个后果都在 1000 条语料里量到了：

    1. **重试窗口太窄**。失败的 retrieval step 耗时中位 8.45s（4 次尝试 + 3×1.0s），
       即服务端只要抖动超过 8.5 秒就能打死整条查询。指数退避把窗口拉到 13~28s。
    2. **没有抖动 = 锁步重试**。4 个 worker 同时撞上同一个降级窗口后会同步重试，
       实测有两条查询在相隔 **0.019s** 内同时耗尽重试。抖动把它们岔开。

    抖动用 [0.5, 1.5) 倍的乘性扰动，而不是"加一个随机小量"——后者在 base 很大时
    起不到岔开的作用。
    """
    return base * (2 ** attempt) * (0.5 + random.random())


class RerankClient:
    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        base = cfg.base_url.rstrip("/")
        if cfg.provider in ("bge_http", "macrolens"):
            self._url = base + "/rerank"
        else:
            # 容错：base_url 写成 https://host/v1 时不要拼成 /v1/v1/rerank
            if base.endswith("/v1"):
                base = base[: -len("/v1")].rstrip("/")
            self._url = base + "/v1/rerank"
        self._bucket = _bucket_for(self._url, getattr(cfg, "tpm_limit", 0))

    @staticmethod
    def _estimate_tokens(body: dict) -> float:
        """一次请求的 token 估算 = （所有文档 + query）的字符数 / 字符每 token。"""
        docs = body.get("documents") or body.get("texts") or []
        return (sum(len(d) for d in docs) + len(body.get("query", ""))) / _CHARS_PER_TOKEN

    def _headers(self) -> dict[str, str]:
        """带 api_key 时发 Bearer 认证；自建服务留空则不发（保持原有行为）。"""
        return {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}

    def rerank(self, query: str, chunks: list[ChunkWindow], top_k: int) -> list[ChunkWindow]:
        """对 chunks 按相关性重新排序，返回前 top_k 个。"""
        texts = [c.text for c in chunks]
        scores = []
        batch_size = self.cfg.batch_size
        for i in range(0, len(texts), batch_size):
            scores.extend(self._score_batch(query, texts[i:i + batch_size]))
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        out = []
        for chunk, score in ranked[:top_k]:
            # 覆盖检索阶段的分数：这里返回的是重排后的名次，分数必须同源。
            chunk.score, chunk.score_kind = float(score), "rerank"
            out.append(chunk)
        return out

    def _post_with_retry(self, body: dict, *, timeout: int, n_docs: int) -> requests.Response:
        """POST 带重试，失败时抛 RuntimeError —— **消息里必须带服务端原话**。

        三个 provider 原本各写一份一模一样的重试循环，且都只抛
        "failed after N retries"，把真正的错误挂在 `__cause__` 上。而 trace_sink
        记的是 `str(e)`，于是 1000 条语料里 13 条查询整条 error 掉、排查时只剩
        这一句废话——payload 超限、限流、鉴权失败被压成了同一条消息。
        """
        # 先按 TPM 限速再发。放在重试循环**外面**：桶已经保证了发送速率，重试是
        # 兜底路径，再扣一次配额只会让本就落后的请求更落后。
        if self._bucket is not None:
            waited = self._bucket.acquire(self._estimate_tokens(body))
            if waited > 1.0:
                print(f"[rerank] TPM 限速等待 {waited:.1f}s", file=sys.stderr, flush=True)

        last_error: Exception | None = None
        detail = "unknown"
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(self._url, json=body, headers=self._headers(),
                                     timeout=timeout)
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_error, detail = e, _describe(e)
                if attempt < self.cfg.max_retries:
                    # 重试必须出声。静默重试会让"没发生故障"和"发生了但被重试盖住"
                    # 变成同一种观测结果，而这正是判断退避策略够不够宽所需要的区分。
                    print(f"[rerank] 第 {attempt + 1} 次尝试失败（{n_docs} docs）：{detail}",
                          file=sys.stderr, flush=True)
                    time.sleep(_backoff_sec(self.cfg.retry_backoff_sec, attempt))
        raise RuntimeError(
            f"_score_batch failed after {self.cfg.max_retries} retries "
            f"({n_docs} docs): {detail}"
        ) from last_error

    def _score_batch(self, query: str, texts: list[str]) -> list[float]:
        """调用 rerank 接口，返回与输入 texts 顺序一致的分数列表。"""
        if self.cfg.provider == "bge_http":
            return self._score_batch_bge_http(query, texts)
        if self.cfg.provider == "macrolens":
            return self._score_batch_macrolens(query, texts)

        resp = self._post_with_retry(
            # 显式 top_n=全部：SiliconFlow 等服务商不传 top_n 时返回条数
            # "由模型决定"，少回来的文档会被当成 0 分排到最后。
            {"model": self.cfg.model, "query": query,
             "documents": texts, "top_n": len(texts)},
            timeout=30, n_docs=len(texts),
        )
        # HTTP 成功后解析不重试，直接抛出
        results = resp.json()["results"]   # [{"index": i, "score"/"relevance_score": f}, ...]
        # 部分服务商的 top_n 默认值会静默截断结果：少掉的文档拿不到分数、
        # 会被当成 0.0 排到最后。这里出声，免得排序悄悄退化成"前 N 个之外全丢"。
        if len(results) < len(texts):
            print(f"[rerank] 服务端只返回 {len(results)}/{len(texts)} 条结果，"
                  f"缺失项按 0.0 计分（可能是服务端 top_n 默认值导致的截断）", flush=True)
        scores = [0.0] * len(texts)
        for item in results:
            # TEI 返回 "score"；部分实现（Cohere 风格）返回 "relevance_score"
            scores[item["index"]] = item.get("score", item.get("relevance_score", 0.0))
        return scores

    def _score_batch_bge_http(self, query: str, texts: list[str]) -> list[float]:
        return self._post_with_retry({"query": query, "texts": texts},
                                     timeout=30, n_docs=len(texts)).json()["scores"]

    def _score_batch_macrolens(self, query: str, texts: list[str]) -> list[float]:
        """MacroLens cloud_server：POST /rerank {query, documents} -> {scores}（顺序与 documents 一致）。"""
        return self._post_with_retry({"query": query, "documents": texts},
                                     timeout=60, n_docs=len(texts)).json()["scores"]
