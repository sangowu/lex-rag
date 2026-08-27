"""Unit tests for RerankClient's batching/sorting logic (network call mocked out)."""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from lex_rag.chunking import ChunkWindow
from lex_rag.config import RerankConfig
from lex_rag.reranker import RerankClient, _TokenBucket, _backoff_sec


def _cfg(batch_size: int = 32) -> RerankConfig:
    return RerankConfig(
        enabled=True,
        provider="direct",
        model="BAAI/bge-reranker-v2-m3",
        base_url="http://fake-reranker",
        api_key="",
        batch_size=batch_size,
        max_retries=0,
        retry_backoff_sec=0.0,
    )


def _chunk(chunk_id: str) -> ChunkWindow:
    return ChunkWindow(chunk_id=chunk_id, doc_id="doc1", text=chunk_id, start=0, end=1)


def test_rerank_sorts_by_score_descending_and_truncates_to_top_k():
    client = RerankClient(_cfg())
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    client._score_batch = MagicMock(return_value=[0.1, 0.9, 0.5])

    result = client.rerank("query", chunks, top_k=2)

    assert [c.chunk_id for c in result] == ["b", "c"]


def test_rerank_splits_requests_by_batch_size():
    client = RerankClient(_cfg(batch_size=2))
    chunks = [_chunk(f"c{i}") for i in range(5)]
    client._score_batch = MagicMock(side_effect=lambda query, texts: [1.0] * len(texts))

    client.rerank("query", chunks, top_k=5)

    # 5 chunks / batch_size=2 -> batches of [2, 2, 1]
    assert client._score_batch.call_count == 3
    call_batch_lens = [len(call.args[1]) for call in client._score_batch.call_args_list]
    assert call_batch_lens == [2, 2, 1]


def test_rerank_empty_chunks_returns_empty_list():
    client = RerankClient(_cfg())
    client._score_batch = MagicMock()

    result = client.rerank("query", [], top_k=5)

    assert result == []
    client._score_batch.assert_not_called()


# ── bge_http provider (custom reranker server, not TEI-compatible) ──


def _bge_http_cfg() -> RerankConfig:
    return RerankConfig(
        enabled=True,
        provider="bge_http",
        model="BAAI/bge-reranker-v2-m3",
        base_url="http://10.0.0.5:8000",
        api_key="",
        batch_size=32,
        max_retries=0,
        retry_backoff_sec=0.0,
    )


def test_bge_http_provider_uses_plain_rerank_path():
    client = RerankClient(_bge_http_cfg())

    assert client._url == "http://10.0.0.5:8000/rerank"


def test_bge_http_provider_posts_query_and_texts_and_parses_scores():
    client = RerankClient(_bge_http_cfg())

    with patch("lex_rag.reranker.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"scores": [0.2, 0.8]}
        mock_post.return_value.raise_for_status.return_value = None

        scores = client._score_batch("query", ["a", "b"])

    assert scores == [0.2, 0.8]
    called_url, called_kwargs = mock_post.call_args[0][0], mock_post.call_args[1]
    assert called_url == "http://10.0.0.5:8000/rerank"
    assert called_kwargs["json"] == {"query": "query", "texts": ["a", "b"]}


# ── 认证与 base_url 约定（换云服务商后才暴露出来的两个坑）──

def _cloud_cfg(base_url: str = "https://api.example.com", api_key: str = "sk-test") -> RerankConfig:
    cfg = _cfg()
    return RerankConfig(
        enabled=True, provider="direct", model=cfg.model, base_url=base_url,
        api_key=api_key, batch_size=32, max_retries=0, retry_backoff_sec=0.0,
    )


def test_direct_provider_sends_bearer_token():
    """自建 TEI 不校验认证，云服务商会 401 —— api_key 非空时必须带 Authorization。"""
    client = RerankClient(_cloud_cfg())

    with patch("lex_rag.reranker.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"results": [{"index": 0, "score": 0.5}]}
        mock_post.return_value.raise_for_status.return_value = None
        client._score_batch("query", ["a"])

    assert mock_post.call_args[1]["headers"] == {"Authorization": "Bearer sk-test"}


def test_no_auth_header_when_api_key_is_empty():
    """自建服务留空 key 时保持原行为，不要发一个空 Bearer。"""
    client = RerankClient(_cloud_cfg(api_key=""))

    with patch("lex_rag.reranker.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"results": [{"index": 0, "score": 0.5}]}
        mock_post.return_value.raise_for_status.return_value = None
        client._score_batch("query", ["a"])

    assert mock_post.call_args[1]["headers"] == {}


def test_base_url_with_v1_suffix_is_not_doubled():
    """embedding 的 base_url 必须带 /v1，同一服务商配 reranker 时极易照抄。"""
    assert RerankClient(_cloud_cfg("https://api.example.com/v1"))._url ==         "https://api.example.com/v1/rerank"
    assert RerankClient(_cloud_cfg("https://api.example.com"))._url ==         "https://api.example.com/v1/rerank"


def test_truncated_results_warn_and_default_to_zero():
    """服务端 top_n 截断时，缺失文档按 0.0 计分并出声告警。"""
    client = RerankClient(_cloud_cfg())

    with patch("lex_rag.reranker.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"results": [{"index": 1, "relevance_score": 0.9}]}
        mock_post.return_value.raise_for_status.return_value = None
        scores = client._score_batch("query", ["a", "b", "c"])

    assert scores == [0.0, 0.9, 0.0]


# ── 重试策略：窗口宽度与去同步 ──────────────────────────────────
# 这两条不是纸面偏好，是 1000 条语料量出来的：失败的 retrieval step 耗时中位
# 8.45s（固定 1.0s backoff 撑不过服务端 8.5 秒的抖动），且有两条查询在相隔
# 0.019s 内同时耗尽重试（4 个 worker 锁步）。见 docs/experiments.md。

def test_backoff_grows_exponentially():
    """固定退避的重试窗口太窄，服务端抖动超过 8.5s 就整条失败。"""
    lows = [1.0 * (2 ** i) * 0.5 for i in range(4)]
    assert lows == [0.5, 1.0, 2.0, 4.0]      # 下界逐次翻倍，否则不是指数退避
    for i in range(4):
        samples = [_backoff_sec(1.0, i) for _ in range(200)]
        assert min(samples) >= lows[i]
        assert max(samples) < 1.0 * (2 ** i) * 1.5


def test_backoff_is_jittered_so_workers_do_not_retry_in_lockstep():
    """没有抖动时多个 worker 会同步重试，把一次降级放大成一片失败。"""
    samples = {_backoff_sec(1.0, 2) for _ in range(200)}
    assert len(samples) > 190          # 几乎全不相同


def test_retry_window_covers_the_observed_outage_length():
    """4 次重试的睡眠总时长下界必须超过实测的 8.45s 失败窗口。"""
    worst_case_floor = sum(1.0 * (2 ** i) * 0.5 for i in range(4))
    assert worst_case_floor >= 7.5     # 0.5+1+2+4 = 7.5s，加上请求耗时已远超 8.45s


def test_failure_message_carries_the_server_response_body():
    """原来只抛 "failed after N retries"，排查时只剩这一句废话。"""
    client = RerankClient(_cloud_cfg())
    resp = MagicMock()
    resp.status_code = 429
    resp.text = "{\"code\":50505,\"message\":\"rate limit exceeded\"}"
    err = requests.HTTPError("429 Client Error")
    err.response = resp

    with patch("lex_rag.reranker.requests.post") as mock_post:
        mock_post.return_value.raise_for_status.side_effect = err
        with pytest.raises(RuntimeError) as ei:
            client._score_batch("query", ["a", "b"])

    msg = str(ei.value)
    assert "429" in msg and "rate limit exceeded" in msg
    assert "2 docs" in msg               # 出错时的 payload 规模也要留下


def test_non_http_failure_still_names_the_exception_type():
    client = RerankClient(_cloud_cfg())
    with patch("lex_rag.reranker.requests.post",
               side_effect=requests.ConnectionError("connection reset")):
        with pytest.raises(RuntimeError) as ei:
            client._score_batch("query", ["a"])
    assert "ConnectionError" in str(ei.value)
    assert "connection reset" in str(ei.value)


def test_each_retry_is_announced_on_stderr(capsys):
    """静默重试会让"没故障"和"故障被重试盖住"变成同一种观测结果。"""
    cfg = _cloud_cfg()
    cfg = RerankConfig(**{**cfg.__dict__, "max_retries": 2, "retry_backoff_sec": 0.0})
    client = RerankClient(cfg)

    with patch("lex_rag.reranker.requests.post",
               side_effect=requests.ConnectionError("boom")):
        with pytest.raises(RuntimeError):
            client._score_batch("query", ["a"])

    err = capsys.readouterr().err
    assert err.count("[rerank]") == 2          # 最后一次失败不再重试，所以不出声
    assert "boom" in err


# ── TPM 限速 ────────────────────────────────────────────────────
# 官方 reranker 档位是扁平的 RPM 2000 / TPM 500000。实测一轮 1000 条语料
# RPM 58（2.9%）、TPM 440132（88%）——RPM 完全不是瓶颈，TPM 均值就贴着上限，
# 而按合同算瞬时速率有 11/25 份超限（最高 181%）。见 docs/experiments.md。

def _tpm_cfg(tpm: int, url: str = "https://api.example.com") -> RerankConfig:
    base = _cloud_cfg(url)
    return RerankConfig(**{**base.__dict__, "tpm_limit": tpm})


def test_bucket_lets_the_first_request_through_immediately():
    b = _TokenBucket(60_000)
    assert b.acquire(10_000) == 0.0


def test_bucket_makes_the_caller_wait_once_the_budget_is_spent():
    b = _TokenBucket(60_000)          # 1000 tok/s
    b.acquire(60_000)                 # 一次性花光
    t0 = time.monotonic()
    waited = b.acquire(2_000)         # 还需攒 2000 tok = 2s
    assert waited >= 1.9
    assert time.monotonic() - t0 >= 1.9


def test_bucket_does_not_deadlock_on_a_request_larger_than_the_bucket():
    """单次请求比整桶还大时限速已无意义，放行去撞服务端，别把循环挂住。"""
    b = _TokenBucket(1_000)
    t0 = time.monotonic()
    b.acquire(10_000_000)
    assert time.monotonic() - t0 < 2.0


def test_clients_on_the_same_endpoint_share_one_bucket():
    """TPM 配额是账号级的；跑批时每个 worker 一个 client，桶必须共享。"""
    a = RerankClient(_tpm_cfg(500_000, "https://shared.example.com"))
    b = RerankClient(_tpm_cfg(500_000, "https://shared.example.com"))
    assert a._bucket is b._bucket is not None


def test_tpm_limit_zero_disables_pacing():
    assert RerankClient(_tpm_cfg(0))._bucket is None


def test_token_estimate_counts_documents_and_query():
    body = {"query": "q" * 40, "documents": ["d" * 400, "d" * 400]}
    # (800 + 40) / 4.0
    assert RerankClient._estimate_tokens(body) == 210.0


def test_pacing_happens_once_per_request_not_once_per_retry():
    """重试是兜底路径，再扣一次配额只会让本就落后的请求更落后。"""
    client = RerankClient(_tpm_cfg(500_000, "https://retry.example.com"))
    client._bucket = MagicMock()
    client._bucket.acquire.return_value = 0.0

    with patch("lex_rag.reranker.requests.post",
               side_effect=requests.ConnectionError("boom")):
        with pytest.raises(RuntimeError):
            client._score_batch("query", ["a"])

    client._bucket.acquire.assert_called_once()
