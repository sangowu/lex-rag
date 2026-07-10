"""Unit tests for RerankClient's batching/sorting logic (network call mocked out)."""

from unittest.mock import MagicMock, patch

from lex_rag.chunking import ChunkWindow
from lex_rag.config import RerankConfig
from lex_rag.reranker import RerankClient


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
