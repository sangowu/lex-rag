"""Unit tests for EmbeddingClient's caching/batching logic (API call mocked out)."""

import pickle
from unittest.mock import MagicMock, patch

from lex_rag.config import EmbeddingConfig
from lex_rag.embeddings import EmbeddingClient


def _cfg(batch_size: int = 2) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="direct",
        model="test-model",
        base_url="http://fake",
        api_key="key",
        batch_size=batch_size,
        max_retries=0,
        retry_backoff_sec=0.0,
    )


def test_embed_texts_skips_recomputing_already_cached_texts(tmp_path_local):
    client = EmbeddingClient(_cfg(), cache_path=tmp_path_local / "cache.pkl")
    client.embed_batch = MagicMock(side_effect=lambda texts: [[float(len(t))] for t in texts])

    first = client.embed_texts(["a", "bb"])
    assert first == [[1.0], [2.0]]
    assert client.embed_batch.call_count == 1

    second = client.embed_texts(["a", "bb", "ccc"])
    assert second == [[1.0], [2.0], [3.0]]
    # "a" and "bb" are already cached -> only the new text triggers a fresh call
    assert client.embed_batch.call_count == 2
    client.embed_batch.assert_called_with(["ccc"])


def test_embed_texts_persists_cache_to_disk(tmp_path_local):
    cache_path = tmp_path_local / "cache.pkl"
    client = EmbeddingClient(_cfg(), cache_path=cache_path)
    client.embed_batch = MagicMock(return_value=[[9.0]])

    client.embed_texts(["x"])

    assert cache_path.exists()
    with open(cache_path, "rb") as f:
        saved = pickle.load(f)
    assert saved == {"x": [9.0]}


def test_embed_texts_batches_uncached_by_batch_size(tmp_path_local):
    client = EmbeddingClient(_cfg(batch_size=2), cache_path=tmp_path_local / "cache.pkl")
    client.embed_batch = MagicMock(side_effect=lambda texts: [[0.0]] * len(texts))

    client.embed_texts([f"t{i}" for i in range(5)])

    assert client.embed_batch.call_count == 3
    call_lens = [len(call.args[0]) for call in client.embed_batch.call_args_list]
    assert call_lens == [2, 2, 1]


# ── bge_http provider (custom embedding server, not OpenAI-compatible) ──


def _bge_http_cfg() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="bge_http",
        model="BAAI/bge-m3",
        base_url="http://10.0.0.5:8000",
        api_key="",
        batch_size=16,
        max_retries=0,
        retry_backoff_sec=0.0,
    )


def test_bge_http_provider_posts_to_embed_endpoint_and_parses_embeddings(tmp_path_local):
    client = EmbeddingClient(_bge_http_cfg(), cache_path=tmp_path_local / "cache.pkl")

    with patch("lex_rag.embeddings.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        mock_post.return_value.raise_for_status.return_value = None

        result = client.embed_batch(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    called_url, called_kwargs = mock_post.call_args[0][0], mock_post.call_args[1]
    assert called_url == "http://10.0.0.5:8000/embed"
    assert called_kwargs["json"] == {"texts": ["a", "b"]}


def test_bge_http_provider_does_not_construct_an_openai_client(tmp_path_local):
    client = EmbeddingClient(_bge_http_cfg(), cache_path=tmp_path_local / "cache.pkl")

    assert client.client is None
