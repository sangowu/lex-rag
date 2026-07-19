"""
RerankClient: 对候选 chunk 列表重新打分排序。

支持两种后端 API 格式：
  provider="direct"/"ssh_tunnel" — text-embeddings-inference (TEI):
    POST {base_url}/v1/rerank
    Body: {"model": ..., "query": str, "documents": [str, ...]}
    Response: {"results": [{"index": int, "score": float}, ...]}

  provider="bge_http" — 自定义 BGE reranker server:
    POST {base_url}/rerank
    Body: {"query": str, "texts": [str, ...]}
    Response: {"scores": [float, ...]}   # 与输入 texts 顺序一致
"""
import time
import requests
from lex_rag.config import RerankConfig
from lex_rag.chunking import ChunkWindow


class RerankClient:
    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        path = "/rerank" if cfg.provider in ("bge_http", "macrolens") else "/v1/rerank"
        self._url = cfg.base_url.rstrip("/") + path

    def rerank(self, query: str, chunks: list[ChunkWindow], top_k: int) -> list[ChunkWindow]:
        """对 chunks 按相关性重新排序，返回前 top_k 个。"""
        texts = [c.text for c in chunks]
        scores = []
        batch_size = self.cfg.batch_size
        for i in range(0, len(texts), batch_size):
            scores.extend(self._score_batch(query, texts[i:i + batch_size]))
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]

    def _score_batch(self, query: str, texts: list[str]) -> list[float]:
        """调用 rerank 接口，返回与输入 texts 顺序一致的分数列表。"""
        if self.cfg.provider == "bge_http":
            return self._score_batch_bge_http(query, texts)
        if self.cfg.provider == "macrolens":
            return self._score_batch_macrolens(query, texts)

        last_error = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(
                    self._url,
                    json={"model": self.cfg.model, "query": query, "documents": texts},
                    timeout=30,
                )
                resp.raise_for_status()
            except Exception as e:
                last_error = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec)
                continue
            # HTTP 成功后解析不重试，直接抛出
            results = resp.json()["results"]   # [{"index": i, "score"/"relevance_score": f}, ...]
            scores = [0.0] * len(texts)
            for item in results:
                # TEI 返回 "score"；llama.cpp reranking 返回 "relevance_score"
                scores[item["index"]] = item.get("score", item.get("relevance_score", 0.0))
            return scores
        raise RuntimeError(f"_score_batch failed after {self.cfg.max_retries} retries") from last_error

    def _score_batch_bge_http(self, query: str, texts: list[str]) -> list[float]:
        last_error = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(self._url, json={"query": query, "texts": texts}, timeout=30)
                resp.raise_for_status()
                return resp.json()["scores"]
            except Exception as e:
                last_error = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec)
        raise RuntimeError(f"_score_batch failed after {self.cfg.max_retries} retries") from last_error

    def _score_batch_macrolens(self, query: str, texts: list[str]) -> list[float]:
        """MacroLens cloud_server：POST /rerank {query, documents} -> {scores}（顺序与 documents 一致）。"""
        last_error = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(self._url, json={"query": query, "documents": texts}, timeout=60)
                resp.raise_for_status()
                return resp.json()["scores"]
            except Exception as e:
                last_error = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec)
        raise RuntimeError(f"_score_batch failed after {self.cfg.max_retries} retries") from last_error
