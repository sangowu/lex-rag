"""
RerankClient: 对候选 chunk 列表重新打分排序。

API 格式（text-embeddings-inference / TEI）:
  POST {base_url}/rerank
  Body: {"query": str, "texts": [str, ...], "truncate": true}
  Response: [{"index": int, "score": float}, ...]
"""
import time
import requests
from legal_rag_v1.config import RerankConfig
from legal_rag_v1.chunking import ChunkWindow


class RerankClient:
    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        self._url = cfg.base_url.rstrip("/") + "/v1/rerank"

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
        """调用 /rerank 接口，返回与输入 texts 顺序一致的分数列表。"""
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
            results = resp.json()["results"]   # [{"index": i, "document": ..., "score": f}, ...]
            scores = [0.0] * len(texts)
            for item in results:
                scores[item["index"]] = item["score"]
            return scores
        raise RuntimeError(f"_score_batch failed after {self.cfg.max_retries} retries") from last_error
