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
import time
import requests
from lex_rag.config import RerankConfig
from lex_rag.chunking import ChunkWindow


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
                    # 显式 top_n=全部：SiliconFlow 等服务商不传 top_n 时返回条数
                    # "由模型决定"，少回来的文档会被当成 0 分排到最后。
                    json={"model": self.cfg.model, "query": query,
                          "documents": texts, "top_n": len(texts)},
                    headers=self._headers(),
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
        raise RuntimeError(f"_score_batch failed after {self.cfg.max_retries} retries") from last_error

    def _score_batch_bge_http(self, query: str, texts: list[str]) -> list[float]:
        last_error = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(self._url, json={"query": query, "texts": texts},
                                     headers=self._headers(), timeout=30)
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
                resp = requests.post(self._url, json={"query": query, "documents": texts},
                                     headers=self._headers(), timeout=60)
                resp.raise_for_status()
                return resp.json()["scores"]
            except Exception as e:
                last_error = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec)
        raise RuntimeError(f"_score_batch failed after {self.cfg.max_retries} retries") from last_error
