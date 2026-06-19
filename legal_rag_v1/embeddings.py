import pickle
import time
from pathlib import Path

from openai import OpenAI

from legal_rag_v1.config import EmbeddingConfig
from legal_rag_v1 import ssh_tunnel as _ssh

_DEFAULT_CACHE = Path("data/embed_cache.pkl")


class EmbeddingClient:
    def __init__(
        self,
        cfg: EmbeddingConfig,
        cache_path: Path = _DEFAULT_CACHE,
        refresh_cache: bool = False,
    ):
        self.cfg = cfg
        if cfg.provider == "ssh_tunnel":
            tunnel = _ssh.get_or_create(
                host=cfg.ssh_host,
                user=cfg.ssh_user,
                ssh_port=cfg.ssh_port,
                key_path=cfg.ssh_key_path,
                local_port=cfg.ssh_local_port,
                remote_host=cfg.ssh_remote_host,
                remote_port=cfg.ssh_remote_port,
            )
            tunnel.ensure()
        elif cfg.provider != "direct":
            raise ValueError(f"Unknown provider: {cfg.provider}")

        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self._cache_path = cache_path
        self._cache: dict[str, list[float]] = {}

        if refresh_cache:
            self._delete_cache_file()
        else:
            self._load_cache()

    # ── 缓存持久化 ──────────────────────────────────────────────

    def _load_cache(self) -> None:
        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                self._cache = pickle.load(f)
            print(f"[embed cache] loaded {len(self._cache)} entries from {self._cache_path}")

    def _save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f)

    def _delete_cache_file(self) -> None:
        if self._cache_path and self._cache_path.exists():
            self._cache_path.unlink()
            print(f"[embed cache] cleared {self._cache_path}")

    # ── API 调用（带重试）────────────────────────────────────────

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_error = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = self.client.embeddings.create(model=self.cfg.model, input=texts)
                return [item.embedding for item in response.data]
            except Exception as e:
                last_error = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_sec)
        raise RuntimeError(f"embed_batch failed after {self.cfg.max_retries} retries") from last_error

    # ── 公开接口（含缓存）───────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        uncached = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if uncached:
            batch_size = self.cfg.batch_size
            batches = [uncached[i:i + batch_size] for i in range(0, len(uncached), batch_size)]
            vecs = [v for batch in batches for v in self.embed_batch(batch)]
            self._cache.update(zip(uncached, vecs))
            self._save_cache()
        return [self._cache[t] for t in texts]

    def embed_text(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self.embed_batch([text])[0]
            self._save_cache()
        return self._cache[text]
