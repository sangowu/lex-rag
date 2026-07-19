from pathlib import Path
from lex_rag.config import AppConfig
from lex_rag.chunking import chunk_text, chunk_parent_child, ChunkWindow
from lex_rag.embeddings import EmbeddingClient
from lex_rag.reranker import RerankClient
from lex_rag.store import VectorStore
from lex_rag import tracing


def _rrf_merge(result_lists: list[list[ChunkWindow]], k: int = 60) -> list[ChunkWindow]:
    """将多路检索结果用 RRF 公式合并，返回按分数降序排列的去重列表。"""
    scores: dict[str, float] = {}
    chunks: dict[str, ChunkWindow] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunks:
                chunks[cid] = chunk
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    return [chunks[cid] for cid in sorted_ids]


class RAGPipeline:
    def __init__(self, cfg: AppConfig, cache_path: Path | None = None, refresh_cache: bool = False):
        self.cfg = cfg
        from lex_rag.embeddings import _DEFAULT_CACHE
        self.embedder = EmbeddingClient(
            cfg.embedding,
            cache_path=cache_path or _DEFAULT_CACHE,
            refresh_cache=refresh_cache,
        )
        self.store = VectorStore(cfg.database.dsn, table=cfg.database.table)
        self.reranker = RerankClient(cfg.reranker) if cfg.reranker.enabled else None

        # contextualizer：根据 contextual_mode 选择实现
        self.contextualizer = None
        if cfg.contextual.enabled:
            if cfg.contextual_mode == "hierarchical":
                from lex_rag.contextualizer import HierarchicalContextualizer
                self.contextualizer = HierarchicalContextualizer(cfg.contextual)
            else:
                from lex_rag.contextualizer import ContextualClient
                self.contextualizer = ContextualClient(cfg.contextual)

        # meta extractor（懒初始化，ingest 时按需创建）
        self._meta_extractor = None
        if cfg.extract_meta:
            from lex_rag.contextualizer import MetadataExtractor
            self._meta_extractor = MetadataExtractor(cfg.contextual)

        # HyDE client（查询阶段使用，不影响 ingest；复用 contextual 的 Gemini 配置）
        self._hyde = None
        if cfg.hyde_enabled:
            from lex_rag.contextualizer import HyDEClient
            self._hyde = HyDEClient(cfg.contextual)

        # Multi-Query expander（复用 contextual 的 Gemini 配置）
        self._expander = None
        if cfg.multi_query_enabled:
            from lex_rag.contextualizer import QueryExpander
            self._expander = QueryExpander(cfg.contextual, n=cfg.multi_query_n)

        # 缓存 chunk_mode，避免每次 query 都访问 DB
        self._chunk_mode_cache: str | None = None

    def _get_chunk_mode(self) -> str:
        if self._chunk_mode_cache is None:
            meta = self.store.load_meta() or {}
            self._chunk_mode_cache = meta.get("chunk_mode") or "standard"
        return self._chunk_mode_cache

    def _ingest_one(self, doc_id: str, text: str) -> None:
        """ingest 单个文档的核心逻辑（不含 TRUNCATE）。"""
        pc_cfg = self.cfg.parent_child
        if self.cfg.chunk_mode == "parent_child":
            parents, children = chunk_parent_child(
                doc_id, text,
                parent_chars=pc_cfg.parent_chars,
                child_chars=pc_cfg.child_chars,
                overlap=pc_cfg.overlap,
            )
            parent_embeddings = self.embedder.embed_texts([p.text for p in parents])
            self.store.add_chunks(parents, parent_embeddings)
            if self.contextualizer:
                children = self.contextualizer.contextualize(text, children)
            child_embeddings = self.embedder.embed_texts([c.text for c in children])
            self.store.add_chunks(children, child_embeddings)
        else:
            chunks = list(chunk_text(doc_id, text, self.cfg.chunking))
            if self.contextualizer:
                chunks = self.contextualizer.contextualize(text, chunks)
            embeddings = self.embedder.embed_texts([c.text for c in chunks])
            self.store.add_chunks(chunks, embeddings)

        if self._meta_extractor:
            meta = self._meta_extractor.extract(doc_id, text)
            self.store.add_doc_meta(doc_id, meta)

    def ingest(self, docs_dir: Path) -> None:
        if not docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
        paths = list(docs_dir.glob("*.txt"))
        # 显式换行日志：tqdm 默认用 \r 刷新进度条，在非 TTY 环境（如 ECS/
        # CloudWatch）里可能被日志驱动缓冲、迟迟看不到任何输出，容易被误判为卡死。
        for i, path in enumerate(paths, 1):
            print(f"[{i}/{len(paths)}] ingesting {path.stem} ...", flush=True)
            self._ingest_one(path.stem, path.read_text(encoding="utf-8"))
        print(f"Ingested {len(paths)} documents.", flush=True)

    def ingest_document(self, path: Path) -> None:
        """增量 ingest 单个文档，不清空现有数据（用于运行时文档上传）。"""
        self._ingest_one(path.stem, path.read_text(encoding="utf-8"))

    def query(self, question: str, doc_id: str | None = None, k: int | None = None) -> list[ChunkWindow]:
        """检索入口。用 span 包裹，使检索与下游生成聚合成同一棵 trace 树。"""
        with tracing.trace_span("lex_rag.retrieval", question):
            return self._query_impl(question, doc_id=doc_id, k=k)

    def _query_impl(self, question: str, doc_id: str | None = None, k: int | None = None) -> list[ChunkWindow]:
        k = k or self.cfg.retrieval.top_k
        fetch_k = self.cfg.retrieval.rerank_top_k if self.reranker else k

        chunk_mode = self._get_chunk_mode()
        children_only = (chunk_mode == "parent_child")
        mode = self.cfg.retrieval.mode

        if self._expander:
            variants = self._expander.expand(question)
            per_k = max(fetch_k // len(variants), 10)
            all_results: list[list[ChunkWindow]] = []
            for v in variants:
                embed_text = self._hyde.generate(v) if self._hyde else v
                vec = self.embedder.embed_text(embed_text)
                if mode == "vector":
                    res = self.store.search_vector(vec, per_k, doc_id, children_only=children_only)
                elif mode == "bm25":
                    res = self.store.search_bm25(v, per_k, doc_id, children_only=children_only)
                else:
                    res = self.store.search_hybrid(v, vec, per_k, doc_id, children_only=children_only)
                all_results.append(res)
            candidates = _rrf_merge(all_results)[:fetch_k]
        else:
            embed_text = self._hyde.generate(question) if self._hyde else question
            query_vec = self.embedder.embed_text(embed_text)
            if mode == "vector":
                candidates = self.store.search_vector(query_vec, fetch_k, doc_id,
                                                       children_only=children_only)
            elif mode == "bm25":
                candidates = self.store.search_bm25(question, fetch_k, doc_id,
                                                     children_only=children_only)
            elif mode == "hybrid":
                candidates = self.store.search_hybrid(question, query_vec, fetch_k, doc_id,
                                                       children_only=children_only)
            else:
                raise ValueError(f"Unknown retrieval mode: {mode}")

        # parent-child：将 child 替换为 parent（更多上下文供 reranker 使用）
        if children_only:
            candidates = self.store.expand_to_parent(candidates)

        if self.reranker:
            return self.reranker.rerank(question, candidates, top_k=k)
        return candidates

    def get_doc_meta(self, doc_id: str) -> dict | None:
        return self.store.get_doc_meta(doc_id)

    def get_doc_metas_for_chunks(self, chunks: list) -> dict[str, dict]:
        """返回 chunks 中所有唯一 doc_id 的 meta，{doc_id: meta}，无 meta 的 doc 不含。"""
        result = {}
        for doc_id in {c.doc_id for c in chunks}:
            m = self.store.get_doc_meta(doc_id)
            if m:
                result[doc_id] = m
        return result

    def close(self) -> None:
        self.store.close()
