from pathlib import Path
from lex_rag.config import AppConfig
from lex_rag.chunking import chunk_text, chunk_parent_child, ChunkWindow
from lex_rag.embeddings import EmbeddingClient
from lex_rag.ingest_guard import SourceRecord, Verdict, classify, content_digest
from lex_rag.reranker import RerankClient
from lex_rag.store import VectorStore
from lex_rag.strategy import RetrievalStrategy
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
    out = []
    for cid in sorted_ids:
        chunk = chunks[cid]
        chunk.score, chunk.score_kind = scores[cid], "rrf"
        out.append(chunk)
    return out


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

        # 下面四个客户端全部懒加载。
        #
        # 改造前它们是 `X if cfg.x_enabled else None`：构造时就按配置决定有没有，
        # 运行时想用也用不了。现在改成"用到才建"，让每一轮检索都能自己决定要不要
        # 走 HyDE / multi-query / rerank —— config 只提供默认策略，不再是开关。
        # 副作用是不用的客户端连构造都省了（各自会 new 一个 ChatClient）。
        self._reranker: RerankClient | None = None
        self._contextualizer = None
        self._meta_extractor = None
        self._hyde = None
        self._expander = None

        # 缓存 chunk_mode，避免每次 query 都访问 DB
        self._chunk_mode_cache: str | None = None

    # ── 懒加载客户端 ────────────────────────────────────────────
    #
    # 保留 `pipeline.reranker` / `pipeline.contextualizer` 这两个属性名：
    # ingest 路径和外部调用方都在用，改名会波及一片而收益为零。

    @property
    def reranker(self) -> RerankClient | None:
        """config 里 enabled=False 时仍可用——策略层可以按轮次决定要不要 rerank。"""
        if self._reranker is None:
            self._reranker = RerankClient(self.cfg.reranker)
        return self._reranker

    @property
    def contextualizer(self):
        """ingest 阶段用。这里仍然尊重 cfg.contextual.enabled：
        它控制的是"要不要给 chunk 加前缀"这个 ingest 决策，不是检索策略。"""
        if self._contextualizer is None and self.cfg.contextual.enabled:
            if self.cfg.contextual_mode == "hierarchical":
                from lex_rag.contextualizer import HierarchicalContextualizer
                self._contextualizer = HierarchicalContextualizer(self.cfg.contextual)
            else:
                from lex_rag.contextualizer import ContextualClient
                self._contextualizer = ContextualClient(self.cfg.contextual)
        return self._contextualizer

    @property
    def meta_extractor(self):
        if self._meta_extractor is None and self.cfg.extract_meta:
            from lex_rag.contextualizer import MetadataExtractor
            self._meta_extractor = MetadataExtractor(self.cfg.contextual)
        return self._meta_extractor

    @property
    def hyde(self):
        if self._hyde is None:
            from lex_rag.contextualizer import HyDEClient
            self._hyde = HyDEClient(self.cfg.contextual)
        return self._hyde

    def expander(self, n: int):
        """QueryExpander 的变体数是构造参数，所以按 n 缓存。"""
        if self._expander is None or getattr(self._expander, "n", None) != n:
            from lex_rag.contextualizer import QueryExpander
            self._expander = QueryExpander(self.cfg.contextual, n=n)
        return self._expander

    def _get_chunk_mode(self) -> str:
        if self._chunk_mode_cache is None:
            meta = self.store.load_meta() or {}
            self._chunk_mode_cache = meta.get("chunk_mode") or "standard"
        return self._chunk_mode_cache

    def _ingest_one(self, doc_id: str, text: str,
                    source: str = "") -> tuple[SourceRecord, Verdict, SourceRecord | None]:
        """ingest 单个文档的核心逻辑（不含 TRUNCATE）。

        返回 `(本次指纹, 判定, 上次指纹)` —— 正是 `ingest_guard.summarise` 要的三元组。
        返回三元组而不是只返回判定，是为了让调用方不必**再查一次** DB 才能打印
        "从 abc 变成 def"；上一版就是那样，多一次往返还多一处会和这里不一致的地方。

        指纹在 chunk 写入**之前**取，取的是喂进来的那份文本——切分、contextual 前缀、
        embedding 都不影响它，它回答的是"这份文件是不是上次那份"，不是"这次 ingest
        的产物是否相同"。
        """
        record = SourceRecord(doc_id=doc_id, sha256=content_digest(text),
                              n_chars=len(text), source=source)
        previous = self.store.get_doc_source(doc_id)
        verdict = classify(previous, record)

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

        if self.meta_extractor:
            meta = self.meta_extractor.extract(doc_id, text)
            self.store.add_doc_meta(doc_id, meta)

        # 写在最后：中途失败就不该留下"已经见过这一版"的记录，否则下次重跑会把
        # 一次失败的 ingest 认成"未变"，然后什么都不做。
        self.store.save_doc_source(record)
        return record, verdict, previous

    def ingest(self, docs_dir: Path) -> list:
        if not docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
        paths = list(docs_dir.glob("*.txt"))
        # 显式换行日志：tqdm 默认用 \r 刷新进度条，在非 TTY 环境（如 ECS/
        # CloudWatch）里可能被日志驱动缓冲、迟迟看不到任何输出，容易被误判为卡死。
        results = []
        for i, path in enumerate(paths, 1):
            print(f"[{i}/{len(paths)}] ingesting {path.stem} ...", flush=True)
            results.append(self._ingest_one(
                path.stem, path.read_text(encoding="utf-8"), source=str(path)))
        print(f"Ingested {len(paths)} documents.", flush=True)
        return results

    def ingest_document(self, path: Path) -> None:
        """增量 ingest 单个文档，不清空现有数据（用于运行时文档上传）。"""
        self._ingest_one(path.stem, path.read_text(encoding="utf-8"))

    def query(self, question: str, doc_id: str | None = None, k: int | None = None,
              strategy: RetrievalStrategy | None = None) -> list[ChunkWindow]:
        """检索入口。用 span 包裹，使检索与下游生成聚合成同一棵 trace 树。

        strategy=None 时从 config 构造默认策略，行为与改造前完全一致。
        """
        with tracing.trace_span("lex_rag.retrieval", question):
            return self._query_impl(question, doc_id=doc_id, k=k, strategy=strategy)

    def _query_impl(self, question: str, doc_id: str | None = None, k: int | None = None,
                    strategy: RetrievalStrategy | None = None) -> list[ChunkWindow]:
        st = (strategy or RetrievalStrategy.from_config(self.cfg)).with_top_k(k)

        # 检索用的查询文本：策略可以给一个重写版，默认用原问题。
        # 注意 rerank 始终用**原问题**打分——重写是为了改善召回，
        # 相关性判断应当对着用户真正问的东西。
        search_text = st.query_text or question

        chunk_mode = self._get_chunk_mode()
        # expand_parent 只在表本身是 parent_child 时才有意义：standard 表里
        # 没有 parent 行，强行展开只会查空。
        children_only = (chunk_mode == "parent_child") and st.expand_parent

        if st.use_multi_query:
            variants = self.expander(st.multi_query_n).expand(search_text)
            per_k = max(st.fetch_k // len(variants), 10)
            all_results: list[list[ChunkWindow]] = []
            for v in variants:
                all_results.append(self._search_one(v, per_k, doc_id, st, children_only))
            candidates = _rrf_merge(all_results)[: st.fetch_k]
        else:
            candidates = self._search_one(search_text, st.fetch_k, doc_id, st, children_only)

        # parent-child：将 child 替换为 parent（更多上下文供 reranker 使用）
        if children_only:
            candidates = self.store.expand_to_parent(candidates)

        if st.rerank:
            return self.reranker.rerank(question, candidates, top_k=st.top_k)
        return candidates[: st.top_k]

    def _search_one(self, text: str, k: int, doc_id: str | None,
                    st: RetrievalStrategy, children_only: bool) -> list[ChunkWindow]:
        """按策略跑一路检索。抽出来是因为 multi-query 要对每个变体重复同样的事。"""
        if st.mode == "bm25":
            # BM25 不需要向量，省掉一次 embedding 调用
            return self.store.search_bm25(text, k, doc_id, children_only=children_only)

        embed_text = self.hyde.generate(text) if st.use_hyde else text
        vec = self.embedder.embed_text(embed_text)
        if st.mode == "vector":
            return self.store.search_vector(vec, k, doc_id, children_only=children_only)
        if st.mode == "hybrid":
            # BM25 那一路用原文而不是 HyDE 生成的假设条款：
            # HyDE 是为了让向量落到条款语义空间，对关键词匹配只会引入噪声。
            return self.store.search_hybrid(text, vec, k, doc_id, children_only=children_only)
        raise ValueError(f"Unknown retrieval mode: {st.mode}")

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
