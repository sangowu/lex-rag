"""
端到端流程测试脚本。

测试顺序：
  1. 配置加载
  2. 数据库连接
  3. Embedding 服务（embed 一条文本）
  4. Reranker 服务（rerank 两条候选，仅 enabled=true 时）
  5. 检索（vector / bm25 / hybrid）
  6. 评估（前 50 条 QA）

用法：
    uv run scripts/test_pipeline.py
    uv run scripts/test_pipeline.py --reranker   # 同时测试 reranker
"""
import argparse
import sys
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, fn):
    try:
        result = fn()
        print(f"  {PASS} {label}" + (f": {result}" if result is not None else ""))
        return True
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker", action="store_true", help="同时测试 reranker 服务")
    parser.add_argument("--qa-file", default="data/qa_cuad.jsonl")
    parser.add_argument("--n-eval", type=int, default=50)
    args = parser.parse_args()

    ok = True

    # ── 1. 配置加载 ─────────────────────────────────────────────
    print("\n[1] Config")
    from lex_rag.config import load_config
    cfg = load_config()
    ok &= check("load_config", lambda: f"chunk={cfg.chunking.chunk_chars}, mode={cfg.retrieval.mode}")

    # ── 2. 数据库连接 ────────────────────────────────────────────
    print("\n[2] Database")
    import psycopg
    ok &= check("connect", lambda: psycopg.connect(cfg.database.dsn).close() or "ok")

    def chunk_count():
        conn = psycopg.connect(cfg.database.dsn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        n = cur.fetchone()[0]
        conn.close()
        if n == 0:
            raise RuntimeError("chunks 表为空，请先运行 ingest")
        return f"{n} rows"
    ok &= check("chunks table", chunk_count)

    # ── 3. Embedding 服务 ────────────────────────────────────────
    print("\n[3] Embedding")
    from lex_rag.embeddings import EmbeddingClient
    client = EmbeddingClient(cfg.embedding, cache_path=None)

    def test_embed():
        vec = client.embed_batch(["This is a test sentence."])[0]
        if len(vec) != 1024:
            raise ValueError(f"expected dim=1024, got {len(vec)}")
        return f"dim={len(vec)}"
    ok &= check("embed_batch", test_embed)

    # ── 4. Reranker 服务 ─────────────────────────────────────────
    print("\n[4] Reranker")
    if not args.reranker:
        print("  [SKIP] 传入 --reranker 参数以启用此测试")
    else:
        from lex_rag.reranker import RerankClient
        from dataclasses import replace
        rr_cfg = replace(cfg.reranker, enabled=True)
        rr_client = RerankClient(rr_cfg)

        def test_rerank():
            from lex_rag.chunking import ChunkWindow
            chunks = [
                ChunkWindow("id1", "doc1", "This agreement is governed by California law.", 0, 44),
                ChunkWindow("id2", "doc1", "The parties agree to arbitration in New York.", 0, 45),
            ]
            ranked = rr_client.rerank("governing law", chunks, top_k=2)
            scores_order = [c.chunk_id for c in ranked]
            return f"order={scores_order}"
        ok &= check("rerank", test_rerank)

    # ── 5. 检索 ──────────────────────────────────────────────────
    print("\n[5] Retrieval")
    from lex_rag.pipeline import RAGPipeline
    from dataclasses import replace

    pipeline = RAGPipeline(cfg)
    query = "What is the governing law of this contract?"

    for mode in ["vector", "bm25", "hybrid"]:
        cfg_m = replace(cfg, retrieval=replace(cfg.retrieval, mode=mode))
        pipeline.cfg = cfg_m

        def test_query(m=mode):
            chunks = pipeline.query(query, k=5)
            if not chunks:
                raise RuntimeError("returned 0 chunks")
            return f"{len(chunks)} chunks, first: {chunks[0].text[:60]!r}"
        ok &= check(f"query mode={mode}", test_query)

    pipeline.close()

    # ── 6. 评估（前 N 条）───────────────────────────────────────
    print(f"\n[6] Evaluation (first {args.n_eval} QA items)")
    from lex_rag.cuad import load_qa
    from lex_rag.evals import evaluate

    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        print(f"  [SKIP] QA 文件不存在: {qa_path}")
    else:
        items = load_qa(qa_path)[: args.n_eval]
        pipeline = RAGPipeline(cfg)

        def test_eval():
            result = evaluate(pipeline, items, cfg.evaluation)
            h1 = result.hit_at_k.get(1, 0)
            h5 = result.hit_at_k.get(5, 0)
            mrr = result.mrr_at_k.get(5, 0)
            if h5 == 0.0:
                raise RuntimeError("hit@5=0.0，检索可能异常")
            return f"hit@1={h1:.3f}  hit@5={h5:.3f}  mrr@5={mrr:.3f}"
        ok &= check("evaluate", test_eval)
        pipeline.close()

    # ── 结果 ──────────────────────────────────────────────────────
    print()
    if ok:
        print("All checks passed.")
    else:
        print("Some checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
