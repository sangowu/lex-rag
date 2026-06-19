"""
Grid search over chunking × retrieval parameters.

搜索空间:
  chunk_chars  : 600, 800, 1000
  overlap      : 100, 200
  strategy     : fixed, recursive
  retrieval mode: vector, bm25, hybrid

每种 chunking 组合需要重新 ingest；retrieval mode 只影响查询，不需要重新 ingest。

断点续传：已完成的单次结果文件存在则跳过，中断后重新运行会继续未完成的组合。

用法:
    uv run scripts/grid_search.py [--qa-file PATH] [--docs-dir PATH] [--out-dir PATH]
"""
import argparse
import itertools
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from tqdm import tqdm

from legal_rag_v1.config import load_config
from legal_rag_v1.cuad import load_qa
from legal_rag_v1.evals import evaluate
from legal_rag_v1.pipeline import RAGPipeline


CHUNK_CHARS     = [1000, 1200]
OVERLAPS        = [100, 150]
STRATEGIES      = ["recursive"]
RETRIEVAL_MODES = ["hybrid"]


def truncate_chunks(dsn: str) -> None:
    conn = psycopg.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE chunks")
    conn.commit()
    conn.close()


def run_key(chunk_chars: int, overlap: int, strategy: str, mode: str) -> str:
    return f"chunk{chunk_chars}_overlap{overlap}_{strategy}_{mode}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search over RAG hyperparameters")
    parser.add_argument("--qa-file",       default="data/qa_cuad.jsonl")
    parser.add_argument("--docs-dir",      default="data/cuad_docs")
    parser.add_argument("--out-dir",       default="data/runs/grid")
    parser.add_argument("--refresh-cache", action="store_true", help="Discard embedding cache before starting")
    parser.add_argument("--reranker",      action="store_true", help="启用 reranker（使用 config.yaml 中的 rerank_top_k）")
    args = parser.parse_args()

    # 每次运行用时间戳命名独立子目录，避免覆盖历史结果
    run_ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    docs_dir = Path(args.docs_dir)

    base_cfg = load_config()
    if args.reranker:
        base_cfg = replace(base_cfg, reranker=replace(base_cfg.reranker, enabled=True))
    qa_items = load_qa(Path(args.qa_file))
    print(f"Run: {run_ts}")
    print(f"Reranker: {base_cfg.reranker.enabled}"
          + (f"  (rerank_top_k={base_cfg.retrieval.rerank_top_k})" if base_cfg.reranker.enabled else ""))
    print(f"Loaded {len(qa_items)} QA items\n")

    chunking_combos = [
        (c, o, s)
        for c, o, s in itertools.product(CHUNK_CHARS, OVERLAPS, STRATEGIES)
        if o < c
    ]

    summary = []
    total   = len(chunking_combos) * len(RETRIEVAL_MODES)
    run_idx = 0

    for chunk_chars, overlap, strategy in tqdm(chunking_combos, desc="Chunking combos"):
        # 断点续传：检查该 chunking 组合下所有 mode 是否都已完成
        pending_modes = [
            m for m in RETRIEVAL_MODES
            if not (out_dir / f"{run_key(chunk_chars, overlap, strategy, m)}.json").exists()
        ]
        if not pending_modes:
            run_idx += len(RETRIEVAL_MODES)
            print(f"[chunk_chars={chunk_chars}, overlap={overlap}, strategy={strategy}] all done, skipping.")
            continue

        # 只在有待完成 mode 时才重新 ingest
        new_chunking = replace(base_cfg.chunking, chunk_chars=chunk_chars, overlap=overlap, strategy=strategy)
        cfg = replace(base_cfg, chunking=new_chunking)

        print(f"\n[chunk_chars={chunk_chars}, overlap={overlap}, strategy={strategy}] Ingesting ...")
        truncate_chunks(cfg.database.dsn)
        refresh = args.refresh_cache and run_idx == 0
        pipeline = RAGPipeline(cfg, refresh_cache=refresh)
        pipeline.ingest(docs_dir)

        for mode in RETRIEVAL_MODES:
            run_idx += 1
            result_path = out_dir / f"{run_key(chunk_chars, overlap, strategy, mode)}.json"

            # 断点续传：单次结果已存在则跳过
            if result_path.exists():
                print(f"  [{run_idx}/{total}] {run_key(chunk_chars, overlap, strategy, mode)} already exists, skipping.")
                with open(result_path) as f:
                    summary.append(json.load(f)["params"])
                continue

            new_retrieval = replace(cfg.retrieval, mode=mode)
            cfg_run = replace(cfg, retrieval=new_retrieval)
            pipeline.cfg = cfg_run

            result = evaluate(pipeline, qa_items, cfg_run.evaluation)

            row = {
                "chunk_chars":   chunk_chars,
                "overlap":       overlap,
                "strategy":      strategy,
                "mode":          mode,
                "reranker":      cfg_run.reranker.enabled,
                "rerank_top_k":  cfg_run.retrieval.rerank_top_k if cfg_run.reranker.enabled else None,
                "hit@1":         result.hit_at_k.get(1,  0.0),
                "hit@5":         result.hit_at_k.get(5,  0.0),
                "hit@10":        result.hit_at_k.get(10, 0.0),
                "mrr@5":         result.mrr_at_k.get(5,  0.0),
                "precision@5":   result.precision_at_k.get(5, 0.0),
                "recall@5":      result.recall_at_k.get(5,    0.0),
                "latency_ms":    round(result.avg_latency_ms, 2),
            }
            summary.append(row)

            # 单次结果立即落盘（中断后可续传）
            with open(result_path, "w") as f:
                json.dump({"params": row, "full": result.__dict__}, f, indent=2)

            print(
                f"  [{run_idx}/{total}] strategy={strategy:9s} mode={mode:6s} "
                f" hit@1={row['hit@1']:.3f}  hit@5={row['hit@5']:.3f}"
                f"  mrr@5={row['mrr@5']:.3f}  p@5={row['precision@5']:.3f}"
                f"  r@5={row['recall@5']:.3f}  lat={row['latency_ms']}ms"
            )

        pipeline.close()

    # 汇总，按 hit@5 降序
    summary.sort(key=lambda r: r["hit@5"], reverse=True)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_id":    run_ts,
            "n_qa":      len(qa_items),
            "results":   summary,
        }, f, indent=2)

    if summary:
        best = summary[0]
        print("\nGrid search complete — best result:")
        print(f"  chunk_chars={best['chunk_chars']}, overlap={best['overlap']}, strategy={best['strategy']}, mode={best['mode']}")
        print(f"  hit@1={best['hit@1']:.3f}  hit@5={best['hit@5']:.3f}  mrr@5={best['mrr@5']:.3f}")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
