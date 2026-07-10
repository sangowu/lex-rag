"""
全量评估脚本，支持 reranker、结果保存、版本对比。

用法：
    # 全量评估（当前 config）
    uv run scripts/eval.py

    # 开启 reranker
    uv run scripts/eval.py --reranker

    # 对比两个历史结果
    uv run scripts/eval.py --compare data/runs/eval/20260523T10Z.json data/runs/eval/20260523T11Z.json

    # 评估并与上一次结果对比
    uv run scripts/eval.py --reranker --baseline data/runs/eval/20260523T10Z.json
"""
import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from lex_rag.config import load_config
from lex_rag.cuad import load_qa
from lex_rag.evals import _hit, evaluate
from lex_rag.pipeline import RAGPipeline


OUT_DIR = Path("data/runs/eval")


def run_eval(args) -> dict:
    cfg = load_config()
    if args.reranker:
        cfg = replace(cfg, reranker=replace(cfg.reranker, enabled=True))
    if args.hyde:
        cfg = replace(cfg, hyde_enabled=True)
    if args.multi_query:
        cfg = replace(cfg, multi_query_enabled=True)
    if args.table:
        cfg = replace(cfg, database=replace(cfg.database, table=args.table))

    qa_items = load_qa(Path(args.qa_file))
    print(f"QA items : {len(qa_items)}")
    pipeline = RAGPipeline(cfg)

    # 从 DB 读取实际 ingest 参数，比 config.yaml 更可信
    meta = pipeline.store.load_meta() or {}
    chunk_chars = meta.get("chunk_chars", cfg.chunking.chunk_chars)
    overlap     = meta.get("overlap",     cfg.chunking.overlap)
    strategy    = meta.get("strategy",    cfg.chunking.strategy)
    contextual  = meta.get("contextual",  False)
    chunk_mode  = meta.get("chunk_mode",  "standard")
    ingested_at = meta.get("ingested_at", None)

    # 将 chunk_mode 传递给 pipeline，使 query() 走正确路径
    cfg = replace(cfg, chunk_mode=chunk_mode)
    pipeline.cfg = cfg
    pipeline._chunk_mode_cache = chunk_mode  # 避免重复读 DB

    print(f"Table      : {cfg.database.table}"
          + (f"  [ingested_at={ingested_at}]" if ingested_at else ""))
    print(f"Chunk mode : {chunk_mode}")
    print(f"Mode       : {cfg.retrieval.mode}")
    print(f"Reranker   : {cfg.reranker.enabled}"
          + (f"  (fetch_k={cfg.retrieval.rerank_top_k} → top_k={cfg.retrieval.top_k})"
             if cfg.reranker.enabled else ""))
    print(f"Chunk      : {chunk_chars} chars, overlap={overlap}, strategy={strategy}, contextual={contextual}")
    print()

    result = evaluate(pipeline, qa_items, cfg.evaluation)
    pipeline.close()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    row = {
        "run_id":      ts,
        "n_qa":        len(qa_items),
        "table":       cfg.database.table,
        "reranker":    cfg.reranker.enabled,
        "hyde":        cfg.hyde_enabled,
        "multi_query": cfg.multi_query_enabled,
        "mode":        cfg.retrieval.mode,
        "strategy":    strategy,
        "chunk_chars": chunk_chars,
        "overlap":     overlap,
        "contextual":  contextual,
        "ingested_at": ingested_at,
        "rerank_top_k": cfg.retrieval.rerank_top_k if cfg.reranker.enabled else None,
        "chunk_mode":  chunk_mode,
        "hit@1":       result.hit_at_k.get(1,  0.0),
        "hit@3":       result.hit_at_k.get(3,  0.0),
        "hit@5":       result.hit_at_k.get(5,  0.0),
        "hit@10":      result.hit_at_k.get(10, 0.0),
        "mrr@5":       result.mrr_at_k.get(5,  0.0),
        "precision@5": result.precision_at_k.get(5, 0.0),
        "recall@5":    result.recall_at_k.get(5,  0.0),
        "latency_ms":  round(result.avg_latency_ms, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{ts}.json"
    with open(out_path, "w") as f:
        json.dump(row, f, indent=2)
    print(f"Saved → {out_path}")
    return row


def run_agentic_eval(pipeline, qa_items, cfg, contextual_cfg) -> dict:
    """
    两遍扫描评估 Agentic 检索的恢复能力。
    Pass 1：标准检索，找出 chunks==[] 的 failed_set。
    Pass 2：仅对 failed_set 运行 AgenticPipeline，统计恢复率。
    """
    from lex_rag.agent import AgenticPipeline

    agent  = AgenticPipeline(pipeline, contextual_cfg)
    max_k  = max(cfg.k_values)
    scope  = cfg.scope

    # ── Pass 1：标准检索，收集空结果 item ──────────────────────────────
    print("\nPass 1: 标准检索（全量）...")
    valid_items  = [it for it in qa_items if it.spans]
    failed_items = []

    for item in tqdm(valid_items, desc="Standard"):
        doc_id = item.doc_id if scope == "contract" else None
        chunks = pipeline.query(item.question, doc_id=doc_id, k=max_k)
        if not chunks:
            failed_items.append(item)

    n_valid  = len(valid_items)
    n_failed = len(failed_items)
    empty_rate = n_failed / n_valid if n_valid else 0.0
    print(f"  空结果：{n_failed}/{n_valid}  empty_rate={empty_rate:.3f}")

    if not failed_items:
        print("  无需 Agentic 恢复，所有 item 均已命中。")
        return {
            "n_valid": n_valid, "n_failed": 0, "empty_rate": 0.0,
            "n_recovered": 0, "recovery_rate": 0.0,
            "avg_iterations": 1.0, "avg_rewrite_latency_ms": 0.0,
            "net_hit_improvement": 0.0,
        }

    # ── Pass 2：仅对 failed_set 运行 Agentic ──────────────────────────
    print(f"\nPass 2: Agentic 检索（{n_failed} 个 item）...")
    n_recovered      = 0
    iter_counts      = []
    rewrite_latencies = []

    rewrite_log = []

    for item in tqdm(failed_items, desc="Agentic"):
        doc_id = item.doc_id if scope == "contract" else None
        t0 = time.perf_counter()
        chunks, query_trace = agent.query(item.question, doc_id=doc_id, k=max_k)
        elapsed = (time.perf_counter() - t0) * 1000

        iter_counts.append(len(query_trace))
        hit = bool(chunks and _hit(chunks, item.spans, max_k))
        if hit:
            n_recovered += 1

        if len(query_trace) > 1:
            rewrite_latencies.append(elapsed)
            rewrite_log.append({
                "doc_id":       item.doc_id,
                "original":     query_trace[0],
                "rewrites":     query_trace[1:],
                "recovered":    hit,
                "latency_ms":   round(elapsed, 1),
            })

    # 打印重写日志
    if rewrite_log:
        tqdm.write(f"\n── 查询重写记录（共 {len(rewrite_log)} 次）──")
        for entry in rewrite_log:
            status = "✅" if entry["recovered"] else "❌"
            tqdm.write(f"  {status} [{entry['doc_id'][:40]}]")
            tqdm.write(f"     原始：{entry['original'][:80]}")
            for rw in entry["rewrites"]:
                tqdm.write(f"     重写：{rw[:80]}")
        tqdm.write("")

    recovery_rate        = n_recovered / n_failed
    avg_iterations       = sum(iter_counts) / len(iter_counts)
    avg_rewrite_latency  = (sum(rewrite_latencies) / len(rewrite_latencies)
                            if rewrite_latencies else 0.0)
    net_improvement      = recovery_rate * empty_rate

    return {
        "n_valid":                n_valid,
        "n_failed":               n_failed,
        "empty_rate":             round(empty_rate, 4),
        "n_recovered":            n_recovered,
        "recovery_rate":          round(recovery_rate, 4),
        "avg_iterations":         round(avg_iterations, 2),
        "avg_rewrite_latency_ms": round(avg_rewrite_latency, 1),
        "net_hit_improvement":    round(net_improvement, 4),
        "rewrite_log":            rewrite_log,
    }


def print_agentic_result(r: dict) -> None:
    print("\n=== Agentic Eval ===")
    print(f"  空结果 item   : {r['n_failed']}/{r['n_valid']}  "
          f"empty_rate={r['empty_rate']:.3f}")
    print(f"  恢复数        : {r['n_recovered']}  "
          f"recovery_rate={r['recovery_rate']:.3f}")
    print(f"  平均迭代轮数  : {r['avg_iterations']:.2f}")
    print(f"  平均重写延迟  : {r['avg_rewrite_latency_ms']:.1f} ms")
    print(f"  净 hit@k 提升 : +{r['net_hit_improvement']:.3f}  "
          f"(recovery_rate × empty_rate)")


def print_result(r: dict, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    print(f"  {tag}hit@1={r['hit@1']:.3f}  hit@5={r['hit@5']:.3f}  hit@10={r['hit@10']:.3f}"
          f"  mrr@5={r['mrr@5']:.3f}  p@5={r['precision@5']:.3f}  r@5={r['recall@5']:.3f}"
          f"  lat={r['latency_ms']}ms")


def print_diff(a: dict, b: dict, label_a: str, label_b: str) -> None:
    metrics = ["hit@1", "hit@5", "hit@10", "mrr@5", "precision@5", "recall@5"]
    print(f"\n  {'Metric':<14} {label_a:>10} {label_b:>10} {'Delta':>10}")
    print("  " + "-" * 48)
    for m in metrics:
        va, vb = a.get(m, 0), b.get(m, 0)
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"  {m:<14} {va:>10.3f} {vb:>10.3f} {sign+f'{delta:.3f}':>10}")


def compare_files(paths: list[str]) -> None:
    results = []
    for p in paths:
        with open(p) as f:
            results.append((Path(p).name, json.load(f)))

    for label, r in results:
        print(f"\n  {label}  (mode={r.get('mode')}  reranker={r.get('reranker')}  chunk={r.get('chunk_chars')})")
        print_result(r)

    if len(results) == 2:
        print("\n  --- Diff ---")
        print_diff(results[0][1], results[1][1], results[0][0][:20], results[1][0][:20])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file",  default="data/qa_cuad.jsonl")
    parser.add_argument("--reranker", action="store_true", help="开启 reranker")
    parser.add_argument("--hyde",        action="store_true", help="开启 HyDE（查询前用 Gemini 生成假设合同条款）")
    parser.add_argument("--multi-query", action="store_true", dest="multi_query", help="开启 Multi-Query（问题改写为 N 个变体多路检索）")
    parser.add_argument("--table",    default=None,        help="查询的表名（默认用 config.yaml 中的 database.table）")
    parser.add_argument("--agentic",  action="store_true",
                        help="在标准检索空结果的 item 上评估 Agentic 恢复能力")
    parser.add_argument("--baseline", default=None, metavar="PATH",
                        help="评估后与此结果文件对比")
    parser.add_argument("--compare",  nargs=2, metavar=("A", "B"),
                        help="直接对比两个结果文件，不运行新评估")
    args = parser.parse_args()

    if args.compare:
        print("=== Compare ===")
        compare_files(args.compare)
        return

    print("=== Evaluation ===")
    current = run_eval(args)
    print()
    print_result(current, label="result")

    if args.agentic:
        cfg = load_config()
        if args.reranker:
            cfg = replace(cfg, reranker=replace(cfg.reranker, enabled=True))
        if args.table:
            cfg = replace(cfg, database=replace(cfg.database, table=args.table))
        pipeline  = RAGPipeline(cfg)
        qa_items  = load_qa(Path(args.qa_file))
        agentic_r = run_agentic_eval(pipeline, qa_items, cfg.evaluation, cfg.contextual)
        pipeline.close()
        print_agentic_result(agentic_r)
        ts = current["run_id"]
        out_path = OUT_DIR / f"{ts}_agentic.json"
        with open(out_path, "w") as f:
            json.dump(agentic_r, f, indent=2)
        print(f"Saved → {out_path}")

    if args.baseline:
        print("\n=== vs Baseline ===")
        with open(args.baseline) as f:
            baseline = json.load(f)
        print_result(baseline, label="baseline")
        print_diff(baseline, current, "baseline", "current")


if __name__ == "__main__":
    main()
