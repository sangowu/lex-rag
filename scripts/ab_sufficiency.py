"""
sufficiency_judge 的 A/B 实测：判定器该只服务检索循环，还是兼做生成层校验？

两个臂共用同一份检索结果（**检索只跑一次**），所以差异只可能来自生成路径本身：

    A 臂  单段生成，thinking=true（当前 v5 基线）。判定器只服务检索循环，
          不参与生成——所以 A 臂的生成指标就是现状。
    B 臂  两段式：thinking=false 快速作答 → unified 判定器校验 → 双向纠正
          （不成立的答案翻成拒答；被误拒且上下文其实够的升级重跑）。

同时比较**判定器本身**的两版 prompt：

    specialized  只看问题 + 上下文（A 臂用法）
    unified      额外看一份草稿答案（B 臂用法，顺带产出）

判定器的准确率不需要人工标注：CUAD 有 gold span，"累积 chunks 里有没有 gold span"
是可自动计算的真值，与判定器的 sufficient 对照就是规格第 3 节那张 2×2 表。

用法：
    uv run scripts/ab_sufficiency.py --limit 200 --reranker
    uv run scripts/ab_sufficiency.py --limit 200 --reranker --concurrency 6
    uv run scripts/ab_sufficiency.py --compare data/runs/ab_judge/<ts>.json

结果写入 data/runs/ab_judge/<ts>.json。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from lex_rag.config import load_config
from lex_rag.cuad import QAItem, load_qa
from lex_rag.evals import _hit
from lex_rag.generator import LegalGenerator, VerifiedGenerator
from lex_rag.pipeline import RAGPipeline
from lex_rag.sufficiency import SufficiencyJudge

OUT_DIR = Path("data/runs/ab_judge")


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def arm_metrics(rows: list[dict]) -> dict:
    """一个臂的拒答混淆矩阵 + 延迟 + 成本。

    J（判别力）= 正确拒答率 − 误拒率。单看 FP 或 FN 会奖励"一律拒答"这种退化
    策略——CUAD 的无答案样本占 75%，全拒答就能拿到 FP=0。J 对这种偏移免疫。
    """
    ok = [r for r in rows if not r["error"]]
    has = [r for r in ok if r["has_answer"]]
    no = [r for r in ok if not r["has_answer"]]

    tp = sum(1 for r in has if not r["refused"])
    fn = sum(1 for r in has if r["refused"])
    tn = sum(1 for r in no if r["refused"])
    fp = sum(1 for r in no if not r["refused"])

    fnr = fn / len(has) if has else 0.0
    tnr = tn / len(no) if no else 0.0
    lat = [r["latency_ms"] for r in ok]
    hits = sum(1 for r in has if r.get("semantic_hit"))

    return {
        "n_evaluated": len(ok),
        "errors": sum(1 for r in rows if r["error"]),
        "n_has_answer": len(has),
        "n_no_answer": len(no),
        "false_positive_rate": round(fp / len(no), 4) if no else None,
        "false_negative_rate": round(fnr, 4) if has else None,
        "true_positive_rate": round(tp / len(has), 4) if has else None,
        "true_negative_rate": round(tnr, 4) if no else None,
        "discrimination_J": round(tnr - fnr, 4) if (has and no) else None,
        "semantic_hit_rate": round(hits / len(has), 4) if has else None,
        "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
        "p50_latency_ms": round(sorted(lat)[len(lat) // 2], 1) if lat else None,
        "avg_llm_calls": round(sum(r.get("llm_calls", 1) for r in ok) / len(ok), 3) if ok else None,
        "n_flipped_to_refusal": sum(1 for r in ok if r.get("flipped") == "to_refusal"),
        "n_escalated": sum(1 for r in ok if r.get("flipped") == "escalated"),
    }


def judge_metrics(rows: list[dict], key: str) -> dict:
    """判定器准确率，对照 gold span 这个自动真值（规格第 3 节的 2×2 表）。

    只在 has_answer=True 的样本上算——无答案样本没有 gold span，"不在 chunks 里"
    是恒真的，混进来会把假阳性率稀释成一个看着很漂亮但没有意义的数。
    无答案样本单独看 out_of_scope 的召回。
    """
    graded = [r for r in rows if r.get(key) is not None]
    has = [r for r in graded if r["has_answer"]]
    no = [r for r in graded if not r["has_answer"]]

    correct_stop = sum(1 for r in has if r["gold_in_chunks"] and r[key]["sufficient"])
    wasted = sum(1 for r in has if r["gold_in_chunks"] and not r[key]["sufficient"])
    premature = sum(1 for r in has if not r["gold_in_chunks"] and r[key]["sufficient"])
    correct_cont = sum(1 for r in has if not r["gold_in_chunks"] and not r[key]["sufficient"])

    n_gold_in = correct_stop + wasted
    n_gold_out = premature + correct_cont
    lat = [r[key]["latency_ms"] for r in graded]

    return {
        "n_graded": len(graded),
        # 2×2（分母是 has_answer 子集）
        "correct_stop": correct_stop,
        "wasted_round_fn": wasted,          # gold 在里面却说不够 → 白烧一轮
        "premature_stop_fp": premature,     # gold 不在里面却说够了 → 答案必错
        "correct_continue": correct_cont,
        "accuracy": round((correct_stop + correct_cont) / len(has), 4) if has else None,
        "fn_rate": round(wasted / n_gold_in, 4) if n_gold_in else None,
        "fp_rate": round(premature / n_gold_out, 4) if n_gold_out else None,
        # 无答案子集：判定器该说"合同里根本没有"
        "oos_recall_on_no_answer": round(
            sum(1 for r in no if r[key]["out_of_scope"]) / len(no), 4) if no else None,
        "oos_falsealarm_on_has_answer": round(
            sum(1 for r in has if r[key]["out_of_scope"]) / len(has), 4) if has else None,
        "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
        "missing_kind_dist": _dist(r[key]["missing_kind"] for r in graded
                                   if not r[key]["sufficient"]),
        "n_judge_errors": sum(1 for r in graded if r[key].get("error")),
    }


def _dist(values) -> dict:
    from collections import Counter
    return dict(Counter(values).most_common())


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def retrieve_all(pipeline: RAGPipeline, items: list[QAItem], k: int, generate_k: int) -> dict:
    """串行检索，一次跑完两个臂都用。

    串行是有意的：VectorStore 只有一个 psycopg 连接，多线程共用不安全。检索占
    总时长的一小部分，没必要为它承担一类难复现的并发 bug。
    """
    out: dict[str, dict] = {}
    for item in tqdm(items, desc="retrieve", unit="q"):
        chunks = pipeline.query(item.question, k=k, doc_id=item.doc_id)
        metas = pipeline.get_doc_metas_for_chunks(chunks)
        gen_chunks = chunks[:generate_k]
        out[item.id] = {
            "chunks": gen_chunks,
            "meta": metas.get(item.doc_id) if metas else None,
            # gold span 是否落在真正喂给模型的那几个 chunk 里——判定器的真值
            "gold_in_chunks": bool(_hit(gen_chunks, item.spans, len(gen_chunks)))
                              if item.spans else False,
        }
    return out


def run_concurrent(items: list[QAItem], ctx: dict, fn, desc: str, workers: int) -> list[dict]:
    """把逐条任务铺到线程池上。任何一条抛出都被收成 error 行，不中断整轮。"""
    rows: list[dict] = [None] * len(items)   # type: ignore[list-item]

    def _one(idx_item):
        idx, item = idx_item
        try:
            return idx, fn(item, ctx[item.id])
        except Exception as e:                # noqa: BLE001 — 任何异常都只该毁掉这一条
            return idx, {"error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, row in tqdm(pool.map(_one, enumerate(items)),
                             total=len(items), desc=desc, unit="q"):
            rows[idx] = row
    return rows


def base_row(item: QAItem, c: dict) -> dict:
    return {
        "id": item.id,
        "has_answer": item.has_answer,
        "gold_in_chunks": c["gold_in_chunks"],
        "error": None,
        "semantic_hit": False,
        "semantic_sim": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="sufficiency_judge A/B 实测")
    ap.add_argument("--qa", default="data/qa_cuad.jsonl")
    ap.add_argument("--limit", type=int, default=200, help="0 = 全量")
    ap.add_argument("--reranker", action="store_true", default=True)
    ap.add_argument("--no-reranker", dest="reranker", action="store_false")
    ap.add_argument("--generate-k", type=int, default=8)
    ap.add_argument("--sim-threshold", type=float, default=0.70)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="LLM 阶段的并发度。检索始终串行")
    ap.add_argument("--no-escalate", dest="escalate", action="store_false", default=True,
                    help="B 臂关闭「误拒 → 升级重跑」这一路")
    ap.add_argument("--compare", default=None, metavar="PATH", help="只打印已有结果文件")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.compare:
        report(json.loads(Path(args.compare).read_text(encoding="utf-8")))
        return

    cfg = load_config()
    if args.reranker:
        cfg = replace(cfg, reranker=replace(cfg.reranker, enabled=True))

    items = load_qa(Path(args.qa))
    if args.limit > 0:
        items = items[: args.limit]

    pipeline = RAGPipeline(cfg)
    print(f"A/B on {len(items)} 条 "
          f"（has_answer={sum(i.has_answer for i in items)}，"
          f"reranker={cfg.reranker.enabled}，concurrency={args.concurrency}）", flush=True)

    ctx = retrieve_all(pipeline, items, cfg.retrieval.top_k, args.generate_k)
    pipeline.close()          # 后面全是 LLM 调用，不再需要数据库连接

    gen_a = LegalGenerator(cfg.contextual)                       # thinking 由配置决定
    gen_b = VerifiedGenerator(cfg.contextual, escalate=args.escalate)
    judge_spec = SufficiencyJudge(replace(cfg.contextual, thinking=False),
                                  mode="sufficiency")

    # ── A 臂：单段生成 ──────────────────────────────────────────
    def _arm_a(item: QAItem, c: dict) -> dict:
        r = gen_a.generate(item.question, c["chunks"], meta=c["meta"])
        row = base_row(item, c)
        row.update(error=r.error, refused=r.is_refused or not r.answer.strip(),
                   answer=r.answer, latency_ms=r.latency_ms, llm_calls=1, flipped=None)
        return row

    # ── B 臂：两段式 ────────────────────────────────────────────
    def _arm_b(item: QAItem, c: dict) -> dict:
        r = gen_b.generate(item.question, c["chunks"], meta=c["meta"])
        row = base_row(item, c)
        row.update(error=r.error, refused=r.is_refused or not r.answer.strip(),
                   answer=r.answer, latency_ms=r.latency_ms,
                   llm_calls=r.llm_calls, flipped=r.flipped, unified=r.verdict)
        return row

    # ── 专用判定器：单独跑一遍，用于与 unified 版比准确率 ────────
    def _judge_only(item: QAItem, c: dict) -> dict:
        v = judge_spec.judge(item.question, c["chunks"])
        row = base_row(item, c)
        row["specialized"] = v.to_dict()
        return row

    t0 = time.perf_counter()
    rows_a = run_concurrent(items, ctx, _arm_a, "arm-A single", args.concurrency)
    rows_b = run_concurrent(items, ctx, _arm_b, "arm-B two-stage", args.concurrency)
    rows_j = run_concurrent(items, ctx, _judge_only, "judge-spec", args.concurrency)
    wall_sec = time.perf_counter() - t0

    # ── 语义相似度（批量 embed，两个臂各算一次）────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_generation import compute_semantic_hits

    by_id = {i.id: i for i in items}
    for rows in (rows_a, rows_b):
        sim_data = [
            {"answer": r["answer"], "golds": [g for g in by_id[r["id"]].answers if g.strip()],
             "row_idx": idx}
            for idx, r in enumerate(rows)
            if not r["error"] and r["has_answer"] and r.get("answer")
        ]
        compute_semantic_hits(sim_data, rows, cfg, args.sim_threshold)

    # ── 判定器指标：specialized 来自单独那轮，unified 来自 B 臂 ──
    judge_rows_spec = rows_j
    judge_rows_uni = [dict(r) for r in rows_b if not r["error"]]

    payload = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "provenance": {
            "generation_model": cfg.contextual.model,
            "generation_base_url": cfg.contextual.base_url,
            "arm_a_thinking": cfg.contextual.thinking,
            "arm_b_fast_thinking": False,
            "arm_b_escalate": args.escalate,
            "judge_thinking": False,
            "embedding_model": cfg.embedding.model,
            "reranker_model": cfg.reranker.model if cfg.reranker.enabled else None,
            "table": cfg.database.table,
            "limit": args.limit,
            "generate_k": args.generate_k,
            "sim_threshold": args.sim_threshold,
            "concurrency": args.concurrency,
            "wall_sec": round(wall_sec, 1),
            "git_commit": _git_commit(),
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "arms": {"A_single_thinking": arm_metrics(rows_a),
                 "B_two_stage": arm_metrics(rows_b)},
        "judges": {"specialized": judge_metrics(judge_rows_spec, "specialized"),
                   "unified": judge_metrics(judge_rows_uni, "unified")},
        "per_item": {"A": rows_a, "B": rows_b, "judge_specialized": rows_j},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{payload['run_id']}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report(payload)
    print(f"\nSaved → {out}")


def report(p: dict) -> None:
    a, b = p["arms"]["A_single_thinking"], p["arms"]["B_two_stage"]
    print("\n" + "=" * 74)
    print(f"{'指标':26s} {'A 单段(thinking)':>16s} {'B 两段式':>13s} {'方向':>8s}")
    print("-" * 74)
    rows = [
        ("false_positive_rate", "越低越好"), ("false_negative_rate", "越低越好"),
        ("semantic_hit_rate", "越高越好"), ("discrimination_J", "越高越好"),
        ("avg_latency_ms", "越低越好"), ("p50_latency_ms", "越低越好"),
        ("avg_llm_calls", "越低越好"), ("errors", "越低越好"),
    ]
    for key, direction in rows:
        va, vb = a.get(key), b.get(key)
        fmt = (lambda v: "-" if v is None else
               (f"{v:.3f}" if isinstance(v, float) and v < 10 else f"{v:.0f}"
                if isinstance(v, float) else str(v)))
        print(f"{key:26s} {fmt(va):>16s} {fmt(vb):>13s} {direction:>8s}")
    print("-" * 74)
    print(f"{'B 臂翻成拒答':26s} {b['n_flipped_to_refusal']:>16d}")
    print(f"{'B 臂升级重跑':26s} {b['n_escalated']:>16d}")

    print("\n" + "=" * 74)
    print(f"{'判定器（对照 gold span）':30s} {'specialized':>14s} {'unified':>12s}")
    print("-" * 74)
    js, ju = p["judges"]["specialized"], p["judges"]["unified"]
    for key in ("accuracy", "fp_rate", "fn_rate", "oos_recall_on_no_answer",
                "oos_falsealarm_on_has_answer", "avg_latency_ms", "n_judge_errors"):
        f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"{key:30s} {f(js.get(key)):>14s} {f(ju.get(key)):>12s}")
    print("-" * 74)
    print("fp_rate = gold 不在上下文却判定「够了」→ 提前停止，答案必错")
    print("fn_rate = gold 已在上下文却判定「不够」→ 白烧一轮")
    print(f"specialized missing_kind: {js.get('missing_kind_dist')}")
    print(f"unified     missing_kind: {ju.get('missing_kind_dist')}")


if __name__ == "__main__":
    main()
