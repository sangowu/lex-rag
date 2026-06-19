"""
Generation 层评估脚本

三个评估维度：
1. 语义相似度      — 生成答案与 gold answer 的最大余弦相似度（embedding 空间）
2. 拒答准确率      — has_answer=false 时 FP 率 / has_answer=true 时 FN 率
3. RAGAS           — Faithfulness / Answer Relevancy（LLM-as-Judge via Gemini）

用法：
    uv run scripts/eval_generation.py --qa data/qa_cuad.jsonl --limit 50
    uv run scripts/eval_generation.py --qa data/qa_cuad.jsonl --limit 50 --ragas
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from dataclasses import replace

from legal_rag_v1.config import load_config, RagasConfig
from legal_rag_v1.cuad import load_qa, QAItem
from legal_rag_v1.generator import LegalGenerator, GenerationResult
from legal_rag_v1.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# 维度一：语义相似度（embedding cosine similarity）
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_semantic_hits(
    sim_data: list[dict],      # [{"answer": str, "golds": list[str], "row_idx": int}]
    per_item_rows: list[dict],
    cfg,
    threshold: float,
) -> int:
    """批量 embed，计算 answer 与最优 gold 的余弦相似度，更新 per_item_rows["semantic_hit"]。"""
    from legal_rag_v1.embeddings import EmbeddingClient

    # 收集所有待 embed 的文本（去重）
    all_texts: list[str] = []
    seen: set[str] = set()
    for d in sim_data:
        for t in [d["answer"]] + d["golds"]:
            if t and t not in seen:
                all_texts.append(t)
                seen.add(t)

    if not all_texts:
        return 0

    print(f"\n[Semantic Similarity] embedding {len(all_texts)} texts ...")
    embedder = EmbeddingClient(cfg.embedding, cache_path=Path("data/embed_cache_eval.pkl"))
    vecs_list = embedder.embed_texts(all_texts)
    vec_map: dict[str, list[float]] = dict(zip(all_texts, vecs_list))

    hits = 0
    for d in sim_data:
        answer = d["answer"]
        golds = [g for g in d["golds"] if g.strip()]
        row = per_item_rows[d["row_idx"]]

        if not answer or not golds:
            row["semantic_hit"] = False
            row["semantic_sim"] = 0.0
            continue

        a_vec = vec_map.get(answer)
        if a_vec is None:
            row["semantic_hit"] = False
            row["semantic_sim"] = 0.0
            continue

        max_sim = max(_cosine(a_vec, vec_map[g]) for g in golds if g in vec_map)
        hit = max_sim >= threshold
        row["semantic_hit"] = hit
        row["semantic_sim"] = round(max_sim, 4)
        if hit:
            hits += 1

    return hits


# ---------------------------------------------------------------------------
# 维度二：拒答准确率
# ---------------------------------------------------------------------------

def check_refusal(result: GenerationResult, item: QAItem) -> dict:
    """
    返回:
        true_negative  — has_answer=false 且模型正确拒答
        false_positive — has_answer=false 但模型给出了答案（最危险）
        true_positive  — has_answer=true  且模型给出了答案
        false_negative — has_answer=true  但模型错误拒答
    """
    has_answer = item.has_answer
    refused = result.is_refused or not result.answer.strip()

    return {
        "true_negative":  not has_answer and refused,
        "false_positive": not has_answer and not refused,
        "true_positive":  has_answer and not refused,
        "false_negative": has_answer and refused,
    }


# ---------------------------------------------------------------------------
# 维度三：LLM-as-Judge（Faithfulness + Answer Relevancy）
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to the provided context.

Question: {question}
Context: {context}
Answer: {answer}

Is every claim in the answer supported by the context? Reply with JSON only:
{{"score": 0-1, "reason": "one sentence"}}
Score 1.0 = fully grounded, 0.0 = contains hallucinations."""

_RELEVANCY_PROMPT = """\
You are evaluating whether an AI-generated answer is relevant to the question.

Question: {question}
Answer: {answer}

Does the answer directly address the question? Reply with JSON only:
{{"score": 0-1, "reason": "one sentence"}}
Score 1.0 = fully relevant, 0.0 = completely off-topic."""


def run_ragas(samples: list[dict], cfg: RagasConfig) -> dict:
    """
    samples 格式：
        [{"question": ..., "answer": ..., "contexts": [...], "ground_truth": ...}]
    使用 Gemini LLM-as-Judge 评估 Faithfulness 和 Answer Relevancy，
    与 RAGAS 框架定义的指标语义相同，但无需引入 ragas 库。
    """
    import time
    from google import genai

    client = genai.Client(api_key=cfg.api_key)
    min_interval = 60.0 / cfg.rpm_limit
    last_call = 0.0

    def _call(prompt: str) -> dict:
        nonlocal last_call
        elapsed = time.monotonic() - last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        resp = client.models.generate_content(model=cfg.model, contents=prompt)
        last_call = time.monotonic()
        raw = resp.text.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"score": 0.5, "reason": "parse error"}

    faithfulness_scores, relevancy_scores = [], []
    per_sample = []

    from tqdm import tqdm
    for s in tqdm(samples, desc="llm-judge", unit="sample"):
        context = "\n---\n".join(s["contexts"])
        f = _call(_FAITHFULNESS_PROMPT.format(
            question=s["question"], context=context[:3000], answer=s["answer"]
        ))
        r = _call(_RELEVANCY_PROMPT.format(
            question=s["question"], answer=s["answer"]
        ))
        faithfulness_scores.append(float(f.get("score", 0.5)))
        relevancy_scores.append(float(r.get("score", 0.5)))
        per_sample.append({
            "question": s["question"][:80],
            "faithfulness": f,
            "answer_relevancy": r,
        })

    return {
        "faithfulness":      sum(faithfulness_scores) / len(faithfulness_scores),
        "answer_relevancy":  sum(relevancy_scores) / len(relevancy_scores),
        "n_samples":         len(samples),
        "per_sample":        per_sample,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_eval(args) -> None:
    cfg = load_config()
    if args.reranker:
        cfg = replace(cfg, reranker=replace(cfg.reranker, enabled=True))
    pipeline = RAGPipeline(cfg)
    generator = LegalGenerator(cfg.contextual)

    qa_items: list[QAItem] = load_qa(Path(args.qa))
    if args.limit > 0:
        qa_items = qa_items[: args.limit]

    print(f"Evaluating {len(qa_items)} questions ...")

    # 累计计数器
    has_answer_total = 0
    no_answer_total = 0
    tp = tn = fp = fn = 0
    errors = 0
    total_latency_ms = 0.0

    ragas_samples: list[dict] = []
    per_item_rows: list[dict] = []
    sim_data: list[dict] = []   # 供循环后批量计算语义相似度

    from tqdm import tqdm
    for item in tqdm(qa_items, desc="gen-eval", unit="q"):
        # Step 1: 检索（--corpus 模式不按 doc_id 过滤）
        query_doc_id = None if args.corpus else item.doc_id
        chunks = pipeline.query(item.question, k=cfg.retrieval.top_k, doc_id=query_doc_id)
        metas = pipeline.get_doc_metas_for_chunks(chunks)

        # Step 2: 生成（单文档传 meta=，多文档传 metas=）
        gen_chunks = chunks[:args.generate_k]
        if args.corpus:
            result = generator.generate(item.question, gen_chunks, metas=metas or None)
        else:
            result = generator.generate(item.question, gen_chunks,
                                        meta=metas.get(item.doc_id) if metas else None)

        if result.error:
            errors += 1
            continue

        total_latency_ms += result.latency_ms

        # Step 3: 拒答准确率
        refusal = check_refusal(result, item)
        if item.has_answer:
            has_answer_total += 1
            tp += int(refusal["true_positive"])
            fn += int(refusal["false_negative"])
        else:
            no_answer_total += 1
            tn += int(refusal["true_negative"])
            fp += int(refusal["false_positive"])

        # Step 4: 收集 RAGAS 样本（仅 has_answer=True 且未超出 ragas_limit）
        if args.ragas and item.has_answer and len(ragas_samples) < args.ragas_limit:
            contexts = [c.text for c in chunks]
            for doc_id, m in (metas or {}).items():
                meta_lines = [f"[Contract: {doc_id}]"]
                for k, v in m.items():
                    if v:
                        meta_lines.append(f"{k}: {', '.join(v) if isinstance(v, list) else v}")
                contexts = ["\n".join(meta_lines)] + contexts
            ragas_samples.append({
                "question": item.question,
                "answer": result.answer,
                "contexts": contexts,
                "ground_truth": item.answers[0] if item.answers else "",
            })

        row_idx = len(per_item_rows)
        per_item_rows.append({
            "id": item.id,
            "has_answer": item.has_answer,
            "semantic_hit": False,      # 由 compute_semantic_hits 更新
            "semantic_sim": 0.0,
            **refusal,
            "is_refused": result.is_refused,
            "latency_ms": round(result.latency_ms, 1),
            "answer_preview": result.answer[:120],
        })

        # 仅 has_answer=True 且有实际答案时计入语义相似度
        if item.has_answer and result.answer:
            sim_data.append({
                "answer":   result.answer,
                "golds":    [g for g in item.answers if g.strip()],
                "row_idx":  row_idx,
            })

    # ---------------------------------------------------------------------------
    # 语义相似度（批量 embed，循环外统一计算）
    # ---------------------------------------------------------------------------

    semantic_hits = compute_semantic_hits(sim_data, per_item_rows, cfg, args.sim_threshold)

    # ---------------------------------------------------------------------------
    # 汇总指标
    # ---------------------------------------------------------------------------

    n_evaluated = len(per_item_rows)
    metrics: dict = {
        "n_evaluated": n_evaluated,
        "errors": errors,
        "sim_threshold": args.sim_threshold,
        # 语义相似度命中率（has_answer=True 子集）
        "semantic_hit_rate": semantic_hits / max(1, has_answer_total),
        # 拒答
        "false_positive_rate": fp / max(1, no_answer_total),   # 越低越好
        "false_negative_rate": fn / max(1, has_answer_total),  # 越低越好
        "true_positive_rate":  tp / max(1, has_answer_total),
        "true_negative_rate":  tn / max(1, no_answer_total),
        # 延迟
        "avg_latency_ms": total_latency_ms / max(1, n_evaluated),
    }

    if args.ragas and ragas_samples:
        print(f"\n[LLM-Judge] 评估 {len(ragas_samples)} 条样本（model={cfg.ragas.model}）...")
        metrics["ragas"] = run_ragas(ragas_samples, cfg.ragas)

    # ---------------------------------------------------------------------------
    # 打印 & 保存
    # ---------------------------------------------------------------------------

    print("\n=== Generation Eval Results ===")
    print(f"  semantic_hit_rate    : {metrics['semantic_hit_rate']:.3f}  (threshold={args.sim_threshold})")
    print(f"  false_positive_rate  : {metrics['false_positive_rate']:.3f}  (编造答案率，越低越好)")
    print(f"  false_negative_rate  : {metrics['false_negative_rate']:.3f}  (错误拒答率，越低越好)")
    print(f"  avg_latency_ms       : {metrics['avg_latency_ms']:.1f}")
    if "ragas" in metrics:
        r = metrics["ragas"]
        print(f"  faithfulness         : {r['faithfulness']:.3f}  (答案忠实度)")
        print(f"  answer_relevancy     : {r['answer_relevancy']:.3f}  (答案相关性)")

    out_dir = Path("data/runs/gen_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}.json"
    out_path.write_text(
        json.dumps({"metrics": metrics, "per_item": per_item_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved → {out_path}")

    pipeline.close()


# ---------------------------------------------------------------------------
# --compare 工具
# ---------------------------------------------------------------------------

def _print_gen_result(label: str, m: dict) -> None:
    print(f"\n  {label}")
    print(f"    semantic_hit_rate  : {m['semantic_hit_rate']:.3f}  (threshold={m.get('sim_threshold', '?')})")
    print(f"    false_positive_rate: {m['false_positive_rate']:.3f}")
    print(f"    false_negative_rate: {m['false_negative_rate']:.3f}")
    print(f"    avg_latency_ms     : {m['avg_latency_ms']:.1f}")
    if "ragas" in m:
        print(f"    faithfulness       : {m['ragas']['faithfulness']:.3f}")
        print(f"    answer_relevancy   : {m['ragas']['answer_relevancy']:.3f}")


def _print_gen_diff(label_a: str, ma: dict, label_b: str, mb: dict) -> None:
    rows = [
        ("semantic_hit_rate",   "semantic_hit_rate",   False),
        ("false_positive_rate", "false_positive_rate", True),
        ("false_negative_rate", "false_negative_rate", True),
        ("avg_latency_ms",      "avg_latency_ms",      True),
    ]
    print(f"\n  {'Metric':<22} {label_a[:18]:>18} {label_b[:18]:>18} {'Delta':>8}")
    print("  " + "-" * 70)
    for key, display, lower_is_better in rows:
        va, vb = ma.get(key, 0.0), mb.get(key, 0.0)
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"  {display:<22} {va:>18.3f} {vb:>18.3f} {sign+f'{delta:.3f}':>8}")
    if "ragas" in ma and "ragas" in mb:
        for key, display in [("faithfulness", "faithfulness"), ("answer_relevancy", "answer_relevancy")]:
            va, vb = ma["ragas"].get(key, 0.0), mb["ragas"].get(key, 0.0)
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            print(f"  {display:<22} {va:>18.3f} {vb:>18.3f} {sign+f'{delta:.3f}':>8}")


def compare_gen_files(paths: list[str]) -> None:
    results = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        results.append((Path(p).name, data["metrics"]))

    print("=== Generation Eval Compare ===")
    for label, m in results:
        _print_gen_result(label, m)

    if len(results) == 2:
        print("\n  --- Diff (B - A) ---")
        _print_gen_diff(results[0][0], results[0][1], results[1][0], results[1][1])


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RAG generation quality")
    p.add_argument("--qa",       default="data/qa_cuad.jsonl")
    p.add_argument("--limit",    type=int, default=50, help="0 = 全量")
    p.add_argument("--table",    default=None,          help="覆盖 config.yaml 中的 database.table")
    p.add_argument("--reranker", action="store_true",   help="开启 reranker")
    p.add_argument("--ragas",          action="store_true", help="同时运行 RAGAS 评估（需安装 ragas）")
    p.add_argument("--ragas-limit",    type=int, default=20, help="RAGAS 评估样本数，默认 20")
    p.add_argument("--sim-threshold",  type=float, default=0.75, help="语义相似度命中阈值，默认 0.75")
    p.add_argument("--generate-k",     type=int,   default=8,    help="喂给生成模型的 chunk 数，默认 8（<= top_k）")
    p.add_argument("--corpus",         action="store_true",      help="不按 doc_id 过滤，全库 corpus 检索")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="对比两个 gen_eval 结果文件，不运行新评估")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.compare:
        compare_gen_files(args.compare)
    else:
        run_eval(args)
