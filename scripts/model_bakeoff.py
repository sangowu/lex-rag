"""
生成模型横评：用一组已知正确答案的诊断样本，快速筛选候选模型。

为什么不用完整的 eval_generation：50 样本 + judge 在限流下要一小时，测三个模型
就是半天，而且 judge 大概率被 429 打成兜底分。这里只问两个问题，各 10 条：

    无答案子集（has_answer=false）→ 模型应该拒答
    有答案子集（has_answer=true） → 模型应该作答

两个子集按 CUAD 的 ground truth 标签抽样（固定 seed），**与任何模型无关**，
所以"正确行为"是已知的，不需要 judge，也不需要语义相似度。20 次请求就能看出
一个模型的拒答门是否可用。

（`--from-run` 是旧模式：从某次评测结果取该模型的 FP/TP 样本。那样的集合是按
某个模型的错误画像定义的，只适合"另一个模型能否修好这些样本"这类定向提问。）

**检索只做一次，跨模型复用。** 这既省掉重复的 embedding/rerank 调用，也把检索
从变量里消掉——否则两个模型拿到的上下文不同，比较就不成立。

用法：
    uv run scripts/model_bakeoff.py \
        --model "label=deepseek-flash,id=deepseek-v4-flash-0731,base_url=https://dashscope.aliyuncs.com/compatible-mode/v1,key_env=GENERATE_MODEL_API,thinking=false" \
        --model "label=deepseek-flash-think,id=deepseek-v4-flash-0731,base_url=https://dashscope.aliyuncs.com/compatible-mode/v1,key_env=GENERATE_MODEL_API,thinking=true"

结果写入 data/runs/bakeoff/<ts>.json。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

OUT_DIR = Path("data/runs/bakeoff")


def parse_model_spec(raw: str) -> dict:
    """把 "k=v,k=v" 解析成 dict，并补默认值。"""
    import re

    spec: dict = {}
    # 只在"逗号后面紧跟 key=" 的位置切分，这样 label 里的逗号不会被误当成分隔符
    # （label=qwen3.7-plus(off,object) 是很自然的写法）。
    for part in re.split(r",(?=[A-Za-z_][A-Za-z0-9_]*=)", raw):
        if not part.strip():
            continue
        if "=" not in part:
            raise SystemExit(f"--model 片段缺少 '='：{part!r}")
        k, v = part.split("=", 1)
        spec[k.strip()] = v.strip()

    if "id" not in spec or "base_url" not in spec:
        raise SystemExit(f"--model 至少需要 id 与 base_url：{raw!r}")

    spec.setdefault("label", spec["id"])
    spec.setdefault("key_env", "GENERATE_MODEL_API")
    spec.setdefault("thinking_style", "auto")
    spec.setdefault("structured", "json_object")   # json_object | json_schema

    t = spec.get("thinking", "none").lower()
    # none = 不发该字段，用服务端默认。各家默认不同，所以横评时最好显式指定。
    spec["thinking"] = {"true": True, "false": False, "none": None}.get(t)
    return spec


def sample_from_qa(qa: dict, n: int, seed: int) -> tuple[list[str], list[str]]:
    """按 ground truth 的 has_answer 标签抽样，**与任何模型无关**。

    早先的版本是从某次评测结果里取 FP / TP 两组，那样得到的集合是按某个模型的
    错误画像定义的：该模型在自己的 FP 集上必然 0 分、在自己的 TP 集上必然满分，
    这两个数字是构造出来的而非测出来的，跨模型排名不成立。

    固定 seed 保证不同模型、不同批次拿到同一组题。
    """
    import random
    no_answer = sorted(qid for qid, r in qa.items() if not r.get("has_answer", True))
    has_answer = sorted(qid for qid, r in qa.items() if r.get("has_answer", True))
    rng = random.Random(seed)
    rng.shuffle(no_answer)
    rng.shuffle(has_answer)
    return no_answer[:n], has_answer[:n]


def load_from_run(run_path: Path, n: int) -> tuple[list[str], list[str]]:
    """旧模式：从一次评测结果里取该模型的 FP / TP 样本。

    仅适用于"某个模型能否修好另一个模型的失败样本"这类定向提问，不适合排名。
    """
    data = json.loads(run_path.read_text(encoding="utf-8"))
    items = data["per_item"]
    fp = [i["id"] for i in items if i.get("false_positive")][:n]
    tp = [i["id"] for i in items if i.get("true_positive")][:n]
    return fp, tp


def main() -> None:
    ap = argparse.ArgumentParser(description="生成模型横评（拒答门双向诊断）")
    ap.add_argument("--model", action="append", required=True,
                    help="模型规格，k=v 逗号分隔；可重复。见模块 docstring")
    ap.add_argument("--from-run", default=None, metavar="PATH",
                    help="旧模式：从某次评测结果取该模型的 FP/TP 样本。"
                         "只适合定向提问，不适合跨模型排名")
    ap.add_argument("--seed", type=int, default=20260825,
                    help="抽样种子，保证不同模型拿到同一组题")
    ap.add_argument("--n", type=int, default=10, help="每个子集取多少条，默认 10")
    ap.add_argument("--qa", default="data/qa_cuad.jsonl")
    ap.add_argument("--generate-k", type=int, default=8, help="喂给生成模型的 chunk 数")
    args = ap.parse_args()

    # Windows 控制台默认 GBK，遇到 emoji 会抛 UnicodeEncodeError —— 重设为 utf-8。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    load_dotenv()

    from lex_rag.config import load_config
    from lex_rag.generator import LegalGenerator
    from lex_rag.pipeline import RAGPipeline

    specs = [parse_model_spec(m) for m in args.model]
    missing = [s["key_env"] for s in specs if not os.environ.get(s["key_env"])]
    if missing:
        raise SystemExit(f".env 里缺少这些变量：{sorted(set(missing))}")

    qa = {}
    with open(args.qa, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qa[r["id"]] = r

    if args.from_run:
        fp_ids, tp_ids = load_from_run(Path(args.from_run), args.n)
        source = f"from-run:{args.from_run}"
    else:
        fp_ids, tp_ids = sample_from_qa(qa, args.n, args.seed)
        source = f"qa-sample:seed={args.seed}"
    print(f"诊断集来源：{source}", flush=True)

    cfg = load_config()
    cfg = replace(cfg, reranker=replace(cfg.reranker, enabled=True))
    pipeline = RAGPipeline(cfg)

    # ── 检索一次，所有模型共用 ──────────────────────────────────
    print(f"检索 {len(fp_ids) + len(tp_ids)} 条样本的上下文（跨模型复用）...", flush=True)
    contexts: dict[str, list] = {}
    for qid in fp_ids + tp_ids:
        item = qa.get(qid)
        if not item:
            print(f"  [跳过] qa 里没有 {qid[:60]}", flush=True)
            continue
        contexts[qid] = pipeline.query(item["question"], doc_id=item["doc_id"],
                                       k=10)[: args.generate_k]

    results = []
    for spec in specs:
        gen_cfg = replace(
            cfg.contextual,
            model=spec["id"],
            base_url=spec["base_url"],
            api_key=os.environ[spec["key_env"]],
            thinking=spec["thinking"],
            thinking_style=spec["thinking_style"],
            structured_output=spec["structured"],
        )
        gen = LegalGenerator(gen_cfg)
        print(f"\n=== {spec['label']}  (model={spec['id']}, thinking={spec['thinking']}) ===",
              flush=True)

        row = {"label": spec["label"], "model": spec["id"], "base_url": spec["base_url"],
               "thinking": spec["thinking"], "structured": spec["structured"],
               "fp_refused": 0, "fp_total": 0, "tp_wrongly_refused": 0, "tp_total": 0,
               "errors": 0, "latencies_ms": []}

        for subset, ids in (("FP", fp_ids), ("TP", tp_ids)):
            for qid in ids:
                chunks = contexts.get(qid)
                if not chunks:
                    continue
                t0 = time.perf_counter()
                try:
                    r = gen.generate(qa[qid]["question"], chunks)
                except Exception as e:
                    row["errors"] += 1
                    print(f"  [{subset}] {qid.split('__')[-1][:30]:30s} 失败 {type(e).__name__}",
                          flush=True)
                    continue
                dt = (time.perf_counter() - t0) * 1000
                row["latencies_ms"].append(dt)

                if subset == "FP":
                    row["fp_total"] += 1
                    row["fp_refused"] += bool(r.is_refused)
                    mark = "正确拒答" if r.is_refused else "误答"
                else:
                    row["tp_total"] += 1
                    row["tp_wrongly_refused"] += bool(r.is_refused)
                    mark = "误拒" if r.is_refused else "正确作答"
                print(f"  [{subset}] {qid.split('__')[-1][:30]:30s} {mark:8s} {dt:7.0f}ms",
                      flush=True)

        lat = row["latencies_ms"]
        row["avg_latency_ms"] = round(sum(lat) / len(lat), 1) if lat else None
        row["fp_refusal_rate"] = round(row["fp_refused"] / row["fp_total"], 3) if row["fp_total"] else None
        row["tp_wrong_refusal_rate"] = (
            round(row["tp_wrongly_refused"] / row["tp_total"], 3) if row["tp_total"] else None
        )
        results.append(row)

    pipeline.close()

    # ── 汇总 ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'模型':28s} {'FP集正确拒答':>13s} {'TP集误拒':>11s} {'延迟':>9s} {'错误':>5s}")
    print("-" * 78)
    for r in results:
        fp_s = f"{r['fp_refused']}/{r['fp_total']}" if r["fp_total"] else "-"
        tp_s = f"{r['tp_wrongly_refused']}/{r['tp_total']}" if r["tp_total"] else "-"
        lat_s = f"{r['avg_latency_ms']:.0f}ms" if r["avg_latency_ms"] else "-"
        print(f"{r['label'][:28]:28s} {fp_s:>13s} {tp_s:>11s} {lat_s:>9s} {r['errors']:>5d}")
    print("=" * 78)
    print("无答案子集正确拒答：越高越好（合同里没有该条款，应拒答）")
    print("有答案子集误拒　：越低越好（信息确实在上下文里，应作答）")
    print(f"\n⚠️ 每个子集仅 {args.n} 条，翻转率误差约 ±15pp——用于筛选方向，不能当作最终指标。")
    print("⚠️ 无答案子集来自 CUAD 标注，标注稀疏：模型引用了真实存在但未被标注的条款也算误答。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / f"{ts}.json"
    # prompt 会直接影响拒答倾向，所以它的指纹必须跟结果存在一起：
    # 同一个模型换一版 prompt 就是另一个实验。
    import hashlib
    import subprocess

    from lex_rag.generator import _GENERATE_PROMPT, _MULTI_DOC_NOTE
    prompt_fp = hashlib.sha256((_GENERATE_PROMPT + _MULTI_DOC_NOTE).encode("utf-8")).hexdigest()[:12]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        commit = None

    out.write_text(json.dumps({
        "run_id": ts,
        "diagnostic_source": source,
        "prompt_sha256_12": prompt_fp,
        "git_commit": commit,
        "embedding_model": cfg.embedding.model,
        "reranker_model": cfg.reranker.model,
        "n_per_subset": args.n,
        "generate_k": args.generate_k,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
