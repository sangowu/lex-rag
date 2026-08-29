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
import sys
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from dataclasses import replace

from lex_rag.config import load_config, RagasConfig
from lex_rag.cuad import load_qa, QAItem
from lex_rag.generator import LegalGenerator, GenerationResult
from lex_rag.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# 维度一：语义相似度（embedding cosine similarity）
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
# 合同原文里全角引号 / en dash 很常见，gold span 抄出来时经常被换成半角。
# 这是排版差异，不是内容差异，判据必须先抹平它。
# str.maketrans 接受 {码位: 替换串}，这里刻意用码位而不是字面字符——
# NBSP 写成字面量就是源码里一个隐形字符，改坏了看不出来。
_TYPOGRAPHY = {
    0x201C: '"', 0x201D: '"',   # 弯双引号
    0x2018: "'", 0x2019: "'",   # 弯单引号
    0x2013: "-", 0x2014: "-",   # en / em dash
    0x00A0: " ",                # NBSP
}

# gold 短于这个长度就不用包含判据。"Inc" / "LLC" / "the" 这种碎片能在几乎任何
# 答案里撞上，那不是命中，是巧合。CUAD 里真正有意义的最短 gold 是 4 字符级别
# （如 "1999"），所以门槛设在这里。
_MIN_GOLD_CHARS = 4


def _normalize(text: str) -> str:
    """包含判据用的归一化：只抹排版差异，不删标点、不动词形。

    删标点会让 "Party A, Inc." 和 "Party AInc" 判成同一个；不删则 gold 里的
    标点必须原样出现。逐字引用场景下后者才是对的——prompt 要求的就是原文照抄。
    """
    return _WS_RE.sub(" ", text.translate(_TYPOGRAPHY)).strip().lower()


def _contains_gold(answer: str, gold: str) -> bool:
    """gold span 是否**逐字**出现在答案里。

    这条判据的存在理由：prompt 明确要求 "quote the exact sentence(s) that contain
    the answer"，而 CUAD 的 gold 是从那句话里抽出来的短 span。于是一句 40 词的
    原文引用 vs 一个 5 词的 span，余弦只有 0.5 左右——**整句里逐字含着 gold，
    却被判成没命中**。旧尺子惩罚的正是 prompt 要求的行为。
    """
    g = _normalize(gold)
    if len(g) < _MIN_GOLD_CHARS:
        return False
    return g in _normalize(answer)


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
) -> dict[str, int]:
    """批量 embed，逐条判定命中，更新 per_item_rows 上的三个 semantic_* 字段。

    命中 = **包含判据 或 余弦判据**。两个都记，因为：
      · 包含判据管逐字引用（prompt 要求的行为，旧尺子恰好判它不及格）；
      · 余弦判据管改写（模型用自己的话答对时 gold 不会字面出现）。
    ⚠️ `semantic_hit_cosine` 必须单独留着——历史上所有 semantic_hit_rate 都是
    余弦判据测的，不留这一格就切断了与历史结果的可比性。
    """
    from lex_rag.embeddings import EmbeddingClient

    # 收集所有待 embed 的文本（去重）
    all_texts: list[str] = []
    seen: set[str] = set()
    for d in sim_data:
        for t in [d["answer"]] + d["golds"]:
            if t and t not in seen:
                all_texts.append(t)
                seen.add(t)

    if not all_texts:
        return {"hit": 0, "cosine": 0, "contain_only": 0}

    print(f"\n[Semantic Similarity] embedding {len(all_texts)} texts ...")
    embedder = EmbeddingClient(cfg.embedding, cache_path=Path("data/embed_cache_eval.pkl"))
    vecs_list = embedder.embed_texts(all_texts)
    vec_map: dict[str, list[float]] = dict(zip(all_texts, vecs_list))

    counts = {"hit": 0, "cosine": 0, "contain_only": 0}
    for d in sim_data:
        answer = d["answer"]
        golds = [g for g in d["golds"] if g.strip()]
        row = per_item_rows[d["row_idx"]]

        if not answer or not golds:
            _set_hit(row, cos_hit=False, contains=False, max_sim=0.0, counts=counts)
            continue

        # 包含判据先算：它不依赖 embedding，embedding 挂了也还剩一半判据在。
        contains = any(_contains_gold(answer, g) for g in golds)

        a_vec = vec_map.get(answer)
        sims = [_cosine(a_vec, vec_map[g]) for g in golds if g in vec_map] if a_vec else []
        max_sim = max(sims) if sims else 0.0
        _set_hit(row, cos_hit=max_sim >= threshold, contains=contains,
                 max_sim=max_sim, counts=counts)

    return counts


def _set_hit(row: dict, *, cos_hit: bool, contains: bool,
             max_sim: float, counts: dict[str, int]) -> None:
    row["semantic_hit"] = cos_hit or contains
    row["semantic_hit_cosine"] = cos_hit
    row["semantic_hit_contains"] = contains
    row["semantic_sim"] = round(max_sim, 4)
    counts["hit"] += int(cos_hit or contains)
    counts["cosine"] += int(cos_hit)
    counts["contain_only"] += int(contains and not cos_hit)


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
    使用 LLM-as-Judge 评估 Faithfulness 和 Answer Relevancy，
    与 RAGAS 框架定义的指标语义相同，但无需引入 ragas 库。
    """
    import time

    from lex_rag.llm import ChatClient

    chat = ChatClient.from_config(cfg)
    min_interval = 60.0 / cfg.rpm_limit
    last_call = 0.0
    n_failed = 0

    def _call(prompt: str) -> dict:
        nonlocal last_call
        elapsed = time.monotonic() - last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        nonlocal n_failed
        try:
            data = chat.complete_json(prompt, trace_name="judge")
        except Exception as e:
            # 单条 judge 失败（多为 429）不该让整轮评测白跑：记中性分继续。
            # n_failed 会写进结果，避免"看起来跑完了其实一半是兜底分"。
            n_failed += 1
            last_call = time.monotonic()
            return {"score": 0.5, "reason": f"llm error: {type(e).__name__}"}
        last_call = time.monotonic()
        # complete_json 解析失败会返回 {}，这里同样按中性分处理
        return data or {"score": 0.5, "reason": "parse error"}

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

    if n_failed:
        print(f"  [warn] {n_failed}/{len(samples) * 2} 次 judge 调用失败，已按中性分 0.5 计入",
              flush=True)

    return {
        "faithfulness":      sum(faithfulness_scores) / len(faithfulness_scores),
        "answer_relevancy":  sum(relevancy_scores) / len(relevancy_scores),
        "n_samples":         len(samples),
        "n_judge_failed":    n_failed,
        "per_sample":        per_sample,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _prompt_fingerprint() -> str:
    """生成 prompt 的短哈希，用来回答"这两轮跑的是不是同一版 prompt"。"""
    import hashlib

    from lex_rag.generator import _GENERATE_PROMPT, _MULTI_DOC_NOTE
    blob = (_GENERATE_PROMPT + _MULTI_DOC_NOTE).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _usage_metrics(rows: list[dict]) -> dict:
    """token 用量的汇总。

    `usage_reported` 是必须的一格：服务端不返回 usage 时所有计数都是 0，而 0 和
    "真的没用 token"在数字上分不开。少了这一格，一次静默的服务端变更会表现为
    "成本降到 0"，看起来还像好消息。
    """
    if not rows:
        return {"avg_prompt_tokens": 0.0, "avg_completion_tokens": 0.0,
                "avg_reasoning_tokens": 0.0, "total_tokens": 0, "usage_reported": 0.0}
    n = len(rows)
    prompt = sum(r.get("prompt_tokens", 0) for r in rows)
    completion = sum(r.get("completion_tokens", 0) for r in rows)
    reasoning = sum(r.get("reasoning_tokens", 0) for r in rows)
    reported = sum(1 for r in rows if r.get("prompt_tokens", 0) or r.get("completion_tokens", 0))
    return {
        "avg_prompt_tokens": prompt / n,
        "avg_completion_tokens": completion / n,
        "avg_reasoning_tokens": reasoning / n,
        "total_tokens": prompt + completion,
        "usage_reported": reported / n,
    }


def _git_commit() -> str | None:
    """当前 commit，便于把结果文件对回代码。不在 git 仓库或 git 不可用时返回 None。"""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def _error_kind(err: str) -> str:
    """把错误信息压成可统计的类别。

    不存原文：错误里可能带 request id 之类的变量，直接计数会散成一堆只出现
    一次的键，失去统计意义。
    """
    e = str(err)
    for needle, kind in (
        ("FreeTierOnly", "quota_exhausted_403"),
        ("429", "rate_limited_429"),
        ("401", "auth_401"),
        ("403", "forbidden_403"),
        ("timeout", "timeout"),
        ("Timeout", "timeout"),
    ):
        if needle in e:
            return kind
    return type(err).__name__ if isinstance(err, Exception) else e[:60]


def run_eval(args) -> None:
    cfg = load_config()
    if args.reranker:
        cfg = replace(cfg, reranker=replace(cfg.reranker, enabled=True))
    # thinking 做 A/B 时必须能从命令行切，不能靠两次运行之间改 config.yaml——
    # 改文件的做法既不可复现，又容易把别的字段一起带进去，单变量就不成立了。
    if args.thinking is not None:
        cfg = replace(cfg, contextual=replace(cfg.contextual, thinking=args.thinking))
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
    error_samples: Counter = Counter()
    total_latency_ms = 0.0

    ragas_samples: list[dict] = []
    per_item_rows: list[dict] = []
    sim_data: list[dict] = []   # 供循环后批量计算语义相似度

    # generate_k 只能从检索回来的东西里切，所以 top_k 必须先够大。
    # 此前这里写死用 cfg.retrieval.top_k（=10），于是 `--generate-k 20` 是个
    # 空操作——切 20 条却只有 10 条可切。要测"多给上下文"就必须两个一起动。
    top_k = args.top_k or max(cfg.retrieval.top_k, args.generate_k)
    # generate_k 缺省跟随 top_k：把"给生成层少一点"当默认曾是个未验证的假设，
    # 配对实测（gold 在前 8 名的 30 条对照组，8 条 vs 20 条上下文）是 1 赚 1 亏，
    # 多给干扰项没有可测的伤害。要复现旧行为显式传 `--generate-k 8`。
    generate_k = args.generate_k or top_k

    from tqdm import tqdm
    for item in tqdm(qa_items, desc="gen-eval", unit="q"):
        # Step 1: 检索（--corpus 模式不按 doc_id 过滤）
        query_doc_id = None if args.corpus else item.doc_id
        chunks = pipeline.query(item.question, k=top_k, doc_id=query_doc_id)
        metas = pipeline.get_doc_metas_for_chunks(chunks)

        # Step 2: 生成（单文档传 meta=，多文档传 metas=）
        gen_chunks = chunks[:generate_k]
        if args.corpus:
            result = generator.generate(item.question, gen_chunks, metas=metas or None)
        else:
            result = generator.generate(item.question, gen_chunks,
                                        meta=metas.get(item.doc_id) if metas else None)

        if result.error:
            errors += 1
            # 只计数不记类型的话，跑完只剩一个 errors=163，看不出是配额、限流
            # 还是请求格式问题——上一轮就是这样，只能另发一次请求才查得出来。
            error_samples[_error_kind(result.error)] += 1
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
            "semantic_hit_cosine": False,
            "semantic_hit_contains": False,
            "semantic_sim": 0.0,
            **refusal,
            "is_refused": result.is_refused,
            "latency_ms": round(result.latency_ms, 1),
            # token 用量。延迟能反映 thinking 的代价，但延迟混着网络和排队，
            # 不能当账单用；要回答"省了多少钱"只能看这三个数。
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "reasoning_tokens": result.usage.reasoning_tokens,
            # ⚠️ 存**完整**答案与 gold，不是 120 字符的预览。
            # 预览是个真实的坑：换判据想离线重算时，长引用的 gold 正好落在
            # 截断之外，重算只能给出下界，非重跑不可。一条几百字节，
            # 200 条不值得为此省。
            "answer": result.answer,
            "gold_answers": list(item.answers),
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
        "error_kinds": dict(error_samples),
        "errors": errors,
        "sim_threshold": args.sim_threshold,
        # 语义相似度命中率（has_answer=True 子集）
        "semantic_hit_rate": semantic_hits["hit"] / max(1, has_answer_total),
        # 旧尺子。历史结果全是这么测的，删掉就没法和 v4/v5 的表比。
        "semantic_hit_rate_cosine": semantic_hits["cosine"] / max(1, has_answer_total),
        "semantic_contain_only": semantic_hits["contain_only"],
        # 拒答
        "false_positive_rate": fp / max(1, no_answer_total),   # 越低越好
        "false_negative_rate": fn / max(1, has_answer_total),  # 越低越好
        "true_positive_rate":  tp / max(1, has_answer_total),
        "true_negative_rate":  tn / max(1, no_answer_total),
        # 延迟
        "avg_latency_ms": total_latency_ms / max(1, n_evaluated),
        # token 用量。延迟能反映 thinking 的代价，但延迟混着网络与排队，不能当账单
        # 用；"关掉 thinking 省了多少钱"这个问题此前答不出来，就是因为这里是空的。
        # ⚠️ avg_reasoning_tokens 是 avg_completion_tokens 的**一部分**，不要相加。
        **_usage_metrics(per_item_rows),
    }

    # ── 实验记录 ────────────────────────────────────────────────
    # 没有这一段的话，两个结果文件之间的差异只能靠时间戳和记忆去还原——做模型
    # 横评时这是硬伤：过两周没人说得清某个基线是哪个配置跑出来的。
    # prompt 用哈希而不是全文：全文太长，而哈希足以回答"这两轮 prompt 是否相同"。
    provenance = {
        "generation_model": cfg.contextual.model,
        "generation_base_url": cfg.contextual.base_url,
        "generation_thinking": cfg.contextual.thinking,
        "structured_output": cfg.contextual.structured_output,
        "judge_model": cfg.ragas.model if args.ragas else None,
        "judge_base_url": cfg.ragas.base_url if args.ragas else None,
        "prompt_sha256_12": _prompt_fingerprint(),
        "embedding_model": cfg.embedding.model,
        "reranker_model": cfg.reranker.model if cfg.reranker.enabled else None,
        "reranker_enabled": cfg.reranker.enabled,
        "table": cfg.database.table,
        "limit": args.limit,
        "generate_k": generate_k,
        "top_k": top_k,
        "sim_threshold": args.sim_threshold,
        "corpus_mode": args.corpus,
        "git_commit": _git_commit(),
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(f"[provenance] gen={provenance['generation_model']} "
          f"thinking={provenance['generation_thinking']} "
          f"judge={provenance['judge_model']} "
          f"prompt={provenance['prompt_sha256_12']}", flush=True)

    out_dir = Path("data/runs/gen_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}.json"

    def _save() -> None:
        out_path.write_text(
            json.dumps({"provenance": provenance, "metrics": metrics,
                        "per_item": per_item_rows},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 主指标先落盘再跑 judge。上一版只用 try/except 包住 judge，但那个 except 自己
    # 也会失败（print 的 emoji 在 GBK 控制台上抛 UnicodeEncodeError），50 条结果
    # 照样丢光。落盘早于一切可能出错的后续步骤，才是真正的保险。
    _save()
    print(f"[saved] 主指标已写入 {out_path}", flush=True)

    if args.ragas and ragas_samples:
        print(f"\n[LLM-Judge] 评估 {len(ragas_samples)} 条样本（model={cfg.ragas.model}）...")
        try:
            metrics["ragas"] = run_ragas(ragas_samples, cfg.ragas)
        except Exception as e:
            print(f"[warn] LLM-Judge 阶段失败，主指标已保存：{type(e).__name__}: {e}", flush=True)
            metrics["ragas"] = {"error": f"{type(e).__name__}: {e}"}
        _save()

    # ---------------------------------------------------------------------------
    # 打印 & 保存
    # ---------------------------------------------------------------------------

    print("\n=== Generation Eval Results ===")
    print(f"  semantic_hit_rate    : {metrics['semantic_hit_rate']:.3f}  "
          f"(逐字包含 或 cosine>={args.sim_threshold})")
    print(f"    └ 仅 cosine（旧尺子）: {metrics['semantic_hit_rate_cosine']:.3f}  ← 与历史结果可比")
    print(f"    └ 仅靠包含认出       : {metrics['semantic_contain_only']} 条")
    print(f"  false_positive_rate  : {metrics['false_positive_rate']:.3f}  (编造答案率，越低越好)")
    print(f"  false_negative_rate  : {metrics['false_negative_rate']:.3f}  (错误拒答率，越低越好)")
    print(f"  avg_latency_ms       : {metrics['avg_latency_ms']:.1f}")
    _print_usage(metrics)
    if "ragas" in metrics:
        r = metrics["ragas"]
        print(f"  faithfulness         : {r['faithfulness']:.3f}  (答案忠实度)")
        print(f"  answer_relevancy     : {r['answer_relevancy']:.3f}  (答案相关性)")

    _save()
    print(f"\nSaved → {out_path}")

    pipeline.close()


# ---------------------------------------------------------------------------
# --compare 工具
# ---------------------------------------------------------------------------

def _print_usage(m: dict, indent: str = "  ") -> None:
    """token 用量。服务端没给 usage 时明说，不要让一排 0 冒充"没花钱"。"""
    if "avg_prompt_tokens" not in m:
        return
    reported = m.get("usage_reported", 0.0)
    if reported == 0.0:
        print(f"{indent}tokens               : 服务端未返回 usage（无法计量）")
        return
    line = (f"{indent}avg_tokens           : in {m['avg_prompt_tokens']:.0f} / "
            f"out {m['avg_completion_tokens']:.0f}")
    if m.get("avg_reasoning_tokens"):
        line += f"（其中 thinking {m['avg_reasoning_tokens']:.0f}）"
    print(line)
    print(f"{indent}total_tokens         : {m['total_tokens']:,}")
    if reported < 1.0:
        print(f"{indent}  ⚠️ 只有 {reported:.0%} 的样本拿到了 usage，均值偏低")


def _print_gen_result(label: str, m: dict) -> None:
    print(f"\n  {label}")
    print(f"    semantic_hit_rate  : {m['semantic_hit_rate']:.3f}  (threshold={m.get('sim_threshold', '?')})")
    if "semantic_hit_rate_cosine" in m:
        print(f"      └ 仅 cosine      : {m['semantic_hit_rate_cosine']:.3f}  "
              f"(仅靠包含认出 {m.get('semantic_contain_only', 0)} 条)")
    else:
        print("      └ 仅 cosine      : 同上（这个文件早于包含判据，semantic_hit_rate 就是纯 cosine）")
    print(f"    false_positive_rate: {m['false_positive_rate']:.3f}")
    print(f"    false_negative_rate: {m['false_negative_rate']:.3f}")
    print(f"    avg_latency_ms     : {m['avg_latency_ms']:.1f}")
    _print_usage(m, indent="    ")
    if "ragas" in m:
        print(f"    faithfulness       : {m['ragas']['faithfulness']:.3f}")
        print(f"    answer_relevancy   : {m['ragas']['answer_relevancy']:.3f}")


def _metric(m: dict, key: str) -> float:
    if key == "semantic_hit_rate_cosine" and key not in m:
        return m.get("semantic_hit_rate", 0.0)
    return m.get(key, 0.0)


def _print_gen_diff(label_a: str, ma: dict, label_b: str, mb: dict) -> None:
    rows = [
        ("semantic_hit_rate",   "semantic_hit_rate",   False),
        # 两臂配置相同时，这一行必须不动——它一动就说明改的不只是尺子。
        ("semantic_hit_rate_cosine", "  └ 仅 cosine",   False),
        ("false_positive_rate", "false_positive_rate", True),
        ("false_negative_rate", "false_negative_rate", True),
        ("avg_latency_ms",      "avg_latency_ms",      True),
        ("avg_prompt_tokens",     "avg_prompt_tokens",     True),
        ("avg_completion_tokens", "avg_completion_tokens", True),
        ("avg_reasoning_tokens",  "avg_reasoning_tokens",  True),
    ]
    print(f"\n  {'Metric':<22} {label_a[:18]:>18} {label_b[:18]:>18} {'Delta':>8}")
    print("  " + "-" * 70)
    for key, display, lower_is_better in rows:
        # 早于包含判据的结果文件没有 *_cosine 字段，但它的 semantic_hit_rate
        # 本来就是纯 cosine 测出来的——回落到它，别拿 0.0 去做差。
        va, vb = _metric(ma, key), _metric(mb, key)
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"  {display:<22} {va:>18.3f} {vb:>18.3f} {sign+f'{delta:.3f}':>8}")
    if "ragas" in ma and "ragas" in mb:
        for key, display in [("faithfulness", "faithfulness"), ("answer_relevancy", "answer_relevancy")]:
            va, vb = ma["ragas"].get(key, 0.0), mb["ragas"].get(key, 0.0)
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            print(f"  {display:<22} {va:>18.3f} {vb:>18.3f} {sign+f'{delta:.3f}':>8}")


def _exact_binom_two_sided(b: int, c: int) -> float:
    """McNemar 精确检验：翻面共 n=b+c 次，问 b 是否偏离 n/2。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)


def _print_single_variable_check(pa: dict, pb: dict) -> None:
    """两臂的 provenance 差在哪几个字段。差超过一个，单变量就不成立。"""
    skip = {"ts", "timestamp", "run_id", "git_commit"}
    diff = [(k, pa.get(k), pb.get(k)) for k in sorted(set(pa) | set(pb))
            if k not in skip and pa.get(k) != pb.get(k)]
    print("\n  --- 单变量检查 ---")
    if not diff:
        print("    ⚠️ 两臂配置完全相同——这是同一个配置跑了两遍，"
              "测的是运行间噪声，不是效应。")
        return
    for k, va, vb in diff:
        print(f"    {k:24s} {str(va)[:26]:>26} -> {str(vb)[:26]}")
    if len(diff) > 1:
        print(f"    ⚠️ 有 {len(diff)} 个字段不同，**不是单变量实验**"
              "——差异归因不到任何一个。")


def _print_paired_diff(rows_a: list[dict], rows_b: list[dict]) -> None:
    """按 id 配对做 McNemar。

    两臂跑的是同一批样本时，比率对比是错的：绝大多数"两臂完全一样"的样本也被
    算进方差，而真正携带信息的只有翻面的那几条。这个仓库为此白跑过一次 200x2
    的实验（结论"无差异"，其实是效应量 2.5 条对上标准误 0.057，根本测不出）。
    所以配对是 --compare 的默认动作，不是可选的额外分析。
    """
    A = {r["id"]: r for r in rows_a}
    B = {r["id"]: r for r in rows_b}
    ids = sorted(set(A) & set(B))
    if not ids:
        print("\n  --- 配对对比 ---\n    两个文件没有共同样本，跳过。")
        return

    has = [i for i in ids if A[i].get("has_answer")]
    no = [i for i in ids if not A[i].get("has_answer")]
    print(f"\n  --- 配对对比（McNemar，共同样本 {len(ids)} 条："
          f"有答案 {len(has)} / 无答案 {len(no)}）---")
    print(f"    {'指标':<20}{'子集':>8}{'只有B好':>9}{'只有A好':>9}{'净':>6}{'p':>8}")
    print("    " + "-" * 60)

    # 新旧尺子混在一起会让 semantic_hit 这一行比的是两把尺子而不是两个配置。
    # 旧文件的 semantic_hit 就是纯 cosine，所以 cosine 那一格总是可比的。
    mixed = ("semantic_hit_cosine" in next(iter(A.values()), {})) !=             ("semantic_hit_cosine" in next(iter(B.values()), {}))
    if mixed:
        print("    ⚠️ 两臂用的判据不同（一边含逐字包含判据，一边没有）。"
              "semantic_hit 这一行不可比，只看 └ 仅 cosine。")

    def _cos(r: dict) -> bool:
        return bool(r.get("semantic_hit_cosine", r.get("semantic_hit")))

    # (字段, 子集, 子集名, 该字段为真是不是好事)
    for field, subset, name, good_is_true in (
        ("semantic_hit",   has, "有答案", True),
        ("__cosine__",     has, "有答案", True),
        ("false_negative", has, "有答案", False),
        ("false_positive", no,  "无答案", False),
    ):
        if not subset:
            continue
        get = _cos if field == "__cosine__" else (lambda r: bool(r.get(field)))
        only_a = sum(1 for i in subset if get(A[i]) and not get(B[i]))
        only_b = sum(1 for i in subset if get(B[i]) and not get(A[i]))
        gain, loss = (only_b, only_a) if good_is_true else (only_a, only_b)
        p = _exact_binom_two_sided(only_a, only_b)
        label = "  └ 仅 cosine" if field == "__cosine__" else field
        print(f"    {label:<20}{name:>8}{gain:>9}{loss:>9}{gain - loss:>+6}{p:>8.3f}")

    deltas = sorted(B[i].get("latency_ms", 0.0) - A[i].get("latency_ms", 0.0)
                    for i in ids)
    med = deltas[len(deltas) // 2]
    faster = sum(1 for d in deltas if d < 0)
    print(f"\n    延迟逐条之差（B−A）中位 {med:+.0f}ms，{faster}/{len(deltas)} 条 B 更快")


def compare_gen_files(paths: list[str]) -> None:
    results = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        results.append((Path(p).name, data))

    print("=== Generation Eval Compare ===")
    for label, d in results:
        _print_gen_result(label, d["metrics"])

    if len(results) != 2:
        return
    (la, da), (lb, db) = results
    print("\n  --- Diff (B - A) ---")
    _print_gen_diff(la, da["metrics"], lb, db["metrics"])
    _print_single_variable_check(da.get("provenance", {}), db.get("provenance", {}))
    if da.get("per_item") and db.get("per_item"):
        _print_paired_diff(da["per_item"], db["per_item"])
    else:
        print("\n  --- 配对对比 ---\n    结果文件里没有 per_item，跳过。")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RAG generation quality")
    p.add_argument("--qa",       default="data/qa_cuad.jsonl")
    p.add_argument("--limit",    type=int, default=50, help="0 = 全量")
    p.add_argument("--table",    default=None,          help="覆盖 config.yaml 中的 database.table")
    p.add_argument("--reranker", action="store_true",   help="开启 reranker")
    p.add_argument("--ragas",          action="store_true", help="同时运行 RAGAS 评估（需安装 ragas）")
    p.add_argument("--ragas-limit",    type=int, default=20, help="RAGAS 评估样本数，默认 20")
    p.add_argument("--sim-threshold",  type=float, default=0.75, help="语义相似度命中阈值，默认 0.75")
    p.add_argument("--generate-k",     type=int,   default=0,    help="喂给生成模型的 chunk 数；0 = 跟随 top_k")
    p.add_argument("--top-k",          type=int,   default=0,    help="检索返回条数，覆盖 config.yaml；0 = max(config, generate_k)")
    p.add_argument("--corpus",         action="store_true",      help="不按 doc_id 过滤，全库 corpus 检索")
    p.add_argument("--thinking",       dest="thinking", action="store_true",  default=None,
                   help="强制开启生成模型的 thinking，覆盖 config.yaml")
    p.add_argument("--no-thinking",    dest="thinking", action="store_false",
                   help="强制关闭生成模型的 thinking，覆盖 config.yaml")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="对比两个 gen_eval 结果文件，不运行新评估")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Windows 控制台默认 GBK，遇到 emoji 会抛 UnicodeEncodeError —— 重设为 utf-8。
    # 与 eval_experiment.py 的处理保持一致。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.compare:
        compare_gen_files(args.compare)
    else:
        run_eval(args)
