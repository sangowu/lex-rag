"""
Generation 层评估 —— Langfuse Dataset + Experiment 模式（行业标准评估工作流）。

与 ``eval_generation.py`` 的区别：
- ``eval_generation.py``：本地跑批，指标打印到终端 / 存 JSON，靠 CLAUDE.md 手工维护 v1→v4 对比表；
- 本脚本：把测试集固定成一个 **Langfuse Dataset**，每次评估作为一个 **Experiment run**，
  faithfulness / answer_relevancy / semantic_similarity 作为 **scores** 挂在每个 item 的 trace 上，
  → Langfuse UI 里可**跨 run 对比回归**（v3→v4 哪些样本掉分、掉在哪、judge 给的理由是什么）。

前置（强依赖，与失败安全的 ``tracing.py`` 不同——评估就是要把结果送上去）::

    .env 配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST

用法::

    # 首次：把 QA 测试集同步成 Langfuse Dataset（幂等，可重复跑）
    uv run scripts/eval_experiment.py --sync-dataset --limit 30

    # 跑一次实验（给这次配置起个可辨识的 run 名，便于 UI 里对比）
    uv run scripts/eval_experiment.py --limit 30 --reranker --run-name v4-fewshot-doc-meta

评估集只收录 ``has_answer=True`` 的样本 —— 这三个都是"答得好不好"的生成质量指标，
与 CLAUDE.md 里 RAGAS v1→v4 的口径一致；拒答准确率（需 no-answer 样本）仍由 eval_generation.py 覆盖。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

from lex_rag.config import load_config
from lex_rag.cuad import load_qa, QAItem
from lex_rag.embeddings import EmbeddingClient
from lex_rag.generator import LegalGenerator
from lex_rag.pipeline import RAGPipeline
from lex_rag import tracing

DATASET_NAME = "cuad-gen-eval"

# LLM-as-Judge 提示词（与 eval_generation.py 语义一致，保持本脚本自包含）
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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def _item_id(item: QAItem) -> str:
    """内容哈希做稳定 id → 重复 sync 时 upsert 而非新增，保证 Dataset 幂等。"""
    key = f"{item.doc_id}::{item.question}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:16]


def _field(item, name):
    """兼容读取实验 item 的字段：DatasetItem 是对象（属性访问），
    LocalExperimentItem 是 dict（键访问）。run_experiment 两种都可能传入。"""
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _parse_judge(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0.5, "reason": "parse error"}


def _judge(client, model: str, prompt: str, name: str) -> dict:
    """一次 LLM-as-Judge 调用；同时记一条 generation（在 item trace 下自动嵌套，含 token）。"""
    gen = tracing.start_generation(f"judge.{name}", model, prompt)
    resp = client.models.generate_content(model=model, contents=prompt)
    in_tok, out_tok = tracing.genai_usage(resp)
    tracing.end_generation(gen, output=resp.text, input_tokens=in_tok, output_tokens=out_tok)
    return _parse_judge(resp.text)


# ---------------------------------------------------------------------------
# Dataset 同步
# ---------------------------------------------------------------------------

def sync_dataset(langfuse, qa_items: list[QAItem]) -> int:
    """把 has_answer 的 QA 样本 upsert 进 Langfuse Dataset。返回写入条数。"""
    try:
        langfuse.create_dataset(
            name=DATASET_NAME,
            description="CUAD 合同问答 · 生成质量评估集（仅 has_answer 样本）",
        )
    except Exception:
        pass  # 已存在即忽略

    n = 0
    for item in qa_items:
        if not item.has_answer:
            continue
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=_item_id(item),  # 稳定 id → 幂等 upsert
            input={"question": item.question, "doc_id": item.doc_id},
            expected_output={"answers": item.answers, "has_answer": item.has_answer},
            metadata={"qa_id": item.id},
        )
        n += 1
    langfuse.flush()
    return n


# ---------------------------------------------------------------------------
# Task（生成）与 Evaluators（评分）
# ---------------------------------------------------------------------------

def build_task(cfg, pipeline: RAGPipeline, generator: LegalGenerator,
               corpus: bool, generate_k: int):
    """返回 run_experiment 的 task：对一个 item 做检索+生成，产出待评估的 output。

    task 内的 generator.generate 已被 tracing 记为 generation，自动挂到该 item 的 trace 下。
    """
    def task(*, item, **kwargs) -> dict:
        data = _field(item, "input") or {}
        question = data["question"]
        doc_id = None if corpus else data.get("doc_id")
        chunks = pipeline.query(question, k=cfg.retrieval.top_k, doc_id=doc_id)
        metas = pipeline.get_doc_metas_for_chunks(chunks)
        gen_chunks = chunks[:generate_k]
        if corpus:
            result = generator.generate(question, gen_chunks, metas=metas or None)
        else:
            result = generator.generate(
                question, gen_chunks,
                meta=metas.get(data.get("doc_id")) if metas else None,
            )
        return {
            "answer": result.answer or "",
            "contexts": [c.text for c in chunks],
            "refused": bool(result.is_refused),
            "error": result.error,
        }

    return task


def build_evaluators(cfg, judge_client, embedder: EmbeddingClient):
    """返回三个 evaluator：faithfulness / answer_relevancy / semantic_similarity。"""
    from langfuse.experiment import Evaluation

    def faithfulness(*, input, output, expected_output, metadata, **kwargs):
        answer = (output or {}).get("answer", "")
        if output.get("refused") or not answer:
            # 空答案不含任何断言 → 无幻觉可言，按惯例记满分并注明
            return Evaluation(name="faithfulness", value=1.0,
                              comment="refused/empty → 无断言，视为无幻觉")
        context = "\n---\n".join(output.get("contexts", []))[:3000]
        r = _judge(judge_client, cfg.ragas.model,
                   _FAITHFULNESS_PROMPT.format(question=input["question"],
                                               context=context, answer=answer),
                   "faithfulness")
        return Evaluation(name="faithfulness", value=float(r.get("score", 0.5)),
                          comment=str(r.get("reason", "")))

    def answer_relevancy(*, input, output, expected_output, metadata, **kwargs):
        answer = (output or {}).get("answer", "")
        if output.get("refused") or not answer:
            return Evaluation(name="answer_relevancy", value=0.0,
                              comment="refused/empty → 未回答问题")
        r = _judge(judge_client, cfg.ragas.model,
                   _RELEVANCY_PROMPT.format(question=input["question"], answer=answer),
                   "answer_relevancy")
        return Evaluation(name="answer_relevancy", value=float(r.get("score", 0.5)),
                          comment=str(r.get("reason", "")))

    def semantic_similarity(*, input, output, expected_output, metadata, **kwargs):
        answer = (output or {}).get("answer", "")
        golds = [g for g in (expected_output or {}).get("answers", []) if g and g.strip()]
        if not answer or not golds:
            return Evaluation(name="semantic_similarity", value=0.0,
                              comment="no answer/gold")
        vecs = embedder.embed_texts([answer] + golds)
        max_sim = max(_cosine(vecs[0], gv) for gv in vecs[1:])
        return Evaluation(name="semantic_similarity", value=round(max_sim, 4))

    return [faithfulness, answer_relevancy, semantic_similarity]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Langfuse Dataset + Experiment 生成层评估")
    ap.add_argument("--qa", default="data/qa_cuad.jsonl", help="QA 测试集 jsonl")
    ap.add_argument("--limit", type=int, default=30, help="评估样本上限（0=全部）")
    ap.add_argument("--reranker", action="store_true", help="启用 reranker")
    ap.add_argument("--corpus", action="store_true", help="corpus 模式（不按 doc_id 过滤）")
    ap.add_argument("--generate-k", type=int, default=8, help="送入生成的 chunk 数")
    ap.add_argument("--run-name", default=None, help="本次实验 run 名（便于 UI 对比）")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="并发度。默认 1（串行）：RAGPipeline/EmbeddingClient 共享且其 "
                         "pickle 缓存非线程安全，>1 会导致缓存文件损坏 / KeyError。"
                         "如需提速，须先把 embedding 缓存改成线程安全再调高。")
    ap.add_argument("--sync-dataset", action="store_true",
                    help="先把 QA 集 upsert 进 Langfuse Dataset 再评估")
    args = ap.parse_args()

    # Windows 控制台默认 GBK，无法打印 Langfuse 汇总里的 emoji（💡）与本脚本的 ✅ —— 重设为 utf-8
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = load_config()  # 内部 load_dotenv()，先于下面读取 LANGFUSE_* / 建 client
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        raise SystemExit(
            "本脚本强依赖 Langfuse：请在 .env 配置 LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY / LANGFUSE_HOST 后重试。"
        )
    if args.reranker:
        cfg = replace(cfg, reranker=replace(cfg.reranker, enabled=True))

    from google import genai
    from langfuse import get_client

    langfuse = get_client()
    pipeline = RAGPipeline(cfg)
    generator = LegalGenerator(cfg.contextual)
    judge_client = genai.Client(api_key=cfg.ragas.api_key)
    embedder = EmbeddingClient(cfg.embedding, cache_path=Path("data/embed_cache_eval.pkl"))

    # 1) 同步 Dataset（幂等）
    if args.sync_dataset:
        qa_items = load_qa(Path(args.qa))
        if args.limit > 0:
            # 只同步覆盖本次评估所需的样本；over-provision 4x 以补足 has_answer 过滤后的数量
            qa_items = qa_items[: args.limit * 4]
        n = sync_dataset(langfuse, qa_items)
        print(f"[Dataset] 已 upsert {n} 条 has_answer 样本到 '{DATASET_NAME}'")

    # 2) 取 Dataset items（跑实验的数据源）
    dataset = langfuse.get_dataset(DATASET_NAME)
    items = list(dataset.items)
    if not items:
        raise SystemExit(f"Dataset '{DATASET_NAME}' 为空，请先加 --sync-dataset 同步。")
    if args.limit > 0:
        items = items[: args.limit]

    # 3) 跑实验
    run_name = args.run_name or f"reranker={args.reranker},gen_k={args.generate_k}"
    print(f"[Experiment] run='{run_name}' · {len(items)} 样本 · concurrency={args.concurrency}")
    result = langfuse.run_experiment(
        name="CUAD generation quality",
        run_name=run_name,
        data=items,
        task=build_task(cfg, pipeline, generator, args.corpus, args.generate_k),
        evaluators=build_evaluators(cfg, judge_client, embedder),
        max_concurrency=args.concurrency,
        metadata={
            "reranker": str(args.reranker),
            "corpus": str(args.corpus),
            "generate_k": str(args.generate_k),
            "gen_model": cfg.contextual.model,
            "judge_model": cfg.ragas.model,
        },
    )

    print("\n" + result.format())
    langfuse.flush()
    pipeline.close()
    print("\n✅ 已上报 Langfuse。打开 Datasets → "
          f"'{DATASET_NAME}' → Runs 可跨 run 对比各指标。")


if __name__ == "__main__":
    main()
