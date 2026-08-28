"""
产出 trace 语料 —— 规格 `docs/agentic_loop_upgrade.md` 第 2.6 节。

三组配置跑同一批 CUAD 问题，天然构成对照：

    baseline    默认配置（LLM 选策略，开 reranker）
    no_rerank   关掉 reranker，其余相同
    fixed       固定策略阶梯（hybrid → bm25 → hyde），**不让系统自己选**

第三组是关键对照：它保留了多轮，只去掉"LLM 选策略"这一件事。有了它才能把
"多跑几轮有用"和"让模型自己挑策略有用"分开——只跟单轮比的话，这两者混在一起。

## scope 默认 contract，尽管 contract 下策略空间更小

两个 scope 各有硬伤，但性质不同：

* **corpus**：策略确实有空间（bm25 vs vector 重合度 0.333），但 **CUAD 的问题不
  标识文档**——1000 条样本只有 41 个不同的问题文本，每条文本原样出现在 24~25 份
  合同里。查询里没有任何信息能指出该找哪一份，任务在信息上不可解。实测 12 条
  有答案样本里只有 1 轮命中 gold span（contract scope 下是 16 轮）。
* **contract**：任务可解（hit@10=0.865），但 17/25 的合同 chunk 总数 <= fetch_k=60，
  候选池就是整份合同，换策略不改变 reranker 的输入——五个动作实际塌缩成两个。

选 contract，理由是**这份语料的用途**：规格第 6 节说它要给下游 tracelens 当带标签的
验证语料做失败归因。corpus scope 下每一条"失败"都是数据集不可解造成的伪失败，拿去
做归因等于污染标签。策略空间小是要如实记录的性质，不是换 scope 的理由。

`--scope corpus` 仍然保留，用于复现上面那个对照。细节见 `docs/experiments.md`。

## 并发

每个 worker 一个 `RAGPipeline`：psycopg 连接不能多线程共用。embedding 缓存反过来
要共享（`attach_shared_cache`），否则同一段文本被每个 worker 各嵌一次。

**并发度的上限不是 CPU，是配额。** 每条查询大约 4 次 LLM 调用（选择器 0~2 +
judge 1~3 + HyDE 0~1），按 120 rpm 算约 30 条/分钟，4 个 worker 就打满了，再加
只会换来 429。

用法：
    uv run scripts/build_trace_corpus.py --limit 200 --concurrency 4
    uv run scripts/build_trace_corpus.py --limit 0 --concurrency 4        # 全量
    uv run scripts/build_trace_corpus.py --limit 0 --scope contract       # 对照
    uv run scripts/build_trace_corpus.py --only fixed                     # 单跑一组

结果写入 data/runs/traces/<ts>_<组名>.jsonl，随后用
`scripts/gold_round_check.py data/runs/traces/<ts>_*.jsonl` 出 2×2 表。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

OUT_DIR = Path("data/runs/traces")


class FixedLadderSelector:
    """第三组用：按固定顺序换策略，完全不看问题、也不看 judge 说缺什么。

    这才是"不让系统自己选"的正确对照——它保留多轮和防重复，只抽掉决策。
    与单轮对照相比，它能把"多跑几轮"与"选得准"这两个效应分开。
    """

    def __init__(self, ladder: list[str] | None = None) -> None:
        from lex_rag.agent import _ACTIONS
        self.ladder = ladder or ["bm25", "hyde", "multi_query", "vector"]
        self._actions = _ACTIONS

    def select(self, question, base, tried, missing, hint):
        tried_keys = {k for k, _ in tried}
        for action in self.ladder:
            _, mutate = self._actions.get(action, (None, None))
            if mutate is None:
                continue
            st = mutate(base)
            if st.key() not in tried_keys:
                return st, f"固定阶梯：{action}", None, None
        return None, "固定阶梯已走完", None, None


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def build_configs(cfg, args) -> list[dict]:
    """三组配置。每组只改一个变量，否则对照不成立。"""
    all_cfgs = {
        "baseline": {
            "label": "baseline（LLM 选策略 + reranker）",
            "cfg": replace(cfg, reranker=replace(cfg.reranker, enabled=True)),
            "selector": "llm",
        },
        "no_rerank": {
            "label": "no_rerank（LLM 选策略，关 reranker）",
            "cfg": replace(cfg, reranker=replace(cfg.reranker, enabled=False)),
            "selector": "llm",
        },
        "fixed": {
            "label": "fixed（固定策略阶梯 + reranker）",
            "cfg": replace(cfg, reranker=replace(cfg.reranker, enabled=True)),
            "selector": "fixed_ladder",
        },
    }
    names = args.only or list(all_cfgs)
    unknown = [n for n in names if n not in all_cfgs]
    if unknown:
        raise SystemExit(f"未知配置名 {unknown}，可选：{list(all_cfgs)}")
    return [dict(all_cfgs[n], name=n) for n in names]


def run_one_config(spec: dict, items, args, ts: str, shared_cache: dict) -> Path:
    from lex_rag.agent import AgenticPipeline
    from lex_rag.pipeline import RAGPipeline
    from lex_rag.trace_sink import TraceSink

    cfg = spec["cfg"]
    out = OUT_DIR / f"{ts}_{spec['name']}.jsonl"
    sink = TraceSink(out, config={
        "label": spec["label"], "group": spec["name"],
        "scope": args.scope, "reranker": cfg.reranker.enabled,
        "selector": spec["selector"], "max_iterations": args.max_iterations,
        "table": cfg.database.table, "top_k": args.top_k,
        "generation_model": cfg.contextual.model,
        "embedding_model": cfg.embedding.model,
        "reranker_model": cfg.reranker.model if cfg.reranker.enabled else None,
        "n_items": len(items), "git_commit": _git_commit(),
    })

    # 每个线程一套 pipeline/agent：psycopg 连接不能多线程共用。
    #
    # **必须在起线程之前串行建好。** VectorStore.__init__ 会跑 _init_schema()
    # （CREATE EXTENSION / CREATE TABLE / 建索引），多个 worker 同时对同一张表做
    # DDL 会互相等 AccessShareLock 直到死锁——实测 4 worker 跑 20 条时，1 次死锁
    # 之后连锁出 15 条 InFailedSqlTransaction。第一个之后的都传 init_schema=False，
    # 表已经建好了，没必要每条连接都重跑一遍 DDL。
    pipes: list = []
    for _ in range(args.concurrency):
        pipe = RAGPipeline(cfg)
        pipe.embedder.attach_shared_cache(shared_cache)
        pipes.append(pipe)

    local = threading.local()
    pipe_queue: list = list(pipes)
    queue_lock = threading.Lock()

    def _agent():
        if getattr(local, "agent", None) is None:
            with queue_lock:
                pipe = pipe_queue.pop()
            selector = FixedLadderSelector() if spec["selector"] == "fixed_ladder" else None
            local.agent = AgenticPipeline(
                pipe, cfg.contextual, max_iterations=args.max_iterations,
                sink=sink, selector=selector,
                select_first_round=args.select_first_round,
            )
        return local.agent

    def _one(item):
        doc_id = None if args.scope == "corpus" else item.doc_id
        try:
            # meta 里带 gold 标签，让语料自包含：下游 tracelens 与
            # gold_round_check 不必再回头 join qa 文件。
            _, trace = _agent().query(
                item.question, doc_id=doc_id, k=args.top_k,
                meta={"id": item.id, "has_answer": item.has_answer,
                      "gold_doc_id": item.doc_id, "n_gold_spans": len(item.spans)})
            # 循环内部会把检索/判定的异常接住并记成 terminated_by=error，不会抛到
            # 这里。只统计抛出来的那些，就会报出"失败 0"而语料里其实有几十条 error
            # ——上一轮全量正是这样（报 0，实际 baseline 14 条、fixed 44 条）。
            if trace and trace[-1] == "terminated_by=error":
                return "loop_error（详见 trace 内的 step.error）"
            return None
        except Exception as e:               # noqa: BLE001 — 一条失败不该毁掉整组
            return f"{type(e).__name__}: {e}"

    print(f"\n=== {spec['label']} → {out.name} ===", flush=True)
    t0 = time.perf_counter()
    errors: list[str] = []

    # 每条查询的 meta 里带上 gold 标签，让语料自包含——下游 tracelens 不必再去
    # join qa 文件就能知道这条问题有没有答案。
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for err in tqdm(pool.map(_one, items), total=len(items),
                        desc=spec["name"], unit="q"):
            if err:
                errors.append(err)

    sink.close()
    for pipe in pipes:
        with contextlib.suppress(Exception):
            pipe.close()

    dt = time.perf_counter() - t0
    print(f"  完成 {len(items)} 条，用时 {dt / 60:.1f} 分钟"
          f"（{dt / max(len(items), 1):.1f}s/条），失败 {len(errors)}", flush=True)
    if errors:
        from collections import Counter
        for kind, n in Counter(e.split(":")[0] for e in errors).most_common(5):
            print(f"    {kind}: {n}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="产出三组 trace 语料（规格 2.6）")
    ap.add_argument("--qa", default="data/qa_cuad.jsonl")
    ap.add_argument("--limit", type=int, default=200, help="0 = 全量 1000 条")
    ap.add_argument("--scope", choices=("contract", "corpus"), default="contract",
                    help="corpus 下 CUAD 的问题不标识文档，任务不可解；见模块 docstring")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="上限由配额决定（约 120 rpm ÷ 4 次调用/条），加大只会换来 429")
    # 默认与 AgenticPipeline 对齐（2）：第 2 轮从未救回过任何一条，
    # 见 docs/experiments.md。要复现旧语料就显式传 --max-iterations 3。
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=0,
                    help="0 = 用 config.yaml 的 retrieval.top_k（别再写死，会与配置漂移）")
    ap.add_argument("--select-first-round", action="store_true",
                    help="首轮也调选择器（默认首轮用默认策略）")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="只跑指定的组：baseline / no_rerank / fixed")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from lex_rag.config import load_config
    from lex_rag.cuad import load_qa
    from lex_rag.embeddings import _DEFAULT_CACHE, EmbeddingClient

    cfg = load_config()
    args.top_k = args.top_k or cfg.retrieval.top_k
    items = load_qa(Path(args.qa))
    if args.limit > 0:
        items = items[: args.limit]

    # 缓存加载一次，三组共用：同一批问题的 embedding 不该被算三遍。
    shared_cache = EmbeddingClient(cfg.embedding, cache_path=_DEFAULT_CACHE)._cache

    specs = build_configs(cfg, args)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} 条 × {len(specs)} 组，scope={args.scope}，"
          f"concurrency={args.concurrency}，run={ts}", flush=True)

    paths = [run_one_config(s, items, args, ts, shared_cache) for s in specs]

    # 跑完在主线程存一次 embedding 缓存（worker 里是关掉落盘的）
    saver = EmbeddingClient(cfg.embedding, cache_path=_DEFAULT_CACHE)
    saver._cache = shared_cache
    saver._save_cache()

    print("\n产出：")
    for p in paths:
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"\n下一步：uv run scripts/gold_round_check.py {OUT_DIR}/{ts}_*.jsonl")

    (OUT_DIR / f"{ts}_manifest.json").write_text(json.dumps({
        "run_id": ts, "scope": args.scope, "n_items": len(items),
        "groups": [s["name"] for s in specs], "files": [str(p) for p in paths],
        "git_commit": _git_commit(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
