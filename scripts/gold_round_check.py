"""
逐轮对照 judge 的判断与 CUAD gold span —— 规格 `docs/agentic_loop_upgrade.md` 第 3 节。

**零人工标注成本**：CUAD 每个有答案的问题都带 gold span（原文字符偏移），而 trace
里每轮都记了累积 chunks 的 `start` / `end`。两者一比就知道"这一轮的上下文里到底有
没有答案"，再和 judge 说的 `sufficient` 对照，就是那张 2×2 表：

| gold span 在累积 chunks 里 | judge 说够了 | 结论 |
|---|---|---|
| ✅ | ✅ | 正确停止 |
| ✅ | ❌ | **假阴性 → 白烧一轮**（成本 ×轮数） |
| ❌ | ✅ | **假阳性 → 提前停止，答案必错** |
| ❌ | ❌ | 正确继续 |

这两类失败是本次改造制造出来的（规格第 3 节的 2 和 3），也是唯一有全自动标签的
两类，所以它们是失败归因最先该看的地方。

纯离线：只读 trace JSONL 与 qa 文件，不碰数据库、不调 LLM。

用法：
    uv run scripts/gold_round_check.py data/runs/traces/<run>.jsonl
    uv run scripts/gold_round_check.py data/runs/traces/*.jsonl      # 多组对照
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from lex_rag.trace_sink import read_meta, read_traces


def _spans_by_id(qa_path: Path) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    with qa_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = [(s["start"], s["end"]) for s in r.get("spans", [])]
    return out


def _gold_in_chunks(chunks: list[dict], spans: list[tuple[int, int]],
                    doc_id: str | None) -> bool:
    """与 `lex_rag.evals._hit` 同一套重叠判定，保持口径一致。

    多文档（corpus）模式下必须按 doc_id 过滤：别的合同里凑巧落在同一字符区间的
    chunk 不算命中，不过滤会把假阳性算成正确停止。
    """
    for c in chunks:
        if doc_id is not None and c.get("doc_id") != doc_id:
            continue
        cs, ce = c.get("start"), c.get("end")
        if cs is None or ce is None:
            continue
        for ss, se in spans:
            if cs <= ss < ce or cs < se <= ce:
                return True
    return False


def analyse(path: Path, spans_by_id: dict[str, list[tuple[int, int]]]) -> dict:
    traces = read_traces(path)
    meta = read_meta(path)

    cell = Counter()          # (gold_in, sufficient) -> n，按轮计
    per_round_cell: dict[int, Counter] = {}
    term = Counter()
    n_rounds_hist = Counter()
    no_span = skipped = 0
    # 循环结束时的最终状态，按查询计（而不是按轮）
    final = Counter()
    wasted_rounds = 0         # gold 已在里面却继续跑的轮次总数

    for t in traces:
        term[t.get("terminated_by")] += 1
        n_rounds_hist[t.get("n_rounds", 0)] += 1
        tmeta = t.get("meta") or {}
        qid = tmeta.get("id")
        spans = spans_by_id.get(qid or "", [])
        if not spans:
            # 无答案样本没有 gold span，"不在里面"是恒真的，混进 2×2 会把假阳性
            # 稀释成一个好看但没意义的数。它们单独看 out_of_scope。
            no_span += 1
            continue

        # corpus scope 下 trace 的 doc_id 是 None（本来就不按文档过滤检索），
        # 但 gold span 的字符偏移只在它自己那份合同里有意义。不回填 gold_doc_id
        # 的话，别的合同里凑巧落在同一区间的 chunk 会被算成命中——那会抬高"正确
        # 停止"、同时把假阳性藏起来，正好毁掉这张表要测的东西。
        doc_id = t.get("doc_id") or tmeta.get("gold_doc_id")
        # trace 落盘的是整个累积池，但 judge 和生成层都只看前 k 个
        # （`agent.py`: judge(question, pool[:k]) / yield pool[:k]）。
        # 按整池判命中会把"gold 排在第 11 位、judge 根本没看见"算成白烧，
        # 冤枉判定器。实测这个偏差占白烧的 4%（白烧率 0.414 → 0.404）——
        # 不大，但口径必须是"判定器实际看到的东西"，否则这张表测的不是它。
        k = int(tmeta.get("k") or 0) or None
        for r in t.get("rounds", []):
            v = r.get("verdict")
            if not v:
                skipped += 1
                continue
            gold = _gold_in_chunks(r.get("chunks", [])[:k], spans, doc_id)
            key = (gold, bool(v.get("sufficient")))
            cell[key] += 1
            per_round_cell.setdefault(r.get("index", 0), Counter())[key] += 1
            if gold and not v.get("sufficient"):
                wasted_rounds += 1

        rounds = t.get("rounds") or []
        if rounds and rounds[-1].get("verdict"):
            gold = _gold_in_chunks(rounds[-1].get("chunks", [])[:k], spans, doc_id)
            final[(gold, t.get("terminated_by"))] += 1

    return {
        "path": str(path), "meta": meta, "n_traces": len(traces),
        "cell": cell, "per_round_cell": per_round_cell, "term": term,
        "n_rounds_hist": n_rounds_hist, "no_span": no_span, "skipped": skipped,
        "final": final, "wasted_rounds": wasted_rounds,
    }


def _rates(cell: Counter) -> dict:
    cs, wa = cell[(True, True)], cell[(True, False)]
    pm, cc = cell[(False, True)], cell[(False, False)]
    total = cs + wa + pm + cc
    gold_in, gold_out = cs + wa, pm + cc
    return {
        "correct_stop": cs, "wasted_fn": wa, "premature_fp": pm, "correct_continue": cc,
        "n": total,
        "accuracy": (cs + cc) / total if total else None,
        # 分母刻意分开：假阴性只在"gold 已在里面"的轮次上有定义，假阳性反之。
        "fn_rate": wa / gold_in if gold_in else None,
        "fp_rate": pm / gold_out if gold_out else None,
    }


def _fmt(v, nd=3) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def report(res: dict) -> None:
    r = _rates(res["cell"])
    cfg = res["meta"].get("config", {})
    print(f"\n{'=' * 78}")
    print(f"{Path(res['path']).name}   {cfg.get('label', '')}")
    print(f"  配置: scope={cfg.get('scope')} reranker={cfg.get('reranker')} "
          f"selector={cfg.get('selector')} max_iterations={cfg.get('max_iterations')}")
    print(f"  {res['n_traces']} 条 trace，其中有 gold span 的 {res['n_traces'] - res['no_span']} 条"
          f"（无答案 {res['no_span']} 条不计入 2×2）")

    print(f"\n  {'':22s}{'judge 说够了':>14s}{'judge 说不够':>14s}")
    print(f"  {'gold 在累积里':22s}{r['correct_stop']:>14d}{r['wasted_fn']:>14d}"
          f"   ← 右列是白烧")
    print(f"  {'gold 不在':22s}{r['premature_fp']:>14d}{r['correct_continue']:>14d}"
          f"   ← 左列是提前停止")
    print(f"\n  accuracy={_fmt(r['accuracy'])}  "
          f"假阴性率(白烧)={_fmt(r['fn_rate'])}  假阳性率(提前停止)={_fmt(r['fp_rate'])}")
    print(f"  白烧轮次合计 {res['wasted_rounds']}（gold 已到手却仍继续的轮数）")

    if res["per_round_cell"]:
        print(f"\n  {'轮次':6s}{'正确停止':>10s}{'白烧':>8s}{'提前停止':>10s}{'正确继续':>10s}")
        for idx in sorted(res["per_round_cell"]):
            c = _rates(res["per_round_cell"][idx])
            print(f"  第{idx}轮{'':2s}{c['correct_stop']:>10d}{c['wasted_fn']:>8d}"
                  f"{c['premature_fp']:>10d}{c['correct_continue']:>10d}")

    print("\n  终止原因: " + "  ".join(f"{k}={v}" for k, v in res["term"].most_common()))
    print("  轮数分布: " + "  ".join(f"{k}轮={v}" for k, v in sorted(res["n_rounds_hist"].items())))
    if res["skipped"]:
        print(f"  [warn] {res['skipped']} 轮没有 verdict（多为被防重复拦下的轮），已跳过")


def compare(results: list[dict]) -> None:
    print(f"\n{'=' * 78}")
    print("三组对照")
    print(f"{'语料':34s}{'accuracy':>10s}{'白烧率':>10s}{'提前停止率':>12s}{'平均轮数':>10s}")
    print("-" * 78)
    for res in results:
        r = _rates(res["cell"])
        hist = res["n_rounds_hist"]
        avg = sum(k * v for k, v in hist.items()) / max(sum(hist.values()), 1)
        label = res["meta"].get("config", {}).get("label") or Path(res["path"]).stem
        print(f"{label[:34]:34s}{_fmt(r['accuracy']):>10s}{_fmt(r['fn_rate']):>10s}"
              f"{_fmt(r['fp_rate']):>12s}{avg:>10.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="judge 判断 vs CUAD gold span 的逐轮对照")
    ap.add_argument("traces", nargs="+", help="trace JSONL（可用通配符）")
    ap.add_argument("--qa", default="data/qa_cuad.jsonl")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    paths: list[Path] = []
    for pat in args.traces:
        paths.extend(Path(p) for p in sorted(glob.glob(pat)) or ([pat] if Path(pat).exists() else []))
    if not paths:
        raise SystemExit(f"没有匹配到 trace 文件：{args.traces}")

    spans = _spans_by_id(Path(args.qa))
    results = [analyse(p, spans) for p in paths]
    for res in results:
        report(res)
    if len(results) > 1:
        compare(results)


if __name__ == "__main__":
    main()
