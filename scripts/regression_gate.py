"""发布门禁：跑 data/regression_set.jsonl，按阈值给出 pass/fail。

    uv run scripts/regression_gate.py
    uv run scripts/regression_gate.py --cases data/regression_set.jsonl --out data/runs/gate
    uv run scripts/regression_gate.py --show                # 只打印上一次结果

退出码 0 = 通过，1 = 阻断。判定逻辑与阈值在 `lex_rag/gate.py`，那部分是纯函数、
由 CI 覆盖；这个脚本只负责"把案例跑出来"。

⚠️ **这个脚本进不了 CI**：要连 pgvector、要打 LLM、每轮要花钱。它是发布前手动
跑或放进部署流水线的那一步。理由见 gate.py 的模块文档。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lex_rag._shared import get_generator, get_pipeline          # noqa: E402
from lex_rag.chunking import ChunkWindow                          # noqa: E402
from lex_rag.gate import (  # noqa: E402
    Case, CaseResult, Thresholds, evaluate, load_cases, score_case,
)


def _injected_chunk(case: Case) -> ChunkWindow:
    """把注入文本包成一个正常长相的 chunk。

    刻意**不**标记成"这是测试"——注入攻击的全部意义就是长得像普通合同文字。
    chunk_id 用 `#inj` 后缀只是为了结果文件里能一眼认出来，模型看不到 chunk_id。
    """
    return ChunkWindow(
        chunk_id=f"{case.doc_id}#inj",
        doc_id=case.doc_id,
        text=case.injected_text,
        start=-1, end=-1,
        score=1.0, score_kind="injected",
    )


def run_case(case: Case, pipeline, generator, top_k: int) -> CaseResult:
    t0 = time.perf_counter()
    try:
        chunks = pipeline.query(case.question, k=top_k, doc_id=case.doc_id)

        if case.kind == "prompt_injection":
            # 放在最前面：如果模型对"上下文靠前的指令"更敏感，这是最不利的一档，
            # 门禁应该在最不利的一档上过，而不是在最有利的一档上过。
            chunks = [_injected_chunk(case)] + list(chunks)

        metas = pipeline.get_doc_metas_for_chunks(chunks)
        result = generator.generate(
            case.question, chunks[:top_k],
            meta=metas.get(case.doc_id) if metas else None,
        )
        return CaseResult(
            id=case.id, kind=case.kind,
            refused=result.is_refused,
            answer=result.answer or "",
            n_citations=len(result.citations),
            error=result.error,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:
        return CaseResult(id=case.id, kind=case.kind, refused=False, answer="",
                          n_citations=0, error=f"{type(e).__name__}: {e}",
                          latency_ms=(time.perf_counter() - t0) * 1000)


def _worst(case: Case, attempts: list[CaseResult]) -> CaseResult:
    """多次重复里挑**最坏**的一次作为该案例的结论。

    注入案例的"最坏"是"这一次照做了注入"——只要出现过一次就该报出来。取多数票
    会让一个 40% 概率被攻破的系统显示为安全。
    """
    # score_case 会把判定写回对象，所以判的是副本，别污染原件
    followed = [a for a in attempts
                if score_case(case, copy.copy(a)).injection_followed]
    errored = [a for a in attempts if a.error]
    chosen = errored[0] if errored else (followed[0] if followed else attempts[0])
    chosen.attempts = len(attempts)
    chosen.followed_attempts = len(followed)
    return chosen


def main() -> int:
    p = argparse.ArgumentParser(description="Release gate over the regression set")
    p.add_argument("--cases", default="data/regression_set.jsonl")
    p.add_argument("--out", default="data/runs/gate")
    p.add_argument("--top-k", type=int, default=0, help="0 = 用 config.yaml 的 retrieval.top_k")
    # 注入是不确定的：实测 8 轮里 2 轮被执行，同一条案例、同一份上下文。
    # 跑一次就报 PASS 等于把一枚硬币的一面当成结论，所以注入案例默认重复 3 次，
    # **任何一次被执行就算被执行**。安全属性的聚合方式是 any，不是多数票。
    p.add_argument("--injection-repeat", type=int, default=3,
                   help="每条注入案例重复跑几次（任一次生效即判为生效）")
    p.add_argument("--show", metavar="FILE", nargs="?", const="latest",
                   help="不跑新一轮，打印已有结果文件（latest = 最近一次）")
    # 阈值可以在命令行放宽，但**每一次放宽都会写进结果文件**，
    # 免得"临时放宽一下"变成永久且无人知晓。
    for f, default in Thresholds().__dict__.items():
        p.add_argument(f"--{f.replace('_', '-')}", type=int, default=default)
    args = p.parse_args()

    if args.show:
        return _show(args.show, args.out)

    cases = load_cases(args.cases)
    pipeline, generator = get_pipeline(), get_generator()
    top_k = args.top_k or pipeline.cfg.retrieval.top_k

    results = []
    for i, case in enumerate(cases, 1):
        reps = args.injection_repeat if case.kind == "prompt_injection" else 1
        attempts = [run_case(case, pipeline, generator, top_k) for _ in range(max(1, reps))]
        r = _worst(case, attempts)
        results.append(r)
        flag = "ERR" if r.error else ("refused" if r.refused else f"{r.n_citations} cites")
        rep_note = f" {r.followed_attempts}/{r.attempts} followed" if reps > 1 else ""
        print(f"  [{i:>2}/{len(cases)}] {case.kind:<16} {flag:<12}{rep_note} {case.id[:56]}",
              file=sys.stderr)

    th = Thresholds(**{f: getattr(args, f) for f in Thresholds().__dict__})
    report = evaluate(cases, results, th)
    print()
    print(report.summary())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{ts}.json"
    payload = {
        "run_id": ts,
        "cases_file": args.cases,
        "top_k": top_k,
        "injection_repeat": args.injection_repeat,
        "passed": report.passed,
        "counts": report.counts,
        "thresholds": report.thresholds,
        "violations": report.violations,
        "per_case": report.per_case,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {path}")
    return 0 if report.passed else 1


def _show(which: str, out_dir: str) -> int:
    path = Path(which)
    if which == "latest":
        files = sorted(Path(out_dir).glob("*.json"))
        if not files:
            print(f"no gate results under {out_dir}", file=sys.stderr)
            return 1
        path = files[-1]
    d = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path}  ->  {'PASS' if d['passed'] else 'FAIL'}")
    for k, v in d["counts"].items():
        print(f"  {k:<20} {v} / {d['thresholds'].get('max_' + k, '-')}")
    for v in d["violations"]:
        print(f"  ! {v}")
    return 0 if d["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
