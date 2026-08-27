"""gold_round_check 的 2×2 统计测试。

这个脚本没有网络也没有数据库，但它算错的后果比别处都大：整个失败归因的结论都
建立在这张表上。重点测两件容易悄悄出错的事——

1. **corpus scope 下必须按文档过滤。** gold span 是原文字符偏移，只在它自己那份
   合同里有意义。不过滤的话，别的合同里凑巧落在同一区间的 chunk 会被算成命中，
   抬高"正确停止"、同时把假阳性藏起来。
2. **无答案样本不能混进 2×2。** 它们没有 gold span，"不在里面"恒真，混进来会把
   假阳性率稀释成一个好看但没意义的数。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gold_round_check import _gold_in_chunks, _rates, _spans_by_id, analyse  # noqa: E402

from lex_rag.trace_sink import TraceSink  # noqa: E402


def _chunk(cid, start, end, doc="D"):
    return {"chunk_id": cid, "doc_id": doc, "start": start, "end": end}


# ── 重叠判定 ──────────────────────────────────────────────────

def test_span_overlap_matches_the_retrieval_eval_predicate():
    chunks = [_chunk("a", 0, 100), _chunk("b", 100, 200)]
    assert _gold_in_chunks(chunks, [(150, 160)], "D") is True     # 落在 b 内
    assert _gold_in_chunks(chunks, [(90, 110)], "D") is True      # 跨界
    assert _gold_in_chunks(chunks, [(300, 310)], "D") is False    # 都不沾


def test_other_documents_do_not_count_as_a_hit():
    """corpus 模式的关键：偏移相同但文档不同，不算命中。"""
    chunks = [_chunk("x", 100, 200, doc="OTHER")]
    assert _gold_in_chunks(chunks, [(150, 160)], "D") is False
    assert _gold_in_chunks(chunks, [(150, 160)], "OTHER") is True


def test_doc_filter_is_skipped_when_no_doc_id_is_known():
    """doc_id 为 None 时不过滤——总比因为拿不到文档名就一律判不命中好。"""
    chunks = [_chunk("x", 100, 200, doc="ANY")]
    assert _gold_in_chunks(chunks, [(150, 160)], None) is True


def test_chunks_without_offsets_are_ignored_not_crashed():
    assert _gold_in_chunks([{"chunk_id": "a", "doc_id": "D"}], [(1, 2)], "D") is False


# ── 2×2 与比率 ────────────────────────────────────────────────

def test_rates_use_separate_denominators():
    """假阴性只在"gold 已在里面"的轮次上有定义，假阳性反之。

    共用一个分母会把两个不同的问题混成一个数。
    """
    from collections import Counter
    cell = Counter({(True, True): 6, (True, False): 4,      # gold 在：10 轮
                    (False, True): 1, (False, False): 9})   # gold 不在：10 轮
    r = _rates(cell)
    assert r["fn_rate"] == 0.4 and r["fp_rate"] == 0.1
    assert r["accuracy"] == 0.75


def test_empty_cell_yields_none_instead_of_dividing_by_zero():
    from collections import Counter
    r = _rates(Counter())
    assert r["accuracy"] is None and r["fn_rate"] is None and r["fp_rate"] is None


# ── 端到端：写一份 trace 再读回来统计 ─────────────────────────

def _write_trace(path, *, qid, has_answer, doc, rounds, terminated):
    """rounds = [(chunk_ranges, sufficient), ...]"""
    from lex_rag.chunking import ChunkWindow

    sink = TraceSink(path, run_id="R")
    with sink.query("q", doc_id=None,
                    meta={"id": qid, "has_answer": has_answer, "gold_doc_id": doc}) as qt:
        for ranges, sufficient in rounds:
            with qt.round() as rt:
                rt.chunks([ChunkWindow(chunk_id=f"{d}#{s}", doc_id=d, text="x" * (e - s),
                                       start=s, end=e) for d, s, e in ranges])
                rt.verdict({"sufficient": sufficient, "missing": "", "out_of_scope": False,
                            "missing_kind": "none", "confidence": 1.0})
        qt.terminate(terminated)
    sink.close()


def _qa(tmp_path, rows):
    p = tmp_path / "qa.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


def test_end_to_end_2x2_counts(tmp_path):
    qa = _qa(tmp_path, [
        {"id": "hit", "spans": [{"start": 150, "end": 160}]},
        {"id": "miss", "spans": [{"start": 900, "end": 910}]},
        {"id": "none", "spans": []},                      # 无答案样本
    ])
    from gold_round_check import _spans_by_id
    spans = _spans_by_id(qa)

    # gold 在里面，judge 说不够 → 白烧
    _write_trace(tmp_path / "a.jsonl", qid="hit", has_answer=True, doc="D",
                 rounds=[([("D", 100, 200)], False), ([("D", 100, 200)], True)],
                 terminated="sufficient")
    res = analyse(tmp_path / "a.jsonl", spans)
    assert res["cell"][(True, False)] == 1 and res["cell"][(True, True)] == 1
    assert res["wasted_rounds"] == 1

    # gold 不在里面，judge 说够了 → 提前停止
    _write_trace(tmp_path / "b.jsonl", qid="miss", has_answer=True, doc="D",
                 rounds=[([("D", 100, 200)], True)], terminated="sufficient")
    res = analyse(tmp_path / "b.jsonl", spans)
    assert res["cell"][(False, True)] == 1

    # 无答案样本不进 2×2
    _write_trace(tmp_path / "c.jsonl", qid="none", has_answer=False, doc="D",
                 rounds=[([("D", 100, 200)], True)], terminated="refused")
    res = analyse(tmp_path / "c.jsonl", spans)
    assert sum(res["cell"].values()) == 0 and res["no_span"] == 1
    assert res["term"]["refused"] == 1


def test_corpus_scope_uses_gold_doc_id_from_meta(tmp_path):
    """trace 的 doc_id 是 None（corpus），必须回填 meta 里的 gold_doc_id。

    不回填就会把别的合同里同偏移的 chunk 当成命中——正好毁掉这张表要测的东西。
    """
    qa = _qa(tmp_path, [{"id": "q", "spans": [{"start": 150, "end": 160}]}])
    from gold_round_check import _spans_by_id
    spans = _spans_by_id(qa)

    _write_trace(tmp_path / "t.jsonl", qid="q", has_answer=True, doc="D",
                 rounds=[([("OTHER", 100, 200)], True)], terminated="sufficient")
    res = analyse(tmp_path / "t.jsonl", spans)
    # 命中的是别的合同 → 应判为"gold 不在"，即提前停止
    assert res["cell"][(False, True)] == 1 and res["cell"][(True, True)] == 0


def test_rounds_without_a_verdict_are_skipped(tmp_path):
    """被防重复拦下的轮没有 verdict，不能当成"judge 说不够"计进去。"""
    from lex_rag.chunking import ChunkWindow
    from gold_round_check import _spans_by_id

    qa = _qa(tmp_path, [{"id": "q", "spans": [{"start": 150, "end": 160}]}])
    sink = TraceSink(tmp_path / "t.jsonl", run_id="R")
    with sink.query("q", meta={"id": "q", "gold_doc_id": "D"}) as qt:
        with qt.round() as rt:
            rt.chunks([ChunkWindow(chunk_id="D#1", doc_id="D", text="x", start=100, end=200)])
            rt.verdict({"sufficient": False, "missing": "", "out_of_scope": False,
                        "missing_kind": "none", "confidence": 1.0})
        with qt.round() as rt:
            rt.rejected_repeat()
        qt.terminate("repeat_blocked")
    sink.close()

    res = analyse(tmp_path / "t.jsonl", _spans_by_id(qa))
    assert sum(res["cell"].values()) == 1 and res["skipped"] == 1


def test_gold_beyond_top_k_is_not_counted_as_waste(tmp_path):
    """trace 落盘整个累积池，但 judge 只看前 k 个。

    gold 排在第 11 位时判定器根本没见过它，说"不够"是对的；按整池判会把这种
    情况算成白烧，冤枉判定器。见 agent.py 的 `judge(question, pool[:k])`。
    """
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({
        "id": "q1", "doc_id": "d", "question": "?",
        "spans": [{"start": 500, "end": 510}],
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    # 前 2 条不含 gold，第 3 条含——k=2 时判定器看不到它
    chunks = [{"chunk_id": "c1", "doc_id": "d", "start": 0, "end": 100},
              {"chunk_id": "c2", "doc_id": "d", "start": 100, "end": 200},
              {"chunk_id": "c3", "doc_id": "d", "start": 450, "end": 600}]
    trace = tmp_path / "t.jsonl"
    trace.write_text(json.dumps({
        "trace_id": "t1", "question": "?", "doc_id": "d",
        "meta": {"id": "q1", "k": 2}, "terminated_by": "max_rounds", "n_rounds": 1,
        "rounds": [{"index": 0, "chunks": chunks,
                    "verdict": {"sufficient": False, "missing_kind": "exact_term"}}],
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    res = analyse(trace, _spans_by_id(qa))

    assert res["cell"][(True, False)] == 0      # 不该算白烧
    assert res["cell"][(False, False)] == 1     # 是"正确继续"
    assert res["wasted_rounds"] == 0
