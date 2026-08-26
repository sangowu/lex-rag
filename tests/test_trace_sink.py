"""TraceSink 单元测试。

这个模块的价值全在"崩了以后还剩下什么"上，所以测试的重点不是happy path，
而是三件事：逐条落盘（崩溃时已完成的不丢）、异常那一步照样留痕、埋点自己
出错不能毁掉被埋点的业务逻辑。
"""

import json

import pytest

from lex_rag.chunking import ChunkWindow
from lex_rag.strategy import RetrievalStrategy
from lex_rag.sufficiency import Verdict
from lex_rag.trace_sink import TraceSink, read_meta, read_traces


def _chunks(n=2, score=0.5):
    return [ChunkWindow(chunk_id=f"d#{i}", doc_id="d", text="x" * 100,
                        start=i * 100, end=i * 100 + 100,
                        score=score - i * 0.1, score_kind="rerank")
            for i in range(n)]


def _sink(tmp_path, name="t.jsonl", **kw):
    return TraceSink(tmp_path / name, run_id="RUN", **kw)


# ── 基本结构 ──────────────────────────────────────────────────

def test_one_line_per_query_plus_a_meta_header(tmp_path):
    s = _sink(tmp_path, config={"reranker": True})
    for q in ("q1", "q2"):
        with s.query(q):
            pass
    s.close()

    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3                       # 1 行 meta + 2 行查询
    assert json.loads(lines[0])["_meta"] is True
    assert read_meta(tmp_path / "t.jsonl")["config"] == {"reranker": True}
    assert [t["question"] for t in read_traces(tmp_path / "t.jsonl")] == ["q1", "q2"]


def test_round_records_every_field_the_spec_asks_for(tmp_path):
    s = _sink(tmp_path)
    st = RetrievalStrategy(mode="bm25", top_k=5)
    v = Verdict(sufficient=False, missing="缺金额", missing_kind="exact_term",
                confidence=0.9)
    with s.query("q", doc_id="D") as qt:
        with qt.round() as rt:
            rt.strategy(st)
            rt.selector(reason="精确术语走 BM25", prompt="P", raw="R")
            rt.chunks(_chunks(2))
            rt.verdict(v)
            rt.step("retrieval", input="q", output=["d#0", "d#1"], tokens=12)
        qt.terminate("sufficient")
        qt.answer("A", refused=False)
    s.close()

    r = read_traces(tmp_path / "t.jsonl")[0]["rounds"][0]
    assert r["strategy"]["mode"] == "bm25"
    assert r["strategy_key"] == st.key()
    assert r["selector_reason"] == "精确术语走 BM25"
    assert [c["chunk_id"] for c in r["chunks"]] == ["d#0", "d#1"]
    assert r["chunks"][0]["score_kind"] == "rerank"       # 分数来源必须跟着分数走
    assert r["verdict"]["missing_kind"] == "exact_term"
    assert r["verdict"]["strategy_hint"] == "bm25"
    assert r["cumulative_chunks"] == 2 and r["cumulative_chars"] == 200
    assert r["cumulative_tokens"] == 12
    assert r["steps"][0]["name"] == "retrieval"


def test_cumulative_counts_dedupe_across_rounds(tmp_path):
    """多轮累加必须去重，否则"上下文污染"这个指标会被重复命中的 chunk 灌水。"""
    s = _sink(tmp_path)
    with s.query("q") as qt:
        with qt.round() as rt:
            rt.chunks(_chunks(2))
        with qt.round() as rt:
            rt.chunks(_chunks(3))          # 前两个与上一轮重复
    s.close()

    t = read_traces(tmp_path / "t.jsonl")[0]
    assert t["rounds"][1]["cumulative_chunks"] == 3
    assert t["cumulative_chunks"] == 3 and t["cumulative_chars"] == 300


def test_terminated_by_defaults_to_max_rounds(tmp_path):
    """没人显式终止 = 循环自然跑完，这是失败分类的第一层，不能留空。"""
    s = _sink(tmp_path)
    with s.query("q") as qt:
        with qt.round():
            pass
    s.close()
    assert read_traces(tmp_path / "t.jsonl")[0]["terminated_by"] == "max_rounds"


def test_rejected_repeat_is_recorded(tmp_path):
    """防重复是在执行层拦截的，拦了几次必须能查出来（策略震荡的证据）。"""
    s = _sink(tmp_path)
    with s.query("q") as qt:
        with qt.round() as rt:
            rt.rejected_repeat()
    s.close()
    assert read_traces(tmp_path / "t.jsonl")[0]["rounds"][0]["rejected_repeat"] is True


# ── 崩溃时还剩下什么 ──────────────────────────────────────────

def test_completed_queries_survive_a_crash_midway(tmp_path):
    """逐行 flush 的意义：崩在第三条，前两条一条不少。

    此前有两轮评测在收尾阶段崩掉、200 条结果全丢，就是因为等到最后才落盘。
    """
    s = _sink(tmp_path)
    with s.query("q1"):
        pass
    with s.query("q2"):
        pass
    with pytest.raises(RuntimeError):
        with s.query("q3"):
            raise RuntimeError("boom")

    done = read_traces(tmp_path / "t.jsonl")
    assert [t["question"] for t in done] == ["q1", "q2", "q3"]
    assert done[2]["terminated_by"] == "error" and "boom" in done[2]["error"]


def test_failing_step_is_recorded_with_its_error(tmp_path):
    """失败的那一步恰恰最该留痕——归因就是要定位到哪一步炸的。"""
    s = _sink(tmp_path)
    with pytest.raises(ValueError):
        with s.query("q") as qt:
            with qt.round() as rt:
                with rt.timed_step("judge", input="ctx"):
                    raise ValueError("judge died")
    s.close()

    step = read_traces(tmp_path / "t.jsonl")[0]["rounds"][0]["steps"][0]
    assert step["name"] == "judge" and step["input"] == "ctx"
    assert "judge died" in step["error"]


def test_timed_step_captures_output_and_tokens(tmp_path):
    s = _sink(tmp_path)
    with s.query("q") as qt:
        with qt.round() as rt:
            with rt.timed_step("gen", input="p") as box:
                box["output"] = "answer"
                box["tokens"] = 7
    s.close()

    step = read_traces(tmp_path / "t.jsonl")[0]["rounds"][0]["steps"][0]
    assert step["output"] == "answer" and step["duration_ms"] >= 0
    assert read_traces(tmp_path / "t.jsonl")[0]["cumulative_tokens"] == 7


# ── 埋点自己出错时的行为 ──────────────────────────────────────

def test_unserialisable_objects_do_not_kill_the_record(tmp_path):
    """一个字段序列化不了，不该毁掉整条 trace。"""
    class Weird:
        def __repr__(self):
            return "<weird>"

    s = _sink(tmp_path)
    with s.query("q") as qt:
        with qt.round() as rt:
            rt.step("odd", input=Weird(), output={1, 2, 3})
    s.close()

    step = read_traces(tmp_path / "t.jsonl")[0]["rounds"][0]["steps"][0]
    assert step["input"] == "<weird>"


def test_chunks_without_score_are_still_recorded(tmp_path):
    """ingest 期的 ChunkWindow 没有 score，埋点不能因此抛。"""
    s = _sink(tmp_path)
    with s.query("q") as qt:
        with qt.round() as rt:
            rt.chunks([ChunkWindow(chunk_id="a", doc_id="d", text="t", start=0, end=1)])
    s.close()

    c = read_traces(tmp_path / "t.jsonl")[0]["rounds"][0]["chunks"][0]
    assert c["chunk_id"] == "a" and c["score"] is None


def test_write_failure_warns_once_and_does_not_raise(tmp_path, capsys):
    """写不出来要响，但不能打断主流程——**也绝不能静默**。

    静默失败最坏：跑完三组全量才发现语料是空的。
    """
    s = _sink(tmp_path)
    s._fh.close()                       # 模拟文件句柄失效
    for q in ("q1", "q2"):
        with s.query(q):
            pass                        # 不抛

    err = capsys.readouterr().err
    assert "trace_sink" in err and err.count("写入失败") == 1


def test_gzip_path_round_trips(tmp_path):
    s = _sink(tmp_path, name="t.jsonl.gz")
    with s.query("q1") as qt:
        qt.terminate("refused")
    s.close()

    traces = read_traces(tmp_path / "t.jsonl.gz")
    assert traces[0]["terminated_by"] == "refused"
