"""
本地 trace 落盘 —— 规格 `docs/agentic_loop_upgrade.md` 第 2.5 节。

**为什么不扩展 `tracing.py`。** 那个模块是 Langfuse 封装，没配 key 时完全 no-op
（这是它的正确设计：在线可观测性不该拖垮主流程）。但 2.5 要的是实验语料：

* 2.6 要跑三组配置 × CUAD 全量产出语料。如果语料存不存在取决于一个环境变量，
  这份语料就不可靠——跑完才发现什么都没留下，代价是整整三轮全量。
* 下游 `tracelens` 需要**能本地 diff 的文件**，而不是远程服务里的记录。
* 每步 input/output 全文动辄上万字符，全量往云端推既慢又无谓。

所以两者职责分开：`tracing.py` 管在线链路的可观测性，本模块管落盘的实验语料。
本模块**不 no-op**——给了路径就一定写，写不出来会响一次，但不抛异常打断主流程。

格式是 JSONL，**一次查询一行，写完立刻 flush**。这条是被教训过的：此前有两轮
评测在收尾阶段崩掉，200 条结果全部丢失，原因就是等到最后才落盘。逐行 flush 意味着
进程被 Ctrl-C 或 OOM 掉时，已完成的查询一条都不少。

用法::

    sink = TraceSink(Path("data/runs/traces/run1.jsonl"), config={"reranker": True})
    with sink.query("What is the governing law?", doc_id="X") as qt:
        for i in range(3):
            with qt.round() as rt:
                rt.selector(reason="精确术语，走 BM25", prompt=p, raw=r)
                rt.strategy(st)
                rt.step("retrieval", input=st.query_text or q, output=chunk_ids)
                rt.chunks(chunks)
                rt.verdict(v)
                if v.sufficient:
                    qt.terminate("sufficient")
                    break
        qt.answer(result.answer)
    sink.close()

路径以 ``.gz`` 结尾时自动 gzip——全量语料压缩比在 5~8 倍。
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DIR = Path("data/runs/traces")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _jsonable(v: Any) -> Any:
    """尽量把任意对象变成可序列化的东西，**绝不因为一个字段毁掉整条记录**。"""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    for attr in ("to_dict", "_asdict"):
        fn = getattr(v, attr, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                return _jsonable(fn())
    # 只在 __dict__ 真有内容时才用它：空字典会把对象序列化成 `{}`，
    # 看起来像"记下来了"，其实什么都没留下——repr 至少还能认出这是个什么东西。
    with contextlib.suppress(Exception):
        d = vars(v)
        if d:
            return {k: _jsonable(x) for k, x in d.items()}
    return repr(v)


@dataclass
class StepRecord:
    """一步的完整输入输出。规格要求全文落盘，所以这里**不做截断**。"""

    name: str
    input: Any = None
    output: Any = None
    meta: dict = field(default_factory=dict)
    started_at: str = ""
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input": _jsonable(self.input),
            "output": _jsonable(self.output),
            "meta": _jsonable(self.meta),
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


@dataclass
class RoundRecord:
    """一轮检索决策。字段与规格 2.5 的表格一一对应。"""

    index: int
    strategy: dict | None = None
    strategy_key: str | None = None
    selector_reason: str | None = None      # 唯一能还原"系统当时怎么想"的字段
    selector_prompt: str | None = None
    selector_raw: str | None = None
    chunks: list[dict] = field(default_factory=list)   # chunk_id + score + 位置
    verdict: dict | None = None
    cumulative_chunks: int = 0
    cumulative_chars: int = 0               # 精确值
    cumulative_tokens: int | None = None    # 仅当各步报了 usage 才有，否则 None
    steps: list[StepRecord] = field(default_factory=list)
    started_at: str = ""
    duration_ms: float = 0.0
    rejected_repeat: bool = False           # 选择器选了已试过的策略，被执行层拦下

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "strategy": self.strategy,
            "strategy_key": self.strategy_key,
            "selector_reason": self.selector_reason,
            "selector_prompt": self.selector_prompt,
            "selector_raw": self.selector_raw,
            "chunks": self.chunks,
            "verdict": self.verdict,
            "cumulative_chunks": self.cumulative_chunks,
            "cumulative_chars": self.cumulative_chars,
            "cumulative_tokens": self.cumulative_tokens,
            "rejected_repeat": self.rejected_repeat,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 2),
            "steps": [s.to_dict() for s in self.steps],
        }


class RoundWriter:
    """一轮的写入句柄。所有 setter 都不抛——埋点坏掉不该毁掉被埋点的东西。"""

    def __init__(self, rec: RoundRecord, owner: "QueryWriter") -> None:
        self._rec = rec
        self._owner = owner
        self._t0 = time.perf_counter()

    def strategy(self, strategy: Any) -> None:
        with contextlib.suppress(Exception):
            self._rec.strategy = _jsonable(strategy)
            key = getattr(strategy, "key", None)
            self._rec.strategy_key = key() if callable(key) else None

    def selector(self, reason: str | None = None, prompt: str | None = None,
                 raw: str | None = None) -> None:
        with contextlib.suppress(Exception):
            self._rec.selector_reason = reason
            self._rec.selector_prompt = prompt
            self._rec.selector_raw = raw

    def rejected_repeat(self, value: bool = True) -> None:
        self._rec.rejected_repeat = value

    def chunks(self, chunks: list[Any]) -> None:
        """记录本轮返回的 chunk 及分数，并更新累积量。

        存 `score_kind` 是必须的：cosine / bm25 / rrf / rerank 四种分数跨阶段
        不可比，只有数字没有来源的话，trace 里的分数无法解释。
        """
        with contextlib.suppress(Exception):
            self._rec.chunks = [{
                "chunk_id": getattr(c, "chunk_id", None),
                "doc_id": getattr(c, "doc_id", None),
                "score": getattr(c, "score", None),
                "score_kind": getattr(c, "score_kind", None),
                "start": getattr(c, "start", None),
                "end": getattr(c, "end", None),
                "chars": len(getattr(c, "text", "") or ""),
            } for c in chunks]
            self._owner._accumulate(chunks)
            self._rec.cumulative_chunks = len(self._owner._seen_ids)
            self._rec.cumulative_chars = self._owner._cum_chars
            self._rec.cumulative_tokens = self._owner._cum_tokens

    def verdict(self, verdict: Any) -> None:
        with contextlib.suppress(Exception):
            self._rec.verdict = _jsonable(verdict)

    def step(self, name: str, *, input: Any = None, output: Any = None,
             meta: dict | None = None, duration_ms: float = 0.0,
             error: str | None = None, tokens: int | None = None) -> None:
        """记一步。input/output 全文进去，不截断。"""
        with contextlib.suppress(Exception):
            self._rec.steps.append(StepRecord(
                name=name, input=input, output=output, meta=meta or {},
                started_at=_now_iso(), duration_ms=duration_ms, error=error,
            ))
            if tokens is not None:
                self._owner._add_tokens(tokens)
                self._rec.cumulative_tokens = self._owner._cum_tokens

    @contextlib.contextmanager
    def timed_step(self, name: str, *, input: Any = None,
                   meta: dict | None = None) -> Iterator[dict]:
        """把一步的耗时和异常自动记下来。

        yield 出一个 dict，调用方往 ``["output"]`` / ``["tokens"]`` 里塞结果；
        即使中途抛异常，这一步也会带着 error 落盘——**失败的那一步恰恰是最该
        留下痕迹的一步**。
        """
        box: dict = {"output": None, "tokens": None}
        t0 = time.perf_counter()
        try:
            yield box
        except Exception as e:
            self.step(name, input=input, output=box.get("output"), meta=meta,
                      duration_ms=(time.perf_counter() - t0) * 1000,
                      error=f"{type(e).__name__}: {e}", tokens=box.get("tokens"))
            raise
        self.step(name, input=input, output=box.get("output"), meta=meta,
                  duration_ms=(time.perf_counter() - t0) * 1000,
                  tokens=box.get("tokens"))

    def _close(self) -> None:
        self._rec.duration_ms = (time.perf_counter() - self._t0) * 1000


class QueryWriter:
    """一次查询的写入句柄。退出 with 块时整条记录落盘。"""

    def __init__(self, sink: "TraceSink", question: str, doc_id: str | None,
                 meta: dict | None) -> None:
        self._sink = sink
        self._t0 = time.perf_counter()
        self._seen_ids: set[str] = set()
        self._cum_chars = 0
        self._cum_tokens: int | None = None
        self._rounds: list[RoundRecord] = []
        self._payload: dict = {
            "trace_id": f"{sink.run_id}-{sink._counter:05d}",
            "question": question,
            "doc_id": doc_id,
            "meta": _jsonable(meta or {}),
            "started_at": _now_iso(),
            "terminated_by": None,
            "answer": None,
            "refused": None,
        }

    # ── 累积量 ──────────────────────────────────────────────
    def _accumulate(self, chunks: list[Any]) -> None:
        for c in chunks:
            cid = getattr(c, "chunk_id", None)
            if cid is None or cid in self._seen_ids:
                continue
            self._seen_ids.add(cid)
            self._cum_chars += len(getattr(c, "text", "") or "")

    def _add_tokens(self, n: int) -> None:
        self._cum_tokens = (self._cum_tokens or 0) + n

    # ── 写入 ────────────────────────────────────────────────
    @contextlib.contextmanager
    def round(self, index: int | None = None) -> Iterator[RoundWriter]:
        rec = RoundRecord(index=len(self._rounds) if index is None else index,
                          started_at=_now_iso())
        self._rounds.append(rec)
        w = RoundWriter(rec, self)
        try:
            yield w
        finally:
            w._close()

    def terminate(self, reason: str) -> None:
        """`sufficient` / `refused` / `max_rounds`——失败分类的第一层。"""
        self._payload["terminated_by"] = reason

    def answer(self, text: str | None, refused: bool | None = None) -> None:
        self._payload["answer"] = text
        self._payload["refused"] = refused

    def set(self, key: str, value: Any) -> None:
        """挂额外字段（如 gold span 命中与否），供离线分析用。"""
        self._payload[key] = _jsonable(value)

    def _finish(self) -> dict:
        self._payload.update({
            "duration_ms": round((time.perf_counter() - self._t0) * 1000, 2),
            "n_rounds": len(self._rounds),
            "cumulative_chunks": len(self._seen_ids),
            "cumulative_chars": self._cum_chars,
            "cumulative_tokens": self._cum_tokens,
            "rounds": [r.to_dict() for r in self._rounds],
        })
        # terminated_by 没被显式设置说明循环是自然跑完的
        if self._payload["terminated_by"] is None:
            self._payload["terminated_by"] = "max_rounds"
        return self._payload


class TraceSink:
    """把 trace 追加写进 JSONL。一次查询一行，写完立刻 flush。"""

    def __init__(self, path: Path | str | None = None, *,
                 run_id: str | None = None, config: dict | None = None) -> None:
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if path is None:
            path = DEFAULT_DIR / f"{self.run_id}.jsonl"
        self.path = Path(path)
        self.config = config or {}
        self._counter = 0
        self._warned = False
        self._fh: Any = None
        # 跑批时多个 worker 同时写：一次查询一行的前提是这一行不被别人插进来。
        self._lock = threading.Lock()
        self._open()

    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.suffix == ".gz":
                import gzip
                self._fh = gzip.open(self.path, "at", encoding="utf-8")
            else:
                self._fh = self.path.open("a", encoding="utf-8")
        except Exception as e:
            self._warn(f"打不开 {self.path}：{type(e).__name__}: {e}")
            self._fh = None
            return
        # 头一行记配置：语料被单独拿走时，还能知道它是什么条件下跑出来的
        self._write_raw({"_meta": True, "run_id": self.run_id,
                         "created_at": _now_iso(), "config": _jsonable(self.config)})

    def _warn(self, msg: str) -> None:
        """只响一次。**不静默**——静默失败会让人跑完全量才发现语料是空的。"""
        if self._warned:
            return
        self._warned = True
        print(f"[trace_sink] 写入失败，本次运行不会留下语料：{msg}",
              file=sys.stderr, flush=True)

    def _write_raw(self, obj: dict) -> None:
        if self._fh is None:
            return
        # 序列化放在锁外：它是纯 CPU 的活，占着锁会把并发写压成串行。
        try:
            line = json.dumps(obj, ensure_ascii=False) + "\n"
        except Exception as e:
            self._warn(f"序列化失败：{type(e).__name__}: {e}")
            return
        try:
            with self._lock:
                self._fh.write(line)
                self._fh.flush()
                # flush 只到 libc 缓冲；崩溃时靠 fsync 才真正落盘。全量语料是几
                # 小时的算力，值这点开销。
                with contextlib.suppress(Exception, AttributeError):
                    os.fsync(self._fh.fileno())
        except Exception as e:
            self._warn(f"{type(e).__name__}: {e}")

    @contextlib.contextmanager
    def query(self, question: str, doc_id: str | None = None,
              meta: dict | None = None) -> Iterator[QueryWriter]:
        """一次查询的 with 块。**异常也会落盘**——崩掉的那条最值得留。"""
        with self._lock:
            self._counter += 1
        qw = QueryWriter(self, question, doc_id, meta)
        try:
            yield qw
        except Exception as e:
            qw.set("error", f"{type(e).__name__}: {e}")
            qw.terminate("error")
            self._write_raw(qw._finish())
            raise
        self._write_raw(qw._finish())

    def close(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(Exception):
                self._fh.close()
            self._fh = None

    def __enter__(self) -> "TraceSink":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _open_text(path: Path) -> Any:
    if path.suffix == ".gz":
        import gzip
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_traces(path: Path | str) -> list[dict]:
    """把 JSONL 读回来（跳过头部的 _meta 行）。给离线分析和测试用。"""
    p = Path(path)
    out = []
    with _open_text(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("_meta"):
                out.append(obj)
    return out


def read_meta(path: Path | str) -> dict:
    """只读头部的配置行。"""
    with _open_text(Path(path)) as f:
        first = f.readline()
    obj = json.loads(first or "{}")
    return obj if obj.get("_meta") else {}
