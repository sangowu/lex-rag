"""Ingest 层的来源指纹：让"文档变了"这件事**发不出声**变成发得出声。

## 为什么需要

`docs/experiments.md` 的可用性注入实验里，唯一打得穿的 payload 是一句
**约束合同双方的条款**——"第 8 条已被修正案 No.2 删除"。模型据此拒答是**正确**的
法律阅读，所以那不是 prompt 漏洞：

> **能往语料里写字的攻击者就能改变答案，而且没有任何 prompt 修得了它**——修了就等于
> 让模型无视真实的修正案条款。

结论当时写的是"防线在 ingest（来源可信 / 签名 / 与已知good版本对拍）"，但一行都没实现。
这个模块是那一行的最小版本：**不阻止改动，只保证改动会被看见。**

同一个洞的另一面来自 OCR：`docs/demo_ocr_rag.md` 实测同一份合同重跑 OCR，文本会
变（换个噪声实例 CER 就差 0.0017）。没有指纹的话，"语料被人改了"和"OCR 这轮跑歪了"
在数据库里长得一模一样。

## 判据

指纹取**空白归一化之后**的 SHA-256。理由是把"重排"和"改词"分开：EDGAR 原文里成串的
空格、重跑 OCR 的换行差异都不该报警，而少一个 `not`、改一个日期必须报警。
⚠️ 代价是**纯排版攻击不会被这里抓到**——如果哪天真要防那个，得另加一条判据，
别把这一条的阈值调松。

`classify` 是纯函数，没有 DB、没有 IO，`tests/test_ingest_guard.py` 覆盖。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum

__all__ = ["Verdict", "SourceRecord", "content_digest", "classify",
           "summarise", "CHANGED_EXIT_NOTE"]

_WS = re.compile(r"\s+")

CHANGED_EXIT_NOTE = (
    "有文档的内容与上次 ingest 时不同。这可能是正常更新，也可能是语料被改写"
    "（见 docs/experiments.md 的可用性注入一节）。确认无误后重跑即可；"
    "要在 CI/流水线里把它当作阻断，用 --fail-on-changed-source。"
)


class Verdict(str, Enum):
    NEW = "new"                # 没见过这个 doc_id
    UNCHANGED = "unchanged"    # 指纹一致
    CHANGED = "changed"        # 同一个 doc_id，内容变了 —— 唯一需要人看的那一类


@dataclass(frozen=True)
class SourceRecord:
    """一份文档在某张表里的来源指纹。"""
    doc_id: str
    sha256: str
    n_chars: int
    source: str = ""


def content_digest(text: str) -> str:
    """空白归一化之后的 SHA-256。

    **不做小写化、不删标点**：`Party A, Inc` 和 `Party AInc` 必须是不同的指纹，
    大小写在合同里也可能是实质差异（定义术语靠首字母大写标记）。
    与 `lex_rag.text_match.normalize` 的取舍不同是**有意的**——那把尺子在比"人写的
    答案像不像 gold"，这把在比"这份文件是不是同一份"。
    """
    return hashlib.sha256(_WS.sub(" ", text).strip().encode("utf-8")).hexdigest()


def classify(previous: SourceRecord | None, current: SourceRecord) -> Verdict:
    if previous is None:
        return Verdict.NEW
    return Verdict.UNCHANGED if previous.sha256 == current.sha256 else Verdict.CHANGED


def summarise(results: list[tuple[SourceRecord, Verdict, SourceRecord | None]]) -> str:
    """给 ingest 脚本收尾用的一段人读文本。

    **变更逐条列出，新增/未变只给计数**：一次全量 ingest 有 25 份文档，把它们全打出来
    会把唯一重要的那一行淹掉——而这个模块存在的全部意义就是让那一行被看见。
    """
    counts = {v: 0 for v in Verdict}
    changed: list[tuple[SourceRecord, SourceRecord | None]] = []
    for cur, verdict, prev in results:
        counts[verdict] += 1
        if verdict is Verdict.CHANGED:
            changed.append((cur, prev))

    lines = [f"来源指纹：{counts[Verdict.NEW]} 新增 / "
             f"{counts[Verdict.UNCHANGED]} 未变 / {counts[Verdict.CHANGED]} 变更"]
    for cur, prev in changed:
        delta = cur.n_chars - (prev.n_chars if prev else 0)
        lines.append(f"  ⚠️ 内容已变更：{cur.doc_id}")
        lines.append(f"       {(prev.sha256[:12] if prev else '?')} -> {cur.sha256[:12]}"
                     f"   {cur.n_chars} 字符（{delta:+d}）")
    if changed:
        lines.append(f"  {CHANGED_EXIT_NOTE}")
    return "\n".join(lines)
