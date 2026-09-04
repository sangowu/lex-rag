"""来源指纹：让"文档变了"从发不出声变成发得出声。

起因是可用性注入实验的负结果：唯一打得穿的 payload 是一句**约束合同双方的条款**
（"第 8 条已被修正案删除"），模型据此拒答是**正确**的法律阅读。所以那不是 prompt
漏洞——**能往语料里写字的攻击者就能改变答案，而且没有任何 prompt 修得了它**。
防线只能在 ingest，而这个模块不阻止改动，只保证改动会被看见。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lex_rag import store as store_mod
from lex_rag.ingest_guard import (
    SourceRecord,
    Verdict,
    classify,
    content_digest,
    summarise,
)
from lex_rag.store import VectorStore


def rec(doc="d", text="hello world", source=""):
    return SourceRecord(doc, content_digest(text), len(text), source)


# --- 指纹 -------------------------------------------------------------------

def test_the_same_text_gives_the_same_digest():
    assert content_digest("a b") == content_digest("a b")


def test_reflowing_whitespace_is_not_a_change():
    """EDGAR 原文里成串的空格、重跑 OCR 的换行差异都不该报警。"""
    assert content_digest("a     b\n\nc") == content_digest("a b c")


def test_changing_a_word_is_a_change():
    assert content_digest("shall not apply") != content_digest("shall apply")


def test_case_is_significant():
    """合同里定义术语靠首字母大写标记，`Term` 和 `term` 不是一回事。

    这与 `text_match.normalize` 的取舍**有意不同**：那把尺子在比"答案像不像 gold"，
    这把在比"这份文件是不是同一份"。
    """
    assert content_digest("the Term") != content_digest("the term")


def test_punctuation_is_significant():
    assert content_digest("Party A, Inc") != content_digest("Party A Inc")


# --- 判定 -------------------------------------------------------------------

def test_a_document_never_seen_before_is_new():
    assert classify(None, rec()) is Verdict.NEW


def test_the_same_content_is_unchanged():
    assert classify(rec(text="x y"), rec(text="x y")) is Verdict.UNCHANGED


def test_different_content_under_the_same_id_is_the_case_that_matters():
    assert classify(rec(text="x y"), rec(text="x z")) is Verdict.CHANGED


def test_a_length_preserving_edit_is_still_caught():
    """"paid" -> "owed" 这类等长改写，只比字符数是抓不到的。"""
    a, b = rec(text="the fee shall be paid"), rec(text="the fee shall be owed")
    assert a.n_chars == b.n_chars
    assert classify(a, b) is Verdict.CHANGED


# --- 汇报 -------------------------------------------------------------------

def test_unchanged_documents_do_not_get_listed_one_by_one():
    """一次全量是 25 份文档。把它们全打出来会把唯一重要的那一行淹掉，
    而这个模块存在的全部意义就是让那一行被看见。"""
    results = [(rec(doc=f"d{i}"), Verdict.UNCHANGED, rec(doc=f"d{i}")) for i in range(25)]
    out = summarise(results)
    assert "25 未变" in out
    assert "d7" not in out


def test_a_changed_document_is_named_with_both_digests():
    prev = rec(doc="contract", text="original text here")
    cur = rec(doc="contract", text="tampered text here!")
    out = summarise([(cur, Verdict.CHANGED, prev)])
    assert "contract" in out
    assert prev.sha256[:12] in out and cur.sha256[:12] in out


def test_the_char_delta_is_signed():
    prev = rec(doc="c", text="x" * 100)
    cur = rec(doc="c", text="x" * 90 + "y")
    assert "-9" in summarise([(cur, Verdict.CHANGED, prev)])


def test_no_changes_means_no_warning_noise():
    out = summarise([(rec(), Verdict.NEW, None)])
    assert "⚠️" not in out


# --- 与 store 的接线 ---------------------------------------------------------

@pytest.fixture
def fake_store(monkeypatch):
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    monkeypatch.setattr(store_mod.psycopg, "connect", lambda dsn, **kw: conn)
    monkeypatch.setattr(store_mod, "register_vector", lambda _c: None)
    return VectorStore("dsn://x", table="chunks_ocr", init_schema=False), cur


def _sql(cur):
    return " ".join(str(c.args[0]) for c in cur.execute.call_args_list)


def test_truncate_must_not_wipe_the_fingerprints(fake_store):
    """**这条是整个机制成立的前提。**

    全量 ingest 的第一步就是 TRUNCATE。指纹要是跟着清掉，每次重灌之后所有文档
    都变成"新增"，`CHANGED` 永远不会出现——机制等于没有，而且完全无声。
    """
    store, cur = fake_store
    store.truncate()
    sql = _sql(cur)
    assert "chunks_ocr" in sql
    assert "doc_source" not in sql


def test_fingerprints_are_scoped_to_the_table(fake_store):
    """同一份合同可以同时躺在 chunks 和 chunks_ocr 里，而那两份文本本来就不同。
    不按表分隔的话，两条 ingest 路径会互相报"内容已变更"。"""
    store, cur = fake_store
    cur.fetchone.return_value = None
    store.get_doc_source("some-contract")
    params = cur.execute.call_args[0][1]
    assert "chunks_ocr" in params


def test_saving_a_fingerprint_overwrites_the_previous_one(fake_store):
    """同一个 (table, doc_id) 只保留最新一版——历史版本不是这个机制的职责，
    它只回答"和上次比变了没有"。"""
    store, cur = fake_store
    store.save_doc_source(rec(doc="c"))
    sql = _sql(cur)
    assert "ON CONFLICT" in sql and "DO UPDATE" in sql
