"""引用的展示片段：应该给出被引的那句话，而不是 chunk 的开头。

起因是 README 的 UI 截图。原实现是 `chunk.text[:120]`，屏幕上显示成
"ment, may be executed for each state or entity representing each state. 1…"
——从半个单词开始，且与答案里引用的句子毫无关系。**引用看上去像是引错了，
而实际上 chunk 是对的、只是预览取错了地方。**
"""
from __future__ import annotations

from lex_rag.generator import _excerpt_for, _locate_quote

# 一段仿 CUAD 原文的 chunk：句中有成串空格（SEC 等宽排版），且开头是半个单词
CHUNK = (
    "ment, may be executed for each state or entity representing each state. "
    "1.4 RENEWAL. If Distributor  complies with all of the terms of this "
    "Agreement, the        Agreement shall be renewable on an annual basis for "
    "one (1) year terms. 1.5 NOTICES. All notices hereunder shall be in writing "
    "and delivered to the addresses set out below, or to such other address as "
    "either party may designate from time to time."
)
ANSWER = ('"If Distributor complies with all of the terms of this Agreement, the Agreement '
          'shall be renewable on an annual basis for one (1) year terms." [3].')


def test_the_excerpt_contains_the_quoted_sentence():
    out = _excerpt_for(CHUNK, ANSWER)
    assert "renewable on an annual basis" in out


def test_the_excerpt_is_not_just_the_start_of_the_chunk():
    """这条是回归本身：旧实现返回的正是 chunk 开头那 120 字。"""
    assert not _excerpt_for(CHUNK, ANSWER).startswith("ment, may be executed")


def test_runs_of_whitespace_do_not_defeat_the_lookup():
    """模型引用时把 `Distributor  complies` 的双空格规整成单空格，
    直接 str.find 必然落空——所以按词拼允许任意空白的正则。"""
    assert _locate_quote(CHUNK, "If Distributor complies with all of the terms") >= 0


def test_a_newline_inside_the_original_is_also_tolerated():
    chunk = "shall be\n     renewable on an annual basis for one year"
    assert _locate_quote(chunk, "shall be renewable on an annual basis") >= 0


def test_the_excerpt_does_not_start_or_end_mid_word():
    out = _excerpt_for(CHUNK, ANSWER).strip("…").strip()
    assert CHUNK.count(out.split()[0]) >= 1
    # 首尾的词必须是 chunk 里完整出现过的词
    assert out.split()[0] in CHUNK.split()
    assert out.split()[-1].rstrip(".,;") in " ".join(CHUNK.split())


def test_an_unlocatable_quote_falls_back_to_the_head():
    """定位不到就退回开头——但仍然要对齐词边界，不能再从半个单词开始。"""
    out = _excerpt_for(CHUNK, '"a sentence that is nowhere in this chunk at all" [1].')
    assert out.startswith("ment, may be executed")
    assert not out.rstrip("…").endswith("repre")


def test_an_answer_with_no_quotes_falls_back_too():
    out = _excerpt_for(CHUNK, "Illinois [1].")
    assert out.startswith("ment, may be executed")


def test_a_very_short_quote_is_not_used_for_lookup():
    """两三个词能在 chunk 里到处撞上，拿它定位只会把窗口开在随机位置。"""
    assert _locate_quote(CHUNK, "the terms") == -1


def test_the_excerpt_stays_short_enough_to_display():
    assert len(_excerpt_for(CHUNK, ANSWER)) <= 260


def test_a_chunk_shorter_than_the_window_is_returned_whole():
    short = "Governed by the laws of the State of Illinois."
    out = _excerpt_for(short, '"Governed by the laws of the State of Illinois." [1]')
    assert out.strip("…") == short
    assert "…" not in out          # 没有截断就不该有省略号
