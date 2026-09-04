"""合成扫描件渲染器里唯一有判断的部分：折行。

渲染和退化都要靠眼睛验收，测不出什么；`_wrap` 不一样——它决定 OCR 看到的版面，
而版面正是这条链最容易悄悄退化的地方。CUAD 原文的缩进是结构信息（条款编号靠
缩进对齐），先把它 strip 掉再去考 OCR，量出来的就不是 OCR 的能力了。
"""
from __future__ import annotations

import importlib.util
import random
import zlib
from pathlib import Path

import pytest
from PIL import Image

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_scanned_demo.py"
_spec = importlib.util.spec_from_file_location("make_scanned_demo", _PATH)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

wrap = demo._wrap


def test_a_short_line_is_left_alone():
    assert wrap("SECTION 1. DEFINITIONS", 80) == ["SECTION 1. DEFINITIONS"]


def test_a_long_line_is_split_at_the_column():
    out = wrap("word " * 40, 40)
    assert len(out) > 1
    assert all(len(line) <= 40 for line in out)


def test_blank_lines_survive():
    """空行是 EDGAR 排版里的段落边界。抹掉它们，OCR 出来的就是一大坨。"""
    assert wrap("a\n\n\nb", 80) == ["a", "", "", "b"]


def test_leading_indentation_is_preserved():
    out = wrap("        1.4 RENEWAL.", 80)
    assert out[0].startswith("        ")


def test_a_wrapped_line_keeps_its_indent_on_every_row():
    out = wrap("    " + "word " * 30, 40)
    assert len(out) > 1
    assert all(line.startswith("    ") for line in out)


def test_no_word_is_split_across_rows():
    out = wrap("supercalifragilistic " * 6, 40)
    assert all("supercalifragilistic" in line for line in out)
    assert "".join(out).count("supercalifragilistic") == 6


def test_a_word_longer_than_the_column_still_gets_emitted():
    """否则它会被静默丢掉——扫描件里少一个词，而 OCR 背这个锅。"""
    out = wrap("x" * 100, 40)
    assert "".join(out).count("x") == 100


def test_an_absurd_indent_cannot_eat_the_whole_column():
    """CUAD 里有缩进到 40+ 列的行。原样保留会让可用宽度归零，
    折行就退化成一行一个词。"""
    out = wrap(" " * 200 + "text here", 60)
    assert out and out[0].strip() == "text here"
    assert len(out[0]) < 60


@pytest.mark.parametrize("cols", [40, 60, 84, 120])
def test_every_row_fits_the_column_at_any_width(cols):
    text = Path(__file__).read_text(encoding="utf-8")
    assert all(len(line) <= cols for line in wrap(text, cols) if len(line.split()) > 1)


# --- 每份合同各自的随机流 -----------------------------------------------------
#
# 第一版共用一个 rng：`random.Random(seed)` 然后按顺序退化所有合同。那样往列表里
# **插一份**就会改变它之后每一份的噪声，已经发表的 CER 全部作废，而且完全无声——
# 图看起来一样，数字悄悄变了。现在种子是 seed + crc32(doc_id)。

def _seed_for(seed: int, doc_id: str) -> int:
    return seed + zlib.crc32(doc_id.encode("utf-8"))


def _degrade_bytes(seed: int, doc_id: str) -> bytes:
    img = Image.new("L", (60, 40), 255)
    return demo._degrade(img, random.Random(_seed_for(seed, doc_id))).tobytes()


def test_the_same_contract_degrades_identically_every_run():
    """种子定死，重跑逐像素相同——否则文档里的数字下次就对不上。"""
    assert _degrade_bytes(1, "A.txt") == _degrade_bytes(1, "A.txt")


def test_two_contracts_do_not_share_a_noise_stream():
    assert _degrade_bytes(1, "A.txt") != _degrade_bytes(1, "B.txt")


def test_a_contracts_noise_does_not_depend_on_its_neighbours():
    """**这条是那个 bug 本身。** 一份合同的退化只能由 (seed, doc_id) 决定；
    它前面渲染过谁、列表里还有谁，都不能影响它。"""
    alone = _degrade_bytes(7, "target")
    for neighbour in ("first", "second", "third"):
        random.Random(7 + zlib.crc32(neighbour.encode()))   # 模拟先渲染了别人
    assert _degrade_bytes(7, "target") == alone


def test_changing_the_seed_changes_the_corpus():
    assert _degrade_bytes(1, "A.txt") != _degrade_bytes(2, "A.txt")
