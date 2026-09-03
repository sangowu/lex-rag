"""gold span 的逐字包含判据——**全仓库唯一的一把尺子**。

原先这套归一化只活在 `scripts/eval_generation.py` 里。`lex_rag/gate.py` 需要
同一个判据时我顺手又写了一个"简单版"（只 `.lower()`），结果门禁第一次真跑就把
3 条正确答案判成没命中——CUAD 的 gold 里有成串的空格
（`If Distributor                            complies with...`），模型引用时把它们
规整掉了，逐字比较自然对不上。

**两把尺子必然分叉，而且分叉时没有任何报错。** 所以判据搬到这里，
`eval_generation.py` 与 `gate.py` 都从这里 import，谁都不再自带一份。
"""
from __future__ import annotations

import re

__all__ = ["normalize", "contains_gold", "quote_overlap",
           "MIN_GOLD_CHARS", "QUOTE_OVERLAP_THRESHOLD"]

_WS_RE = re.compile(r"\s+")

# str.maketrans 接受 {码位: 替换串}，这里刻意用码位而不是字面字符——
# NBSP 写成字面量就是源码里一个隐形字符，改坏了看不出来。
_TYPOGRAPHY = {
    0x201C: '"', 0x201D: '"',   # 弯双引号
    0x2018: "'", 0x2019: "'",   # 弯单引号
    0x2013: "-", 0x2014: "-",   # en / em dash
    0x00A0: " ",                # NBSP
}

# gold 短于这个长度就不用包含判据。"Inc" / "LLC" / "the" 这种碎片能在几乎任何
# 答案里撞上，那不是命中，是巧合。CUAD 里真正有意义的最短 gold 是 4 字符级别
# （如 "1999"），所以门槛设在这里。
MIN_GOLD_CHARS = 4


def normalize(text: str) -> str:
    """包含判据用的归一化：只抹排版差异，不删标点、不动词形。

    删标点会让 "Party A, Inc." 和 "Party AInc" 判成同一个；不删则 gold 里的
    标点必须原样出现。逐字引用场景下后者才是对的——prompt 要求的就是原文照抄。

    ⚠️ 空格折叠不是可选项：CUAD 的 gold 直接来自 SEC 文件的等宽排版，句中常有
    十几个连续空格。不折叠的话，模型每一次正确的引用都会被判成没命中。
    """
    return _WS_RE.sub(" ", text.translate(_TYPOGRAPHY)).strip().lower()


def contains_gold(answer: str, gold: str) -> bool:
    """gold span 是否**逐字**出现在答案里。

    这条判据的存在理由：prompt 明确要求 "quote the exact sentence(s) that contain
    the answer"，而 CUAD 的 gold 是从那句话里抽出来的短 span。于是一句 40 词的
    原文引用 vs 一个 5 词的 span，余弦只有 0.5 左右——**整句里逐字含着 gold，
    却被判成没命中**。旧尺子惩罚的正是 prompt 要求的行为。
    """
    g = normalize(gold)
    if len(g) < MIN_GOLD_CHARS:
        return False
    return g in normalize(answer)


# ---------------------------------------------------------------------------
# 引用重合度：给门禁用的、比逐字包含宽一档的判据
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.;])\s+")

# 门禁的默认判定线。0.70 是量出来的，不是拍的：把 8 条有答案案例的答案与全部
# 8 条 gold 交叉配对（run 20260903T112*），
#     配对正确 n=8：最低 0.806，中位 1.000
#     配错     n=56：最高 0.500，中位 0.048
# 0.500 与 0.806 之间是空的，线画在中间。
# ⚠️ 配错那侧的 0.500 全部来自只有 2 个 token 的 gold（"DISTRIBUTOR AGREEMENT"
# 撞上另一条答案里的 "agreement"）——短 gold 的分辨率只有 0/0.5/1，这条线在它们
# 身上没有余量。短 gold 主要靠 `contains_gold` 兜住，两条判据是 OR 的关系。
QUOTE_OVERLAP_THRESHOLD = 0.70


def _tokens(text: str) -> list[str]:
    """分词，词内的撇号保留、词首词尾的引号剥掉。

    `company's` 必须保持一个 token；但 `'electric` 不能——模型把合同里的
    `"Electric City"` 写成 `'Electric City'` 时，粘在词首的引号会让这个 token
    对不上，连续段正好从那里断开，重合度直接腰斩（实测 1.000 -> 0.469，
    而两边是同一句逐字引用）。
    """
    return [t for t in (w.strip("'") for w in _WORD_RE.findall(normalize(text))) if t]


def _longest_common_run(a: list[str], b: list[str]) -> int:
    """a、b 的最长**连续**公共 token 段长度。

    刻意不用集合重合度：集合会把"把 gold 的词打散重排"也算成命中，而这里要判的
    是"有没有引用那一段话"。连续段才对应"引用"。
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def quote_overlap(answer: str, gold: str) -> float:
    """答案对 gold 的最大引用重合度，0~1。

    **按 gold 的句子分别算，取最大值。** CUAD 的 gold 常常是两三句连在一起，
    而模型只引其中一句——那是正确回答，整段比会把它判成没命中（实测门禁里
    "Cap On Liability" 就是这样，答案逐字引了 gold 的第二句）。

    标点在这里被剥掉，与 `contains_gold` 相反。两条判据服务不同问题：
      - `contains_gold` 问"是不是逐字照抄"，标点必须留着（`Party A, Inc` 与
        `Party AInc` 不是一回事），用在 `eval_generation` 的命中率上；
      - `quote_overlap` 问"引的是不是那一段"，引号写成单引号还是双引号、
        末尾有没有句号都无所谓，用在门禁上。
    实测这五种漏判全是后一类：`"term"` vs `the term`、末尾多一个句号、
    `'electric city'` vs `"electric city"`、引用被 `...` 截断。
    """
    a = _tokens(answer)
    if not a:
        return 0.0

    # 按句拆开，但**只在拆得出多于一句时**才丢掉短片段：短片段在长 gold 里没有
    # 判别力，可整条 gold 本身就很短时（"DISTRIBUTOR AGREEMENT" 只有 2 个 token）
    # 丢掉它就等于把这条案例判成 0。第一版漏了这一条，实测把 4 条 Document Name
    # 全判成 0.000 而它们其实是逐字命中。
    sents = [_tokens(x) for x in _SENT_SPLIT_RE.split(gold)]
    candidates = [g for g in sents if len(g) >= 3]
    if not candidates:
        whole = _tokens(gold)
        candidates = [whole] if len(whole) >= 2 else []

    best = 0.0
    for g in candidates:
        best = max(best, _longest_common_run(a, g) / len(g))
    return best
