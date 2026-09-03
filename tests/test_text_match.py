"""gold 匹配的两把尺子。

`contains_gold` 问"是不是逐字照抄"（`eval_generation` 的命中率用它）；
`quote_overlap` 问"引的是不是那一段"（发布门禁用它，宽一档）。

下面每条 `quote_overlap` 的测试都对应门禁第一次真跑时的一个实测漏判——
五条正确引用被逐字包含判成没命中，逐条查下来全是引用风格差异，不是答错。
"""
from __future__ import annotations

from lex_rag.text_match import (
    MIN_GOLD_CHARS,
    QUOTE_OVERLAP_THRESHOLD,
    contains_gold,
    normalize,
    quote_overlap,
)

TH = QUOTE_OVERLAP_THRESHOLD


# --- normalize / contains_gold ---------------------------------------------

def test_runs_of_whitespace_are_folded():
    """CUAD 的 gold 来自 SEC 的等宽排版，句中常有十几个连续空格。"""
    assert normalize("a     b\n\n c") == "a b c"


def test_curly_quotes_and_dashes_are_flattened():
    assert normalize("“x” – y") == '"x" - y'


def test_punctuation_is_kept():
    """删标点会把 "Party A, Inc." 和 "Party AInc" 判成同一个。"""
    assert normalize("Party A, Inc.") == "party a, inc."


def test_containment_ignores_layout_but_not_wording():
    gold = "No  assignment  of this  Agreement\n     shall be made"
    assert contains_gold("... no assignment of this agreement shall be made ...", gold)
    assert not contains_gold("no transfer of this agreement shall be made", gold)


def test_a_fragment_shorter_than_the_floor_never_counts():
    """"Inc" / "the" 能在几乎任何答案里撞上，那是巧合不是命中。"""
    assert len("Inc") < MIN_GOLD_CHARS
    assert not contains_gold("Acme Inc is the counterparty", "Inc")


# --- quote_overlap：五种实测漏判 --------------------------------------------

def test_a_missing_pair_of_quotes_around_a_defined_term():
    gold = '"Term" means the earlier of: (a) the end of the two year period'
    answer = 'The Term means the earlier of: (a) the end of the two year period'
    assert not contains_gold(answer, gold)          # 逐字包含判它没命中
    assert quote_overlap(answer, gold) >= TH        # 引用重合度判它命中


def test_a_trailing_full_stop():
    gold = "It will be governed by the law of the People's Republic of China."
    answer = '"It will be governed by the law of the People\'s Republic of China" [1].'
    assert not contains_gold(answer, gold)
    assert quote_overlap(answer, gold) >= TH


def test_single_quotes_where_the_contract_used_double_ones():
    """引号粘在词首会让那个 token 对不上，连续段正好从那里断开。

    实测这一条把重合度从 1.000 打到 0.469——两边是同一句逐字引用。
    """
    gold = 'use the name "Electric City of Illinois" or a similar variation thereof'
    answer = "the right to use the name 'Electric City of Illinois' or a similar variation thereof"
    assert quote_overlap(answer, gold) >= TH


def test_quoting_only_one_sentence_of_a_multi_sentence_gold():
    """CUAD 的 gold 常把两三句连在一起，模型只引其中一句是正确回答。"""
    gold = ("i-on will not be liable for any lost profits or other consequential damages. "
            "i-on's liability shall be limited to one month's fees.")
    answer = '"i-on\'s liability shall be limited to one month\'s fees." [1]'
    assert quote_overlap(answer, gold) >= TH


def test_a_quote_truncated_with_an_ellipsis():
    gold = ("No assignment of this Agreement or any right accruing hereunder shall be "
            "made by the Distributor in whole or in part, without the prior written "
            "consent of the Company, which consent shall not be unreasonably withheld")
    answer = ('"No assignment of this Agreement or any right accruing hereunder shall be '
              'made by the Distributor in whole or in part, without the prior written '
              'consent of the Company..." [1]')
    assert quote_overlap(answer, gold) >= TH


def test_a_short_gold_is_not_discarded_by_the_sentence_filter():
    """第一版丢掉了少于 3 个 token 的片段，于是 "DISTRIBUTOR AGREEMENT" 这类
    整条 gold 直接判成 0.000，而它们其实是逐字命中。"""
    assert quote_overlap('"LIMEENERGYCO-EX-10-DISTRIBUTOR AGREEMENT" [1].',
                         "DISTRIBUTOR AGREEMENT") == 1.0


# --- quote_overlap：不能放水 -------------------------------------------------

def test_a_different_clause_scores_below_the_line():
    gold = "This Agreement is governed by the laws of the State of California"
    assert quote_overlap("The contract is governed by Texas law.", gold) < TH


def test_scrambled_words_do_not_count_as_a_quote():
    """用集合重合度就会把这条算成满分——所以判据是最长**连续**公共段。"""
    gold = "the Company shall indemnify the Distributor against all claims"
    answer = "claims all against Distributor the indemnify shall Company the"
    assert quote_overlap(answer, gold) < TH


def test_an_empty_answer_scores_zero():
    assert quote_overlap("", "some gold text here") == 0.0


def test_the_threshold_sits_in_the_measured_gap():
    """交叉配对实测：配对正确最低 0.806，配错最高 0.500。线必须落在中间。"""
    assert 0.500 < QUOTE_OVERLAP_THRESHOLD < 0.806
