"""semantic_hit 的判据：逐字包含 或 余弦。

加包含判据是因为一个可复现的系统性偏差：generator 的 prompt 明确要求
"quote the exact sentence(s) that contain the answer"，而 CUAD 的 gold 是从那句话里
抽出来的短 span。40 词的整句 vs 5 词的 span，余弦只有 0.5——**整句里逐字含着
gold，旧尺子却判它没命中**。实测 50 条有答案样本里有 6 条落在这一格。
旧尺子惩罚的正是 prompt 要求的行为。

所以这里钉三件事：
  ① 包含判据认得出真实的漏判形态（长引用套短 span）；
  ② 它不会滥认（太短的 gold、标点不同的 gold）；
  ③ `semantic_hit_cosine` 永远单独留着——历史结果全是余弦测的，
     丢了这一格就切断了与 v4/v5 的可比性。
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "eval_generation", Path(__file__).resolve().parents[1] / "scripts" / "eval_generation.py"
)
eg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eg)


# --- 包含判据 -------------------------------------------------------------

def test_gold_quoted_verbatim_inside_a_long_sentence_is_a_hit():
    """真实漏判形态：模型照抄整句，gold 是句子里的一小段。"""
    answer = ('The agreement states: "This Agreement is made and entered into as of '
              'the 6th day of April, 1999 by and between the parties." [3]')

    assert eg._contains_gold(answer, "6th day of April, 1999")


def test_match_is_case_insensitive():
    assert eg._contains_gold("The SUPPLY CONTRACT governs.", "supply contract")


def test_line_breaks_in_the_answer_do_not_block_the_match():
    """答案里的换行是排版，gold 里是空格，两边都归一化成单空格。"""
    assert eg._contains_gold("party of the\n   first part", "party of the first part")


def test_curly_quotes_and_dashes_are_normalized():
    assert eg._contains_gold("titled \u201cSUPPLY CONTRACT\u201d here", '"SUPPLY CONTRACT"')
    assert eg._contains_gold("2011\u20132012 term", "2011-2012")


def test_nbsp_is_normalized():
    assert eg._contains_gold("Centrack\u00a0International", "Centrack International")


def test_very_short_gold_is_not_matched_by_containment():
    """"Inc" / "the" 这种碎片能在几乎任何答案里撞上，那是巧合不是命中。"""
    assert not eg._contains_gold("Acme Inc and others", "Inc")


def test_punctuation_is_not_stripped():
    """删标点会把不同的实体判成同一个；逐字引用场景下标点必须原样在场。"""
    assert not eg._contains_gold("Party A Inc", "Party A, Inc")


def test_absent_gold_is_not_a_hit():
    assert not eg._contains_gold("The agreement is silent on this.", "governing law")


# --- 两条判据的组合 -------------------------------------------------------

def _fake_embedder(monkeypatch, vecs: dict[str, list[float]], truncate: int | None = None):
    class _Fake:
        def __init__(self, *a, **kw): pass
        def embed_texts(self, texts):
            out = [vecs.get(t, [0.0, 1.0]) for t in texts]
            return out[:truncate] if truncate is not None else out

    import lex_rag.embeddings as emb
    monkeypatch.setattr(emb, "EmbeddingClient", _Fake)


def _run(monkeypatch, answer, golds, vecs=None, truncate=None, threshold=0.7):
    _fake_embedder(monkeypatch, vecs or {}, truncate)
    rows = [{}]
    sim_data = [{"answer": answer, "golds": golds, "row_idx": 0}]
    cfg = SimpleNamespace(embedding=None)   # 只被传给 EmbeddingClient，这里已替身
    counts = eg.compute_semantic_hits(sim_data, rows, cfg, threshold)
    return rows[0], counts


def test_containment_rescues_a_low_cosine_answer(monkeypatch):
    """余弦判否、包含判是 —— 这一格正是这次要修的东西。"""
    answer = 'The contract is titled "SUPPLY CONTRACT" [4].'
    row, counts = _run(monkeypatch, answer, ["SUPPLY CONTRACT"],
                       vecs={answer: [1.0, 0.0], "SUPPLY CONTRACT": [0.0, 1.0]})

    assert row["semantic_hit"] is True
    assert row["semantic_hit_cosine"] is False
    assert counts == {"hit": 1, "cosine": 0, "contain_only": 1}


def test_paraphrase_still_counts_via_cosine(monkeypatch):
    """模型用自己的话答对时 gold 不会字面出现，余弦判据必须还在。"""
    row, counts = _run(monkeypatch, "The buyer is Acme.", ["Acme Corporation"],
                       vecs={"The buyer is Acme.": [1.0, 0.0],
                             "Acme Corporation": [1.0, 0.0]})

    assert (row["semantic_hit"], row["semantic_hit_cosine"]) == (True, True)
    assert counts["contain_only"] == 0     # 余弦已经认了，不算包含独立救回


def test_cosine_count_is_unchanged_by_the_new_criterion(monkeypatch):
    """新判据只能往上加，绝不能改动 cosine 那一格——历史可比性押在它身上。"""
    answer = 'Titled "SUPPLY CONTRACT".'
    _, counts = _run(monkeypatch, answer, ["SUPPLY CONTRACT"],
                     vecs={answer: [1.0, 0.0], "SUPPLY CONTRACT": [0.0, 1.0]})

    assert counts["cosine"] == 0


def test_containment_survives_a_partial_embedding_failure(monkeypatch):
    """embedding 只回了一部分向量时，包含判据仍然独立成立。"""
    answer = 'Titled "SUPPLY CONTRACT".'
    row, counts = _run(monkeypatch, answer, ["SUPPLY CONTRACT"], truncate=1)

    assert row["semantic_sim"] == 0.0      # 算不出余弦
    assert row["semantic_hit"] is True     # 但照样命中
    assert counts["contain_only"] == 1


def test_empty_answer_is_a_miss_on_both_criteria(monkeypatch):
    row, counts = _run(monkeypatch, "", ["SUPPLY CONTRACT"])

    assert (row["semantic_hit"], row["semantic_hit_cosine"]) == (False, False)
    assert counts == {"hit": 0, "cosine": 0, "contain_only": 0}


def test_no_texts_to_embed_returns_zeroed_counts(monkeypatch):
    _fake_embedder(monkeypatch, {})

    assert eg.compute_semantic_hits([], [], SimpleNamespace(embedding=None), 0.7) == {
        "hit": 0, "cosine": 0, "contain_only": 0}


# --- 与历史结果文件的兼容 -------------------------------------------------

def test_old_result_files_fall_back_to_their_own_semantic_hit_rate():
    """包含判据之前的文件没有 *_cosine 字段，但它的 semantic_hit_rate 就是纯 cosine。

    不回落的话 --compare 会拿 0.0 去做差，凭空造出一个 -0.8 的"退化"。
    """
    old = {"semantic_hit_rate": 0.800}

    assert eg._metric(old, "semantic_hit_rate_cosine") == 0.800


def test_new_result_files_use_their_own_cosine_field():
    new = {"semantic_hit_rate": 0.880, "semantic_hit_rate_cosine": 0.800}

    assert eg._metric(new, "semantic_hit_rate_cosine") == 0.800


def _row(rid, **flags):
    base = {"id": rid, "has_answer": True, "semantic_hit": False,
            "false_negative": False, "false_positive": False, "latency_ms": 1.0}
    base.update(flags)
    return base


def test_paired_diff_warns_when_the_two_arms_used_different_rulers(capsys):
    """新旧尺子混比时 semantic_hit 那一行比的是两把尺子，不是两个配置。"""
    old = [_row("x", semantic_hit=True)]
    new = [_row("x", semantic_hit=True, semantic_hit_cosine=False)]

    eg._print_paired_diff(old, new)

    assert "判据不同" in capsys.readouterr().out


def test_paired_diff_scores_cosine_against_an_old_arm_without_the_field(capsys):
    """旧臂的 semantic_hit 就是 cosine，所以 cosine 这一行跨新旧永远可比。"""
    old = [_row("x", semantic_hit=True), _row("y", semantic_hit=False)]
    new = [_row("x", semantic_hit=True, semantic_hit_cosine=True),
           _row("y", semantic_hit=True, semantic_hit_cosine=False)]

    eg._print_paired_diff(old, new)
    out = capsys.readouterr().out

    # y 在新臂只是被包含判据救回，cosine 没变 → cosine 行应该零翻面
    line = next(l for l in out.splitlines() if l.strip().startswith("└ 仅 cosine"))
    assert "        0        0    +0" in line


def test_paired_diff_is_quiet_when_both_arms_share_a_ruler(capsys):
    a = [_row("x", semantic_hit=True, semantic_hit_cosine=True)]
    b = [_row("x", semantic_hit=True, semantic_hit_cosine=True)]

    eg._print_paired_diff(a, b)

    assert "判据不同" not in capsys.readouterr().out
