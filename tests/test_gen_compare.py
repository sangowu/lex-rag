"""`--compare` 的配对分析与单变量检查。

这两个东西是针对同一个教训加的：这个仓库白跑过一次 200x2 的 generate_k 实验，
按比率比得出"无差异"，其实 200 条里只有 50 条有答案，理论效应 5 个点 = 2.5 条，
而 n=50 的标准误是 0.057——预期效应只有噪声的一半。按 id 配对之后，同一批数据
立刻看得出净 +6 的目标群体效应。

所以配对是 `--compare` 的默认动作，不是可选的额外分析；单变量检查则是防止
下一次把两个变量捆在一起测。
"""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "eval_generation", Path(__file__).resolve().parents[1] / "scripts" / "eval_generation.py"
)
eg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eg)


# --- McNemar 精确检验 ------------------------------------------------------

def test_no_flips_is_not_evidence_of_anything():
    assert eg._exact_binom_two_sided(0, 0) == 1.0


def test_symmetric_flips_give_p_one():
    """5 赚 5 亏就是没效应，不管翻面总数多大。"""
    assert eg._exact_binom_two_sided(5, 5) == 1.0


def test_one_sided_flips_are_significant():
    """10 次翻面全在一个方向，p 应该很小。"""
    assert eg._exact_binom_two_sided(10, 0) < 0.01


def test_p_is_symmetric_in_its_arguments():
    assert eg._exact_binom_two_sided(7, 2) == eg._exact_binom_two_sided(2, 7)


def test_small_net_on_few_flips_is_not_significant():
    """净 +1（5 比 4）绝不该被当成效应——真实数据里出现过这一格。"""
    assert eg._exact_binom_two_sided(4, 5) > 0.5


# --- 配对统计 --------------------------------------------------------------

def _row(rid, has_answer, **flags):
    base = {"id": rid, "has_answer": has_answer, "semantic_hit": False,
            "false_negative": False, "false_positive": False, "latency_ms": 1000.0}
    base.update(flags)
    return base


def test_paired_counts_only_discordant_pairs(capsys):
    """两臂一致的样本不该出现在任何一格里。"""
    a = [_row("x", True, semantic_hit=True), _row("y", True), _row("z", True)]
    b = [_row("x", True, semantic_hit=True), _row("y", True, semantic_hit=True), _row("z", True)]

    eg._print_paired_diff(a, b)
    out = capsys.readouterr().out

    line = next(l for l in out.splitlines() if "semantic_hit" in l)
    # 只有 y 翻了面，方向对 B 有利
    assert "        1        0    +1" in line


def test_paired_ignores_ids_missing_from_either_arm(capsys):
    a = [_row("x", True, semantic_hit=True), _row("only_a", True)]
    b = [_row("x", True, semantic_hit=True), _row("only_b", True)]

    eg._print_paired_diff(a, b)

    assert "共同样本 1 条" in capsys.readouterr().out


def test_false_positive_is_scored_on_the_no_answer_subset(capsys):
    """FP 只在无答案样本上有意义，别把有答案的样本混进分母。"""
    a = [_row("n1", False, false_positive=True), _row("h1", True)]
    b = [_row("n1", False), _row("h1", True)]

    eg._print_paired_diff(a, b)
    out = capsys.readouterr().out

    assert "有答案 1 / 无答案 1" in out
    line = next(l for l in out.splitlines() if "false_positive" in l)
    assert "        1        0    +1" in line  # B 少编造了一次，算 B 好


def test_latency_is_compared_per_item_not_as_an_average(capsys):
    """均值会被少数长尾拖走；逐条之差的中位才是"通常快多少"。"""
    a = [_row(f"i{i}", True, latency_ms=1000.0) for i in range(5)]
    b = [_row(f"i{i}", True, latency_ms=400.0) for i in range(5)]
    b[0]["latency_ms"] = 99999.0   # 一条极端慢的，不该翻转结论

    eg._print_paired_diff(a, b)
    out = capsys.readouterr().out

    assert "中位 -600ms" in out
    assert "4/5 条 B 更快" in out


# --- 单变量检查 ------------------------------------------------------------

def test_identical_configs_are_called_out_as_noise(capsys):
    eg._print_single_variable_check({"thinking": True}, {"thinking": True})

    assert "配置完全相同" in capsys.readouterr().out


def test_one_differing_field_is_a_clean_experiment(capsys):
    eg._print_single_variable_check({"thinking": True}, {"thinking": False})
    out = capsys.readouterr().out

    assert "thinking" in out
    assert "不是单变量实验" not in out


def test_two_differing_fields_are_flagged(capsys):
    eg._print_single_variable_check(
        {"thinking": True, "generate_k": 8}, {"thinking": False, "generate_k": 20}
    )

    assert "不是单变量实验" in capsys.readouterr().out


@pytest.mark.parametrize("field", ["ts", "timestamp", "run_id", "git_commit"])
def test_bookkeeping_fields_do_not_count_as_variables(field, capsys):
    """时间戳和 commit 每次都不同，算进去会让每次对比都报"不是单变量"。"""
    eg._print_single_variable_check({field: "a"}, {field: "b"})

    assert "配置完全相同" in capsys.readouterr().out
