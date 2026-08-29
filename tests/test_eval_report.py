"""检索评测的报表必须跟着 `evaluation.k_values` 走，不能写死一组 k。

这是这个仓库第三次同形态的漂移：`serve.py` 硬编码 `top_k=10`、
`reranker.enabled` 停在 false、以及这里——`run_eval` 把 `hit@1/3/5/10` 写死在
结果行里，于是 `k_values` 加了 20 之后 `evaluate()` 照算，算完却落不进文件。

三次的共同点是**完全无声**：功能照常，报表照出，只有拿两轮结果逐格对比才看得见。
所以报表这一端也要 pin。
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_script", Path(__file__).resolve().parents[1] / "scripts" / "eval.py"
)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _row(**over):
    r = {"hit@1": 0.54, "hit@3": 0.70, "hit@5": 0.81, "hit@10": 0.86,
         "mrr@5": 0.64, "precision@5": 0.23, "recall@5": 0.75, "latency_ms": 1617.0}
    r.update(over)
    return r


# --- 从结果文件里读出实际有哪些 k -----------------------------------------

def test_ks_are_discovered_from_the_file_not_hardcoded():
    assert ev._ks(_row(**{"hit@20": 0.90}), "hit") == [1, 3, 5, 10, 20]


def test_ks_are_sorted_numerically_not_lexically():
    """按字符串排会把 hit@20 排在 hit@3 前面。"""
    assert ev._ks(_row(**{"hit@20": 0.9}), "hit") == [1, 3, 5, 10, 20]


def test_hit_and_mrr_are_read_independently():
    """k_values 展开后 mrr 也是每个 k 一格，别拿 hit 的 k 去索引 mrr。"""
    r = _row(**{"hit@20": 0.9, "mrr@1": 0.5, "mrr@20": 0.66})

    assert ev._ks(r, "hit") == [1, 3, 5, 10, 20]
    assert ev._ks(r, "mrr") == [1, 5, 20]


# --- 打印 ------------------------------------------------------------------

def test_new_k_shows_up_in_the_printed_line(capsys):
    ev.print_result(_row(**{"hit@20": 0.904}))

    assert "hit@20=0.904" in capsys.readouterr().out


def test_old_result_files_still_print(capsys):
    """加 @20 之前的文件没有那一格，报表不该因此 KeyError。"""
    ev.print_result(_row())
    out = capsys.readouterr().out

    assert "hit@10=0.860" in out and "hit@20" not in out


# --- diff ------------------------------------------------------------------

def test_diff_compares_only_the_k_both_files_have(capsys):
    """k_values 不同的两轮放一起比是常态——多出来的那一格不能凭空当成 0。"""
    old = _row()
    new = _row(**{"hit@20": 0.904})

    ev.print_diff(old, new, "old", "new")
    out = capsys.readouterr().out

    assert "hit@10" in out
    assert "hit@20" not in out       # old 没有这一格，比不了


def test_diff_includes_a_shared_new_k(capsys):
    a = _row(**{"hit@20": 0.900})
    b = _row(**{"hit@20": 0.904})

    ev.print_diff(a, b, "a", "b")

    assert "hit@20" in capsys.readouterr().out


def test_diff_still_reports_precision_and_recall(capsys):
    ev.print_diff(_row(), _row(), "a", "b")
    out = capsys.readouterr().out

    assert "precision@5" in out and "recall@5" in out
