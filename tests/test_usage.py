"""token 用量的提取与汇总。

加这个是因为一次真实的空白：关掉 thinking 之后延迟从 7.9s 掉到 0.9s，但被问到
"省了多少钱"时答不出来——usage 一直取到了，却只喂给 Langfuse，没配 key 时 tracing
是 no-op，本地评测完全看不到 token。所以这里钉两件事：能取到，且取不到时说得出来。
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from lex_rag.llm import Usage, _usage_of

_spec = importlib.util.spec_from_file_location(
    "eval_generation", Path(__file__).resolve().parents[1] / "scripts" / "eval_generation.py"
)
eg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eg)


def _resp(**usage_fields):
    details = usage_fields.pop("details", None)
    usage = SimpleNamespace(**usage_fields)
    if details is not None:
        usage.completion_tokens_details = SimpleNamespace(**details)
    return SimpleNamespace(usage=usage)


# --- 从响应里提取 ---------------------------------------------------------

def test_extracts_prompt_and_completion_tokens():
    u = _usage_of(_resp(prompt_tokens=1200, completion_tokens=48))

    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (1200, 48, 1248)
    assert u.reported is True


def test_extracts_reasoning_tokens_when_present():
    u = _usage_of(_resp(prompt_tokens=1200, completion_tokens=376,
                        details={"reasoning_tokens": 348}))

    assert u.reasoning_tokens == 348


def test_reasoning_is_part_of_completion_not_extra():
    """thinking 的 token 已经含在 completion_tokens 里，total 不该再加一遍。"""
    u = _usage_of(_resp(prompt_tokens=100, completion_tokens=376,
                        details={"reasoning_tokens": 348}))

    assert u.total_tokens == 476


def test_missing_usage_is_reported_as_not_reported():
    """服务端不给 usage 时，0 必须能和"真的没花 token"区分开。"""
    u = _usage_of(SimpleNamespace())

    assert u.total_tokens == 0
    assert u.reported is False


def test_missing_details_does_not_raise():
    u = _usage_of(_resp(prompt_tokens=10, completion_tokens=20))

    assert u.reasoning_tokens == 0


def test_non_numeric_fields_are_ignored():
    """有的服务商会回 null，别让它变成 TypeError。"""
    u = _usage_of(_resp(prompt_tokens=None, completion_tokens="oops"))

    assert (u.prompt_tokens, u.completion_tokens) == (0, 0)


def test_usages_add_up():
    total = Usage(10, 20, 5, True) + Usage(1, 2, 1, False)

    assert (total.prompt_tokens, total.completion_tokens, total.reasoning_tokens) == (11, 22, 6)
    assert total.reported is True


# --- 评测侧汇总 -----------------------------------------------------------

def test_usage_metrics_average_over_all_rows():
    rows = [{"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 0},
            {"prompt_tokens": 300, "completion_tokens": 40, "reasoning_tokens": 10}]

    m = eg._usage_metrics(rows)

    assert m["avg_prompt_tokens"] == 200
    assert m["avg_completion_tokens"] == 30
    assert m["total_tokens"] == 460
    assert m["usage_reported"] == 1.0


def test_usage_metrics_flags_when_nothing_was_reported():
    """一排 0 不该冒充"没花钱"——服务端静默停发 usage 会长这样。"""
    rows = [{"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}] * 3

    m = eg._usage_metrics(rows)

    assert m["usage_reported"] == 0.0


def test_usage_metrics_reports_partial_coverage():
    rows = [{"prompt_tokens": 100, "completion_tokens": 10, "reasoning_tokens": 0},
            {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}]

    assert eg._usage_metrics(rows)["usage_reported"] == 0.5


def test_usage_metrics_on_empty_input():
    m = eg._usage_metrics([])

    assert m["total_tokens"] == 0 and m["usage_reported"] == 0.0


def test_print_usage_says_so_when_unmeasurable(capsys):
    eg._print_usage({"avg_prompt_tokens": 0.0, "avg_completion_tokens": 0.0,
                     "avg_reasoning_tokens": 0.0, "total_tokens": 0, "usage_reported": 0.0})

    assert "未返回 usage" in capsys.readouterr().out


def test_print_usage_warns_on_partial_coverage(capsys):
    eg._print_usage({"avg_prompt_tokens": 50.0, "avg_completion_tokens": 5.0,
                     "avg_reasoning_tokens": 0.0, "total_tokens": 110, "usage_reported": 0.5})

    assert "只有 50%" in capsys.readouterr().out


def test_print_usage_is_a_noop_for_old_result_files(capsys):
    """加 usage 之前的结果文件没有这些字段，--compare 不该因此炸掉。"""
    eg._print_usage({"avg_latency_ms": 100.0})

    assert capsys.readouterr().out == ""
