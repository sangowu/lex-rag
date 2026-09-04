"""`eval_generation.py` 的命令行开关是否真的落到 config 上。

起因：`--table` 曾经**只被声明、从没被读过**。`--table chunks_ocr` 被静默忽略，
评测照样跑完、照样出一份漂亮结果——只不过跑的是默认表。两臂指标于是完全相同，
看上去像"OCR 没有造成任何损失"这么一个好消息。

⚠️ **判据不能是"跑起来不报错"**——坏掉的那版也不报错。判据必须是"覆盖之后 cfg
里那个字段真的变了"。所以每个开关各钉一条，新增开关时照抄一条。

本仓库第五次栽在"配置变了、读它的人没跟着变、而且完全无声"上（前四次：serve.py
写死 top_k、reranker.enabled、eval.py 写死 hit@k、门禁自带的第二份归一化）。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_generation.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("eval_generation", _PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def cfg(monkeypatch):
    """load_config() 硬性要求这几个环境变量；CI 里没有 .env。

    本地有 .env 时不覆盖真值——否则这条测试在两台机器上测的不是同一个 cfg。
    """
    from lex_rag.config import load_config
    for key in ("EMBED_API_KEY", "PG_PASSWORD"):
        if key not in os.environ:
            monkeypatch.setenv(key, "test-not-a-real-credential")
    return load_config()


def make_args(**kw):
    base = dict(table=None, reranker=False, thinking=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_table_override_reaches_the_config(mod, cfg):
    """这一条就是那个 bug 本身。"""
    out = mod.apply_cli_overrides(cfg, make_args(table="chunks_ocr"))
    assert out.database.table == "chunks_ocr"


def test_no_table_flag_keeps_the_configured_table(mod, cfg):
    out = mod.apply_cli_overrides(cfg, make_args())
    assert out.database.table == cfg.database.table


def test_reranker_flag_reaches_the_config(mod, cfg):
    assert mod.apply_cli_overrides(cfg, make_args(reranker=True)).reranker.enabled is True


def test_the_reranker_flag_only_adds_never_removes(mod, cfg):
    """不传 `--reranker` 不等于关掉它——config.yaml 里它本来就是 true，
    而文档里所有基线都是开着 reranker 测的。"""
    out = mod.apply_cli_overrides(cfg, make_args(reranker=False))
    assert out.reranker.enabled == cfg.reranker.enabled


@pytest.mark.parametrize("value", [True, False])
def test_thinking_can_be_switched_both_ways(mod, cfg, value):
    """A/B 要单变量，所以两个方向都得能从命令行切；
    `None` 表示"别动它"，与 `False` 是不同的意思。"""
    assert mod.apply_cli_overrides(cfg, make_args(thinking=value)).contextual.thinking is value


def test_thinking_left_unset_does_not_touch_the_config(mod, cfg):
    out = mod.apply_cli_overrides(cfg, make_args(thinking=None))
    assert out.contextual.thinking == cfg.contextual.thinking


def test_overrides_do_not_mutate_the_original_config(mod, cfg):
    """dataclasses.replace 返回新对象；就地改会让同进程里的后续 run 串味。"""
    before = cfg.database.table
    mod.apply_cli_overrides(cfg, make_args(table="chunks_other", reranker=True))
    assert cfg.database.table == before


def test_every_declared_switch_is_actually_applied(mod, cfg):
    """**这条是防止同一个 bug 再来一次的那一条。**

    上面每条测的是一个已知开关。这条反过来问：`apply_cli_overrides` 认得的开关，
    是不是每一个都真的改动了 cfg？新加一个只声明不读的开关会在这里露出来。
    """
    sentinel = {"table": "a-table-that-is-not-the-default"}
    for field, value in sentinel.items():
        out = mod.apply_cli_overrides(cfg, make_args(**{field: value}))
        assert out != cfg, f"{field} 传了值却没改动 cfg"
