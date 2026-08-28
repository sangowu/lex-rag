"""API 默认值不再写死在 serve.py 里。

这里钉的是一个具体的漂移事故：QueryRequest 曾把 top_k=10 / generate_k=8 硬编码，
config.yaml 把 top_k 提到 20 之后 API 仍按 10 检索，而且完全无声——只有跑评测
才看得出来。所以两条：默认 top_k 必须取自配置，generate_k 必须跟随 top_k。
"""

import pytest

from scripts import serve
from scripts.serve import QueryRequest


@pytest.fixture
def cfg_top_k(monkeypatch):
    def _set(value: int):
        monkeypatch.setattr(serve, "_default_top_k", lambda: value)
    return _set


def test_top_k_defaults_to_config(cfg_top_k):
    cfg_top_k(20)
    assert QueryRequest(question="q").resolved_top_k() == 20


def test_config_change_reaches_the_api(cfg_top_k):
    """配置改了，API 默认值就得跟着改——这正是原来漂移掉的那一环。"""
    cfg_top_k(30)
    assert QueryRequest(question="q").resolved_top_k() == 30


def test_generate_k_follows_top_k(cfg_top_k):
    cfg_top_k(20)
    req = QueryRequest(question="q")
    assert req.resolved_generate_k() == req.resolved_top_k() == 20


def test_generate_k_follows_an_explicit_top_k(cfg_top_k):
    """跟随的是这次请求的 top_k，不是配置里的那个。"""
    cfg_top_k(20)
    req = QueryRequest(question="q", top_k=5)
    assert req.resolved_generate_k() == 5


def test_explicit_values_still_win(cfg_top_k):
    cfg_top_k(20)
    req = QueryRequest(question="q", top_k=12, generate_k=8)
    assert (req.resolved_top_k(), req.resolved_generate_k()) == (12, 8)


def test_zero_is_accepted_as_follow_the_default(cfg_top_k):
    """0 是"跟随"的哨兵值，不能被 ge=1 挡掉。"""
    cfg_top_k(20)
    req = QueryRequest(question="q", top_k=0, generate_k=0)
    assert (req.resolved_top_k(), req.resolved_generate_k()) == (20, 20)
