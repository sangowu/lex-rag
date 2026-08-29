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


# --- 线上跑的必须就是被评测的那条配置 -------------------------------------

def test_served_path_reranks_like_the_benchmarks_do():
    """`reranker.enabled` 曾是 false，于是线上走在一条从没被评测过的档位上。

    各评测脚本都靠 `--reranker` 打开它（`if args.reranker: enabled=True`，只加不减），
    所以文档里**所有**基线数字都是开着 reranker 测的；而 serve.py 从不覆盖这个
    字段，线上就一直不重排。症状和 top_k 那次漂移一样：完全无声，功能照常返回答案。
    """
    from lex_rag.config import load_config
    from lex_rag.strategy import RetrievalStrategy

    st = RetrievalStrategy.from_config(load_config())

    assert st.rerank is True


def test_candidate_pool_does_not_collapse_to_top_k():
    """不开 rerank 时 `fetch_k` 会塌成 `top_k`——线上连候选都比评测时少。

    见 strategy.py: `fetch_k = rerank_top_k if rerank_on else top_k`。
    这一条比上一条更隐蔽：就算哪天有人手工在 serve 里补上重排，候选池也得是 60。
    """
    from lex_rag.config import load_config
    from lex_rag.strategy import RetrievalStrategy

    cfg = load_config()
    st = RetrievalStrategy.from_config(cfg)

    assert st.fetch_k == cfg.retrieval.rerank_top_k
    assert st.fetch_k > st.top_k
