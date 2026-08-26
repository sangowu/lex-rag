"""三个辅助 LLM 客户端的缓存行为测试。

只测一件事，但这件事是个真实事故：**降级结果不许进缓存**。

`.cache/hyde.json` 曾经 41 条全部等于原问题——迁移期 LLM 调用失败走了降级分支
`hypo = question`，而降级结果被写进了缓存。缓存命中让它此后永不重试，HyDE 就此
长期是个空操作，而且完全静默：`hybrid vs hyde` 的检索结果重合度精确等于 1.000，
不去量它根本发现不了。

同样的写法在 MetadataExtractor 和 QueryExpander 里各有一份，一并钉住。
"""

import json
from unittest.mock import MagicMock

from lex_rag.config import ContextualConfig
from lex_rag.contextualizer import HyDEClient, MetadataExtractor, QueryExpander


def _cfg(**kw) -> ContextualConfig:
    base = dict(enabled=True, model="m", api_key="k", rpm_limit=6000,
                max_retries=0, retry_backoff_sec=0.0)
    base.update(kw)
    return ContextualConfig(**base)


def _client(cls, tmp_path, name, reply=None, error=None, **kw):
    c = cls(_cfg(), cache_path=tmp_path / name, **kw)
    c._chat = MagicMock()
    if error is not None:
        c._chat.complete.side_effect = error
    else:
        c._chat.complete.return_value = reply
    return c


def _cache_on_disk(tmp_path, name) -> dict:
    p = tmp_path / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ── HyDE ──────────────────────────────────────────────────────

def test_hyde_does_not_cache_the_degraded_fallback(tmp_path):
    """LLM 挂了要降级返回原问题，但**不能把它写进缓存**——那会永久化一次故障。"""
    h = _client(HyDEClient, tmp_path, "h.json", error=RuntimeError("429"))
    assert h.generate("q") == "q"                 # 降级行为保留
    assert _cache_on_disk(tmp_path, "h.json") == {}   # 但没落盘


def test_hyde_retries_the_llm_after_a_failure_instead_of_serving_the_fallback(tmp_path):
    """这才是修复的意义：下一次调用要真的再试一遍，而不是从缓存里取回降级值。"""
    h = _client(HyDEClient, tmp_path, "h.json", error=RuntimeError("429"))
    h.generate("q")

    h._chat.complete.side_effect = None
    h._chat.complete.return_value = "This Agreement shall be governed by ..."
    assert h.generate("q").startswith("This Agreement")
    assert len(_cache_on_disk(tmp_path, "h.json")) == 1


def test_hyde_caches_a_successful_generation(tmp_path):
    h = _client(HyDEClient, tmp_path, "h.json", reply="hypothetical clause")
    assert h.generate("q") == "hypothetical clause"
    assert list(_cache_on_disk(tmp_path, "h.json").values()) == ["hypothetical clause"]

    h._chat.complete.reset_mock()
    assert h.generate("q") == "hypothetical clause"
    assert not h._chat.complete.called            # 第二次走缓存，不再调 LLM


def test_hyde_caches_output_that_happens_to_resemble_the_question(tmp_path):
    """判据是"这次有没有降级"，不是"结果是否等于输入"。

    按后者判会把模型合法的短回答误当成故障，反而制造新的不缓存路径。
    """
    h = _client(HyDEClient, tmp_path, "h.json", reply="q")
    assert h.generate("q") == "q"
    assert len(_cache_on_disk(tmp_path, "h.json")) == 1


# ── MetadataExtractor ─────────────────────────────────────────

def test_meta_extractor_does_not_cache_empty_meta_on_failure(tmp_path):
    """空 meta 被缓存的话，生成层从此永远拿不到元数据前缀，且毫无报错。"""
    m = _client(MetadataExtractor, tmp_path, "m.json", error=RuntimeError("boom"))
    out = m.extract("D", "contract text")
    assert not any(out.get(k) for k in ("contract_type", "party_a", "governing_law"))
    assert _cache_on_disk(tmp_path, "m.json") == {}


def test_meta_extractor_does_not_cache_unparseable_output(tmp_path):
    m = _client(MetadataExtractor, tmp_path, "m.json", reply="not json")
    m.extract("D", "contract text")
    assert _cache_on_disk(tmp_path, "m.json") == {}


def test_meta_extractor_caches_a_successful_extraction(tmp_path):
    m = _client(MetadataExtractor, tmp_path, "m.json",
                reply='{"contract_type": "Distributor Agreement"}')
    out = m.extract("D", "contract text")
    assert out["contract_type"] == "Distributor Agreement"
    assert len(_cache_on_disk(tmp_path, "m.json")) == 1


# ── QueryExpander ─────────────────────────────────────────────

def test_expander_does_not_cache_the_single_variant_fallback(tmp_path):
    """只剩原问题就是降级，缓存了等于永久关闭 multi-query。"""
    e = _client(QueryExpander, tmp_path, "e.json", error=RuntimeError("boom"), n=3)
    assert e.expand("q") == ["q"]
    assert _cache_on_disk(tmp_path, "e.json") == {}


def test_expander_does_not_cache_unparseable_output(tmp_path):
    e = _client(QueryExpander, tmp_path, "e.json", reply="not json", n=3)
    e.expand("q")
    assert _cache_on_disk(tmp_path, "e.json") == {}


def test_expander_caches_a_successful_expansion(tmp_path):
    e = _client(QueryExpander, tmp_path, "e.json", reply='["a", "b"]', n=3)
    assert e.expand("q") == ["q", "a", "b"]
    assert len(_cache_on_disk(tmp_path, "e.json")) == 1
