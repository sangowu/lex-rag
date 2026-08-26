"""SufficiencyJudge 与 VerifiedGenerator 的单元测试（不碰网络）。

重点在**容错方向**。json_object 模式不保证字段齐全，所以每个字段都会走到缺省
分支；而这里的缺省不是随便挑的：判成"不够"最多白烧一轮，判成"够了"会拿着残缺
上下文直接生成。测试把这个方向钉死，避免以后有人为了"少跑一轮"把它翻过来。
"""

from unittest.mock import MagicMock, patch

import pytest

from lex_rag.chunking import ChunkWindow
from lex_rag.config import ContextualConfig
from lex_rag.generator import VerifiedGenerator
from lex_rag.sufficiency import STRATEGY_HINT, SufficiencyJudge, Verdict, _coerce


def _cfg(**kw) -> ContextualConfig:
    base = dict(enabled=True, model="m", api_key="k", rpm_limit=60,
                max_retries=0, retry_backoff_sec=0.0)
    base.update(kw)
    return ContextualConfig(**base)


def _chunks(n: int = 2) -> list[ChunkWindow]:
    return [ChunkWindow(chunk_id=f"d#{i}", doc_id="d", text=f"clause {i}",
                        start=i * 100, end=i * 100 + 50) for i in range(n)]


def _judge(mode: str = "sufficiency", reply: dict | None = None) -> SufficiencyJudge:
    j = SufficiencyJudge(_cfg(), mode=mode)
    j._chat = MagicMock()
    j._chat.complete_json.return_value = reply if reply is not None else {}
    return j


# ── 解析容错：缺省方向必须偏保守 ──────────────────────────────────

def test_missing_fields_default_to_insufficient():
    """模型少给字段时按"不够"处理——反向的代价大得多。"""
    v = _coerce({}, unified=False)
    assert v.sufficient is False and v.out_of_scope is False
    assert v.missing_kind == "none" and v.confidence == 0.0


def test_unknown_missing_kind_falls_back_to_none():
    """枚举外的值不能直接透传给策略选择器，否则它会查不到映射。"""
    assert _coerce({"missing_kind": "needs_more_vibes"}, unified=False).missing_kind == "none"
    assert _coerce({"missing_kind": "EXACT_TERM"}, unified=False).missing_kind == "exact_term"


def test_confidence_is_clamped_and_survives_garbage():
    assert _coerce({"confidence": 3.7}, unified=False).confidence == 1.0
    assert _coerce({"confidence": -1}, unified=False).confidence == 0.0
    assert _coerce({"confidence": "high"}, unified=False).confidence == 0.0


def test_answer_supported_only_exists_in_unified_mode():
    assert _coerce({}, unified=False).answer_supported is None
    # unified 下缺字段同样按危险方向缺省：不认为答案已被支持
    assert _coerce({}, unified=True).answer_supported is False


def test_strategy_hint_covers_every_missing_kind():
    """任何一个 kind 都必须能查到下一步动作，否则选择器会拿到空手。"""
    for kind in STRATEGY_HINT:
        v = Verdict(missing_kind=kind)
        assert v.strategy_hint == STRATEGY_HINT[kind]
    assert Verdict(missing_kind="exact_term").strategy_hint == "bm25"


# ── judge 调用 ────────────────────────────────────────────────────

def test_empty_chunks_short_circuits_without_calling_llm():
    j = _judge()
    v = j.judge("q", [])
    assert v.sufficient is False and not j._chat.complete_json.called


def test_llm_failure_returns_verdict_instead_of_raising():
    """判定器崩了不该中断整个检索循环——错误落进 Verdict.error。"""
    j = _judge()
    j._chat.complete_json.side_effect = RuntimeError("429 rate limited")

    v = j.judge("q", _chunks())
    assert v.sufficient is False and v.error is not None and "429" in v.error


def test_unified_prompt_includes_draft_and_sufficiency_prompt_does_not():
    uni = _judge("unified")
    uni.judge("q", _chunks(), draft_answer="Yes. 'clause 0' [1].")
    assert "clause 0" in uni._chat.complete_json.call_args[0][0]

    spec = _judge("sufficiency")
    spec.judge("q", _chunks(), draft_answer="Yes. 'clause 0' [1].")
    prompt = spec._chat.complete_json.call_args[0][0]
    assert "Draft answer" not in prompt


def test_refusing_draft_is_described_rather_than_left_blank():
    """空字符串会让 prompt 变成 "Draft answer:" 后面什么都没有，模型只能瞎猜。"""
    uni = _judge("unified")
    uni.judge("q", _chunks(), draft_answer="")
    assert "refused" in uni._chat.complete_json.call_args[0][0]


def test_unknown_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        SufficiencyJudge(_cfg(), mode="whatever")


# ── VerifiedGenerator：两段式的翻转规则 ───────────────────────────

def _vgen(draft_refused: bool, verdict_fields: dict,
          escalate: bool = True) -> tuple[VerifiedGenerator, MagicMock]:
    from lex_rag.generator import GenerationResult

    g = VerifiedGenerator(_cfg(), escalate=escalate)
    g._fast = MagicMock()
    g._fast.generate.return_value = GenerationResult(
        question="q", answer="" if draft_refused else "Yes. 'clause 0' [1].",
        is_refused=draft_refused, latency_ms=800.0,
    )
    g._slow = MagicMock()
    g._slow.generate.return_value = GenerationResult(
        question="q", answer="rescued answer", is_refused=False, latency_ms=7000.0,
    )
    g._judge = MagicMock()
    g._judge.judge.return_value = Verdict(**verdict_fields)
    return g, g._slow


def test_unsupported_answer_is_flipped_to_refusal():
    """FP 的主要来源：模型引用了一条相关但不同的条款。"""
    g, slow = _vgen(False, {"answer_supported": False, "sufficient": True})
    r = g.generate("q", _chunks())
    assert r.is_refused and r.answer == "" and r.flipped == "to_refusal"
    assert r.llm_calls == 2 and not slow.generate.called


def test_out_of_scope_also_flips_even_when_answer_looks_supported():
    g, _ = _vgen(False, {"answer_supported": True, "out_of_scope": True})
    assert g.generate("q", _chunks()).flipped == "to_refusal"


def test_supported_answer_is_kept_untouched():
    g, slow = _vgen(False, {"answer_supported": True, "sufficient": True})
    r = g.generate("q", _chunks())
    assert not r.is_refused and r.flipped is None and r.llm_calls == 2
    assert not slow.generate.called


def test_wrong_refusal_is_escalated_to_the_thinking_path():
    """FN 的补救：草稿拒答了，但判定器说上下文其实够。"""
    g, slow = _vgen(True, {"sufficient": True, "out_of_scope": False})
    r = g.generate("q", _chunks())
    assert slow.generate.called
    assert r.answer == "rescued answer" and r.flipped == "escalated" and r.llm_calls == 3


def test_refusal_is_kept_when_judge_agrees_context_is_missing():
    """判定器也说不够时不该升级——那只是白花一次 thinking 的钱。"""
    g, slow = _vgen(True, {"sufficient": False})
    r = g.generate("q", _chunks())
    assert r.is_refused and r.flipped is None and not slow.generate.called


def test_out_of_scope_refusal_is_never_escalated():
    g, slow = _vgen(True, {"sufficient": True, "out_of_scope": True})
    assert not slow.generate.called
    assert g.generate("q", _chunks()).is_refused


def test_escalation_can_be_switched_off():
    g, slow = _vgen(True, {"sufficient": True}, escalate=False)
    g.generate("q", _chunks())
    assert not slow.generate.called


def test_generation_error_short_circuits_before_the_judge():
    """草稿都没拿到就没什么可校验的，多打一次 judge 只是浪费额度。"""
    from lex_rag.generator import GenerationResult

    g, _ = _vgen(False, {"answer_supported": True})
    g._fast.generate.return_value = GenerationResult(
        question="q", answer="", is_refused=False, error="429")
    r = g.generate("q", _chunks())
    assert r.error == "429" and not g._judge.judge.called


def test_fast_path_disables_thinking_and_escalation_path_enables_it():
    """两段式的全部意义就在这两个开关上，配错了就是白改。"""
    with patch("lex_rag.generator.LegalGenerator") as LG:
        VerifiedGenerator(_cfg(thinking=True))
    fast_cfg, slow_cfg = LG.call_args_list[0][0][0], LG.call_args_list[1][0][0]
    assert fast_cfg.thinking is False and slow_cfg.thinking is True
