"""Unit tests for LegalGenerator's refusal-gate parsing logic (Gemini call mocked out)."""

from unittest.mock import MagicMock

from legal_rag_v1.chunking import ChunkWindow
from legal_rag_v1.config import ContextualConfig
from legal_rag_v1.generator import LegalGenerator


def _cfg() -> ContextualConfig:
    return ContextualConfig(
        enabled=True,
        model="gemini-test",
        api_key="key",
        rpm_limit=60,
        max_retries=0,
        retry_backoff_sec=0.0,
    )


def _chunk(chunk_id: str, doc_id: str, text: str) -> ChunkWindow:
    return ChunkWindow(chunk_id=chunk_id, doc_id=doc_id, text=text, start=0, end=len(text))


# ── _parse_response ──────────────────────────────────────────────


def test_parse_response_refused_flag_returns_empty_answer_and_no_citations():
    gen = LegalGenerator(_cfg())
    data = {"refused": True, "answer": "should be ignored"}

    answer, is_refused, citations = gen._parse_response(data, [])

    assert (answer, is_refused, citations) == ("", True, [])


def test_parse_response_blank_answer_is_treated_as_refusal():
    gen = LegalGenerator(_cfg())
    data = {"refused": False, "answer": "   "}

    answer, is_refused, citations = gen._parse_response(data, [])

    assert (answer, is_refused, citations) == ("", True, [])


def test_parse_response_extracts_numeric_citation():
    gen = LegalGenerator(_cfg())
    chunks = [_chunk("c1", "doc1", "hello world"), _chunk("c2", "doc1", "other text")]
    data = {"refused": False, "answer": 'Yes. "hello world" [1].'}

    answer, is_refused, citations = gen._parse_response(data, chunks)

    assert is_refused is False
    assert answer == 'Yes. "hello world" [1].'
    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"
    assert citations[0].num == 1


def test_parse_response_dedups_repeated_citation_numbers():
    gen = LegalGenerator(_cfg())
    chunks = [_chunk("c1", "doc1", "hello world")]
    data = {"refused": False, "answer": '"hello" [1], also "world" [1].'}

    _, _, citations = gen._parse_response(data, chunks)

    assert len(citations) == 1


def test_parse_response_out_of_range_citation_number_is_ignored():
    gen = LegalGenerator(_cfg())
    chunks = [_chunk("c1", "doc1", "hello world")]
    data = {"refused": False, "answer": 'Yes. "hello" [5].'}

    _, _, citations = gen._parse_response(data, chunks)

    assert citations == []


def test_parse_response_falls_back_to_contract_name_citation():
    gen = LegalGenerator(_cfg())
    chunks = [_chunk("c1", "doc1", "hello world")]
    data = {"refused": False, "answer": 'Yes. "hello world" [Contract: doc1].'}

    _, _, citations = gen._parse_response(data, chunks)

    assert len(citations) == 1
    assert citations[0].doc_id == "doc1"


# ── generate() ───────────────────────────────────────────────────


def test_generate_refuses_immediately_when_no_chunks_retrieved():
    gen = LegalGenerator(_cfg())

    result = gen.generate("question?", [])

    assert result.is_refused is True
    assert result.error == "no chunks retrieved"


def test_generate_parses_call_gemini_result_into_citations():
    gen = LegalGenerator(_cfg())
    gen._call_gemini = MagicMock(return_value={"refused": False, "answer": 'Yes "clause" [1].'})
    chunks = [_chunk("c1", "doc1", "clause text")]

    result = gen.generate("question?", chunks)

    assert result.is_refused is False
    assert result.answer == 'Yes "clause" [1].'
    assert len(result.citations) == 1
    assert result.error is None


def test_generate_returns_error_result_when_call_gemini_raises():
    gen = LegalGenerator(_cfg())
    gen._call_gemini = MagicMock(side_effect=RuntimeError("boom"))
    chunks = [_chunk("c1", "doc1", "clause text")]

    result = gen.generate("question?", chunks)

    assert result.error == "boom"
    assert result.answer == ""
