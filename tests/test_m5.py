"""Tests for Module 5: Enrichment Pipeline."""
import sys, os
import json
from types import ModuleType, SimpleNamespace
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m5_enrichment import (
    summarize_chunk, generate_hypothesis_questions,
    contextual_prepend, extract_metadata, enrich_chunks, EnrichedChunk,
)
import src.m5_enrichment as m5

SAMPLE = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
CHUNKS = [
    {"text": SAMPLE, "metadata": {"source": "policy.md"}},
    {"text": "Mật khẩu phải thay đổi mỗi 90 ngày.", "metadata": {"source": "it.md"}},
]


def test_summarize_returns_string():
    result = summarize_chunk(SAMPLE)
    assert isinstance(result, str)


def test_summarize_shorter_than_original():
    result = summarize_chunk(SAMPLE)
    if result:  # May be empty if no API key
        assert len(result) <= len(SAMPLE) * 2  # Summary should not be much longer


def test_hyqa_returns_list():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    assert isinstance(result, list)


def test_hyqa_generates_questions():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    if result:
        assert len(result) >= 1
        assert any("?" in q or "bao" in q.lower() or "mấy" in q.lower() for q in result)


def test_contextual_prepend_returns_string():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert isinstance(result, str)
    assert len(result) >= len(SAMPLE)  # Should be at least as long as original


def test_contextual_contains_original():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert SAMPLE in result  # Original text must be preserved


def test_extract_metadata_returns_dict():
    result = extract_metadata(SAMPLE)
    assert isinstance(result, dict)


def test_enrich_chunks_returns_list():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    assert isinstance(result, list)


def test_enrich_chunks_type():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    if result:
        assert all(isinstance(c, EnrichedChunk) for c in result)


def test_enrich_preserves_original():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    if result:
        assert result[0].original_text == SAMPLE


def test_contextual_fallback_includes_optional_source_title(monkeypatch):
    monkeypatch.setattr(m5, "OPENAI_API_KEY", "")

    with_title = contextual_prepend(SAMPLE, "policy.md")
    without_title = contextual_prepend(SAMPLE)

    assert with_title.startswith("Trích từ policy.md.")
    assert SAMPLE in with_title
    assert without_title == SAMPLE


def test_combined_enrichment_makes_one_call_per_chunk(monkeypatch):
    monkeypatch.setattr(m5, "OPENAI_API_KEY", "test-key")
    calls = []
    payload = {
        "summary": "Tóm tắt ngắn.",
        "questions": ["Nhân viên được nghỉ bao nhiêu ngày?"],
        "context": "Quy định nghỉ phép trong policy.md.",
        "metadata": {"topic": "nghỉ phép", "entities": ["nhân viên"], "category": "hr", "language": "vi"},
    }

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    enriched = enrich_chunks(CHUNKS)

    assert len(calls) == len(CHUNKS)
    assert all(call["model"] == "gpt-4o-mini" for call in calls)
    assert [chunk.original_text for chunk in enriched] == [chunk["text"] for chunk in CHUNKS]
    assert enriched[0].enriched_text == "\n\n".join([
        payload["context"], payload["summary"], payload["questions"][0], SAMPLE
    ])
    assert enriched[0].auto_metadata["source"] == "policy.md"
    assert enriched[0].auto_metadata["category"] == "hr"


def test_combined_enrichment_falls_back_on_invalid_response(monkeypatch, capsys):
    monkeypatch.setattr(m5, "OPENAI_API_KEY", "test-key")
    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))])
    )))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = m5._enrich_single_call(SAMPLE, "policy.md")

    assert result["summary"]
    assert result["questions"]
    assert result["context"].startswith("Trích từ policy.md.")
    assert result["metadata"]["source"] == "policy.md"
    assert "Expecting value" not in capsys.readouterr().out


def test_combined_enrichment_falls_back_on_api_error(monkeypatch):
    monkeypatch.setattr(m5, "OPENAI_API_KEY", "test-key")

    def fail(**kwargs):
        raise OSError("network unavailable")

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = m5._enrich_single_call(SAMPLE, "policy.md")
    assert result["context"].startswith("Trích từ policy.md.")
    assert result["metadata"]["source"] == "policy.md"
