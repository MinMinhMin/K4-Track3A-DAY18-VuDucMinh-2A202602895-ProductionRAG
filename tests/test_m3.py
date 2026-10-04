"""Tests for Module 3: Reranking."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m3_rerank import CrossEncoderReranker, benchmark_reranker, RerankResult

Q = "Nhân viên được nghỉ phép bao nhiêu ngày?"
DOCS = [
    {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
    {"text": "VPN dùng WireGuard AES-256.", "score": 0.6, "metadata": {}},
]


def test_rerank_empty_documents_skips_model(monkeypatch):
    reranker = CrossEncoderReranker()

    def fail_if_loaded():
        raise AssertionError("model must not load for empty input")

    monkeypatch.setattr(reranker, "_load_model", fail_if_loaded)
    assert reranker.rerank(Q, []) == []


def test_rerank_preserves_score_document_pairing_and_top_k():
    class FakeCrossEncoder:
        def predict(self, pairs):
            assert pairs == [(Q, doc["text"]) for doc in DOCS]
            return [0.1, 0.9, 0.4]

    docs = [
        {**DOCS[0], "metadata": {"source": "leave"}},
        {**DOCS[1], "metadata": {"source": "password"}},
        {**DOCS[2], "metadata": {"source": "vpn"}},
    ]
    reranker = CrossEncoderReranker()
    reranker._model = FakeCrossEncoder()

    results = reranker.rerank(Q, docs, top_k=2)

    assert [result.text for result in results] == [DOCS[1]["text"], DOCS[2]["text"]]
    assert [result.rerank_score for result in results] == [0.9, 0.4]
    assert [result.original_score for result in results] == [0.7, 0.6]
    assert [result.metadata["source"] for result in results] == ["password", "vpn"]
    assert [result.rank for result in results] == [0, 1]


def test_rerank_model_error_returns_empty_for_pipeline_fallback(monkeypatch, capsys):
    reranker = CrossEncoderReranker()

    def fail_to_load():
        raise OSError("model unavailable")

    monkeypatch.setattr(reranker, "_load_model", fail_to_load)
    assert reranker.rerank(Q, DOCS) == []
    assert "model unavailable" in capsys.readouterr().out


def test_cross_encoder_model_is_cached_across_rerankers(monkeypatch):
    import src.m3_rerank as m3

    calls = []
    fake_model = object()

    def fake_constructor(model_name):
        calls.append(model_name)
        return fake_model

    if hasattr(m3, "_load_cross_encoder_model"):
        m3._load_cross_encoder_model.cache_clear()
    monkeypatch.setattr("sentence_transformers.CrossEncoder", fake_constructor)
    try:
        first = CrossEncoderReranker()
        second = CrossEncoderReranker()
        assert first._load_model() is fake_model
        assert second._load_model() is fake_model
        assert calls == ["BAAI/bge-reranker-v2-m3"]
    finally:
        if hasattr(m3, "_load_cross_encoder_model"):
            m3._load_cross_encoder_model.cache_clear()


def test_cross_encoder_records_model_load_latency(monkeypatch):
    import src.m3_rerank as m3

    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(m3.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(m3, "_load_cross_encoder_model", lambda model_name: object())
    reranker = CrossEncoderReranker()

    reranker._load_model()

    assert reranker.model_load_seconds == 0.25

def test_rerank_returns():
    r = CrossEncoderReranker().rerank(Q, DOCS, top_k=2)
    assert len(r) > 0 and len(r) <= 2

def test_rerank_type():
    assert all(isinstance(x, RerankResult) for x in CrossEncoderReranker().rerank(Q, DOCS))

def test_rerank_sorted():
    r = CrossEncoderReranker().rerank(Q, DOCS)
    if len(r) >= 2:
        assert r[0].rerank_score >= r[1].rerank_score

def test_rerank_relevant_first():
    r = CrossEncoderReranker().rerank(Q, DOCS)
    if r:
        assert "nghỉ" in r[0].text.lower() or "12" in r[0].text

def test_benchmark_stats():
    stats = benchmark_reranker(CrossEncoderReranker(), Q, DOCS, n_runs=2)
    assert "avg_ms" in stats and "min_ms" in stats and "max_ms" in stats
