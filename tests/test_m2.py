"""Tests for Module 2: Hybrid Search."""
import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m2_search import segment_vietnamese, BM25Search, reciprocal_rank_fusion, SearchResult

CHUNKS = [
    {"text": "Nhân viên được nghỉ phép năm 12 ngày.", "metadata": {"source": "policy"}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "metadata": {"source": "it"}},
    {"text": "Thời gian thử việc là 60 ngày.", "metadata": {"source": "hr"}},
]

def test_segment_returns_string():
    assert isinstance(segment_vietnamese("nghỉ phép năm"), str)


def test_segment_vietnamese_splits_compounds(monkeypatch):
    import underthesea

    monkeypatch.setattr(underthesea, "word_tokenize", lambda text, format: "nghỉ_phép năm")
    assert segment_vietnamese("input ignored by fake tokenizer") == "nghỉ phép năm"

def test_bm25_search():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search("nghỉ phép", top_k=2)
    assert len(results) > 0 and results[0].method == "bm25"

def test_bm25_relevant_first():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search("nghỉ phép năm", top_k=2)
    if results:
        assert "nghỉ" in results[0].text.lower() or "12" in results[0].text


def test_bm25_omits_nonpositive_scores():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    assert bm25.search("unmatchedterm_xyz", top_k=3) == []

def test_rrf_merges():
    a = [SearchResult("doc1", 0.9, {}, "bm25"), SearchResult("doc2", 0.8, {}, "bm25")]
    b = [SearchResult("doc2", 0.95, {}, "dense"), SearchResult("doc3", 0.85, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], top_k=3)
    assert len(merged) > 0 and "doc2" in [r.text for r in merged]

def test_rrf_method():
    a = [SearchResult("d1", 0.9, {}, "bm25")]
    b = [SearchResult("d1", 0.8, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], top_k=1)
    if merged:
        assert merged[0].method == "hybrid"


def test_rrf_deduplicates_and_sums_reciprocal_ranks():
    a = [SearchResult("doc1", 0.9, {"source": "a"}, "bm25"),
         SearchResult("doc2", 0.8, {}, "bm25")]
    b = [SearchResult("doc2", 0.95, {}, "dense"),
         SearchResult("doc1", 0.85, {}, "dense")]

    merged = reciprocal_rank_fusion([a, b], k=60, top_k=5)
    by_text = {result.text: result for result in merged}

    assert set(by_text) == {"doc1", "doc2"}
    assert by_text["doc1"].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_text["doc2"].score == pytest.approx(1 / 62 + 1 / 61)
    assert all(result.method == "hybrid" for result in merged)


def test_dense_search_queries_in_memory_qdrant(monkeypatch):
    import numpy as np
    from qdrant_client import QdrantClient
    from src.m2_search import DenseSearch

    class FakeEncoder:
        def encode(self, values, **kwargs):
            if isinstance(values, str):
                values = [values]
            vectors = []
            for value in values:
                vector = [0.0] * 1024
                vector[0 if "nghỉ" in value.lower() else 1] = 1.0
                vectors.append(vector)
            return np.asarray(vectors, dtype=np.float32)

    dense = DenseSearch()
    dense.client = QdrantClient(":memory:")
    dense._encoder = FakeEncoder()
    dense.index(CHUNKS, collection="test_dense_query_points")

    with monkeypatch.context() as patch:
        original_query_points = dense.client.query_points
        calls = []

        def recording_query_points(*args, **kwargs):
            calls.append((args, kwargs))
            return original_query_points(*args, **kwargs)

        patch.setattr(dense.client, "query_points", recording_query_points)
        results = dense.search("nghỉ phép", top_k=2, collection="test_dense_query_points")

    assert len(calls) == 1
    assert results
    assert results[0].method == "dense"
    assert results[0].text == CHUNKS[0]["text"]
