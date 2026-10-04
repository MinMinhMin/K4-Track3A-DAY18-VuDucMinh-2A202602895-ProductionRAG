"""Tests for parent-aware and version-aware pipeline integration."""
import json

from src import pipeline
from src.m2_search import SearchResult
from src.m3_rerank import RerankResult


def test_current_policy_query_prefers_latest_version():
    results = [
        SearchResult("leave 2023", 0.9, {"source": "nghi_phep_nam_v2023.md"}, "hybrid"),
        SearchResult("password v1", 0.8, {"source": "mat_khau_v1.md"}, "hybrid"),
        SearchResult("leave 2024", 0.7, {"source": "nghi_phep_nam_v2024.md"}, "hybrid"),
        SearchResult("password v2", 0.6, {"source": "mat_khau_v2.md"}, "hybrid"),
        SearchResult("other policy", 0.5, {"source": "thu_viec.md"}, "hybrid"),
    ]

    current = pipeline._prefer_latest_policy_versions(results, "Nhân viên được nghỉ bao nhiêu ngày phép năm?")
    explicit_old = pipeline._prefer_latest_policy_versions(results, "Theo chính sách năm 2023, phép năm là bao nhiêu?")
    explicit_v1 = pipeline._prefer_latest_policy_versions(results, "Theo chính sách mật khẩu v1.0, yêu cầu thế nào?")
    historical_comparison = pipeline._prefer_latest_policy_versions(results, "So sánh chính sách cũ và hiện hành")

    assert [result.text for result in current] == ["leave 2024", "password v2", "other policy"]
    assert [result.text for result in explicit_old] == ["leave 2023", "password v2", "other policy"]
    assert [result.text for result in explicit_v1] == ["password v1", "leave 2024", "other policy"]
    assert [result.text for result in historical_comparison] == [result.text for result in results]


def test_run_query_uses_parent_context(monkeypatch):
    parent_id = "policy.md::parent_0"
    parent_text = "# Chính sách nghỉ phép\nNhân viên được nghỉ 15 ngày phép năm."
    child_text = "được nghỉ 15 ngày"

    class FakeSearch:
        parent_text_by_id = {parent_id: parent_text}

        def search(self, query):
            return [SearchResult(child_text, 0.9, {"source": "policy.md", "parent_id": parent_id}, "hybrid")]

    class FakeReranker:
        def rerank(self, query, documents, top_k):
            assert documents[0]["text"] == child_text
            return [RerankResult(child_text, 0.9, 0.95, documents[0]["metadata"], 0)]

    monkeypatch.setattr(pipeline, "OPENAI_API_KEY", "")
    answer, contexts = pipeline.run_query("Bao nhiêu ngày phép?", FakeSearch(), FakeReranker())

    assert contexts == [parent_text]
    assert answer == parent_text


def test_run_query_reranks_one_child_per_parent(monkeypatch):
    parent_texts = {
        "leave.md::parent_0": "Ngày phép năm là 18 ngày.",
        "salary.md::parent_0": "Lương Senior là 20-35 triệu VNĐ/tháng.",
        "policy.md::parent_0": "Chính sách liên quan khác.",
    }
    results = [
        SearchResult("leave child 1", 0.9, {"source": "leave.md", "parent_id": "leave.md::parent_0"}, "hybrid"),
        SearchResult("leave child 2", 0.85, {"source": "leave.md", "parent_id": "leave.md::parent_0"}, "hybrid"),
        SearchResult("salary child", 0.8, {"source": "salary.md", "parent_id": "salary.md::parent_0"}, "hybrid"),
        SearchResult("other child", 0.7, {"source": "policy.md", "parent_id": "policy.md::parent_0"}, "hybrid"),
    ]

    class FakeSearch:
        parent_text_by_id = parent_texts

        def search(self, query):
            return results

    class FakeReranker:
        def rerank(self, query, documents, top_k):
            parent_ids = [doc["metadata"]["parent_id"] for doc in documents]
            assert len(parent_ids) == len(set(parent_ids))
            return [
                RerankResult(doc["text"], doc["score"], 1.0 - index / 10,
                             doc["metadata"], index)
                for index, doc in enumerate(documents[:top_k])
            ]

    monkeypatch.setattr(pipeline, "OPENAI_API_KEY", "")
    _, contexts = pipeline.run_query("Ngày phép và lương Senior?", FakeSearch(), FakeReranker())

    assert contexts == list(parent_texts.values())


def test_save_latency_report_writes_json(tmp_path):
    report = {
        "unit": "seconds",
        "num_queries": 20,
        "stages": {"chunking": {"total_seconds": 1.25}},
        "notes": ["query_processing overlaps individual query-stage totals."],
    }
    path = tmp_path / "latency_report.json"

    pipeline.save_latency_report(report, str(path))

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["num_queries"] == 20
    assert saved["stages"]["chunking"]["total_seconds"] == 1.25
    assert saved["notes"] == report["notes"]
