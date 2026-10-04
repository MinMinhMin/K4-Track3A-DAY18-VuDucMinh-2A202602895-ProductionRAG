"""Tests for Module 4: Evaluation."""
import sys, os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, EvalResult, save_report

def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]

def test_evaluate_returns_metrics():
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))

def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]


def test_evaluate_missing_key_has_safe_schema(monkeypatch):
    import src.m4_eval as m4

    monkeypatch.setattr(m4, "OPENAI_API_KEY", "", raising=False)
    result = m4.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])

    assert set(result) >= {
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "per_question", "evaluation_error",
    }
    assert all(result[key] == 0.0 for key in [
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    ])
    assert result["per_question"] == []
    assert result["evaluation_error"]


def test_ragas_normalizes_nan_scores_and_aggregates(monkeypatch):
    import numpy as np
    import pandas as pd
    import src.m4_eval as m4
    from types import ModuleType, SimpleNamespace

    monkeypatch.setattr(m4, "OPENAI_API_KEY", "configured-for-mocked-evaluation", raising=False)
    rows = [
        {"faithfulness": 0.5, "answer_relevancy": 0.25,
         "context_precision": float("nan"), "context_recall": None,
         "contexts": np.array(["c1", "c1-extra"])},
        {"faithfulness": 1.0, "answer_relevancy": 0.75,
         "context_precision": 0.2, "context_recall": 0.8},
    ]
    ragas_module = ModuleType("ragas")
    metrics_module = ModuleType("ragas.metrics")
    datasets_module = ModuleType("datasets")
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for name in metric_names:
        setattr(metrics_module, name, name)

    def fake_evaluate(dataset, metrics):
        assert dataset.data["question"] == ["q1", "q2"]
        assert metrics == metric_names
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame(rows))

    class FakeDataset:
        @staticmethod
        def from_dict(data):
            return SimpleNamespace(data=data)

    ragas_module.evaluate = fake_evaluate
    datasets_module.Dataset = FakeDataset
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_module)
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    result = m4.evaluate_ragas(["q1", "q2"], ["a1", "a2"], [["c1"], ["c2"]], ["g1", "g2"])

    assert result["faithfulness"] == 0.75
    assert result["answer_relevancy"] == 0.5
    assert result["context_precision"] == 0.1
    assert result["context_recall"] == 0.4
    assert result["per_question"][0].context_precision == 0.0
    assert result["per_question"][0].context_recall == 0.0


def test_failure_analysis_orders_bottom_n_by_average_score():
    results = [
        EvalResult("strong", "a", ["c"], "g", 0.9, 0.9, 0.9, 0.9),
        EvalResult("weak recall", "a", ["c"], "g", 0.4, 0.4, 0.4, 0.1),
        EvalResult("weak faithfulness", "a", ["c"], "g", 0.1, 0.6, 0.6, 0.6),
    ]

    failures = failure_analysis(results, bottom_n=2)

    assert [failure["question"] for failure in failures] == ["weak recall", "weak faithfulness"]
    assert failures[0]["worst_metric"] == "context_recall"
    assert failures[0]["diagnosis"]
    assert failures[0]["suggested_fix"]


def test_save_report_serializes_per_question_rows(tmp_path):
    result = EvalResult("q", "a", ["c"], "gt", 0.5, 0.6, 0.4, 0.3)
    path = tmp_path / "report.json"

    save_report({
        "faithfulness": 0.5,
        "answer_relevancy": 0.6,
        "context_precision": 0.4,
        "context_recall": 0.3,
        "per_question": [result],
    }, [], str(path))

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["num_questions"] == 1
    assert saved["per_question"][0]["question"] == "q"
    assert saved["aggregate"]["faithfulness"] == 0.5
