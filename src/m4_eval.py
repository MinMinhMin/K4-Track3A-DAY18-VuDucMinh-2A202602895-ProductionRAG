from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
import math
from dataclasses import asdict, dataclass, is_dataclass
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _empty_evaluation(error: str | None = None) -> dict:
    result = {name: 0.0 for name in METRIC_NAMES}
    result["per_question"] = []
    if error:
        result["evaluation_error"] = error
    return result


def _score_or_zero(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _contexts_or_fallback(value, fallback: list[str]) -> list[str]:
    """Normalize list-like dataframe cells without truth-testing NumPy arrays."""
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        return [value] if value else list(fallback)
    try:
        contexts = list(value)
    except TypeError:
        return list(fallback)
    return [str(context) for context in contexts if context is not None]


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        return _empty_evaluation("Input lists have different lengths")
    if not questions:
        return _empty_evaluation()
    if not OPENAI_API_KEY:
        return _empty_evaluation("OPENAI_API_KEY is not configured")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        frame = result.to_pandas()
        per_question = []
        for index, row in frame.iterrows():
            source_index = min(index, len(questions) - 1)
            row_contexts = _contexts_or_fallback(
                row.get("contexts", contexts[source_index]), contexts[source_index]
            )
            per_question.append(EvalResult(
                question=str(row.get("question", questions[source_index])),
                answer=str(row.get("answer", answers[source_index])),
                contexts=row_contexts,
                ground_truth=str(row.get("ground_truth", ground_truths[source_index])),
                faithfulness=_score_or_zero(row.get("faithfulness", 0.0)),
                answer_relevancy=_score_or_zero(row.get("answer_relevancy", 0.0)),
                context_precision=_score_or_zero(row.get("context_precision", 0.0)),
                context_recall=_score_or_zero(row.get("context_recall", 0.0)),
            ))

        aggregates = {
            name: (
                sum(getattr(item, name) for item in per_question) / len(per_question)
                if per_question else 0.0
            )
            for name in METRIC_NAMES
        }
        return {**aggregates, "per_question": per_question}
    except Exception as exc:
        error = f"RAGAS evaluation failed ({type(exc).__name__})"
        print(f"  ⚠️  {error}", flush=True)
        return _empty_evaluation(error)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": (
            "LLM generated claims not supported by the retrieved context",
            "Tighten the system prompt and lower answer temperature.",
        ),
        "answer_relevancy": (
            "The answer does not directly address the question",
            "Make the answer prompt require a direct response to the asked question.",
        ),
        "context_precision": (
            "Irrelevant context is ranked above useful evidence",
            "Improve hybrid retrieval and CrossEncoder reranking.",
        ),
        "context_recall": (
            "Retrieved context is missing evidence needed for the answer",
            "Improve chunk boundaries or add exact Vietnamese keywords to BM25.",
        ),
    }

    analyses = []
    for result in eval_results:
        scores = {name: _score_or_zero(getattr(result, name, 0.0)) for name in METRIC_NAMES}
        worst_metric = min(METRIC_NAMES, key=scores.get)
        average_score = sum(scores.values()) / len(METRIC_NAMES)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        item = asdict(result) if is_dataclass(result) else dict(result)
        item.update({
            "average_score": average_score,
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
        analyses.append(item)
    analyses.sort(key=lambda item: item["average_score"])
    return analyses[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    per_question = [
        asdict(item) if is_dataclass(item) else dict(item)
        for item in results.get("per_question", [])
    ]
    report = {
        "aggregate": {name: _score_or_zero(results.get(name, 0.0)) for name in METRIC_NAMES},
        "num_questions": len(per_question),
        "per_question": per_question,
        "failures": failures,
    }
    if results.get("evaluation_error"):
        report["evaluation_error"] = results["evaluation_error"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
