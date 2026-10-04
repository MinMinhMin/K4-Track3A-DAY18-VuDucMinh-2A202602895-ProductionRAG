from __future__ import annotations

"""Production RAG pipeline integrating chunking, enrichment, search and evaluation."""

import json
import os
import re
import sys
import time
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, RERANK_TOP_K
from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks


_VERSION_SUFFIX = re.compile(r"^(?P<family>.*?)(?:_v(?P<version>\d+))(?=\.[^.]+$|$)", re.IGNORECASE)
_REQUESTED_YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_REQUESTED_VERSION = re.compile(r"\b(?:v|version|phiên\s*bản)\s*0*(\d+)(?:\.\d+)?\b", re.IGNORECASE)
_GENERIC_HISTORY = re.compile(r"\b(?:cũ|trước đây|lịch sử|historical|old|previous)\b", re.IGNORECASE)


def _source_version(source: str) -> tuple[str, int] | None:
    """Return normalized source family and numeric version for ``*_vN.ext``."""
    filename = os.path.basename(source or "")
    match = _VERSION_SUFFIX.match(filename)
    if not match:
        return None
    return match.group("family").casefold(), int(match.group("version"))


def _prefer_latest_policy_versions(results: list[Any], query: str) -> list[Any]:
    """Select a requested version, preserve comparisons, or default to newest."""
    requested = _REQUESTED_YEAR.search(query or "")
    requested_version = int(requested.group(1)) if requested else None
    if requested_version is None:
        requested = _REQUESTED_VERSION.search(query or "")
        requested_version = int(requested.group(1)) if requested else None
    if requested_version is None and _GENERIC_HISTORY.search(query or ""):
        return list(results)

    available: dict[str, set[int]] = {}
    parsed: list[tuple[Any, tuple[str, int] | None]] = []
    for result in results:
        metadata = getattr(result, "metadata", {}) or {}
        version = _source_version(str(metadata.get("source", "")))
        parsed.append((result, version))
        if version:
            family, number = version
            available.setdefault(family, set()).add(number)

    selected_versions = {
        family: requested_version if requested_version in versions else max(versions)
        for family, versions in available.items()
    }

    return [
        result
        for result, version in parsed
        if version is None or version[1] == selected_versions[version[0]]
    ]


def build_pipeline():
    """Load and prepare all source documents for retrieval."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    build_times: dict[str, float] = {}

    # M1: Chunk each document and retain original parent text for answer contexts.
    started = time.perf_counter()
    print("\n[1/4] Chunking documents...", flush=True)
    documents = load_documents()
    all_chunks: list[dict] = []
    parent_text_by_id: dict[str, str] = {}
    for document_index, document in enumerate(documents):
        document_metadata = dict(document.get("metadata", {}))
        source = str(document_metadata.get("source") or f"document_{document_index}")
        parents, children = chunk_hierarchical(document.get("text", ""), metadata=document_metadata)
        for parent in parents:
            namespaced_id = f"{source}::{parent.parent_id}"
            parent_text_by_id[namespaced_id] = parent.text
        for child in children:
            namespaced_parent_id = f"{source}::{child.parent_id}"
            all_chunks.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "source": source,
                    "parent_id": namespaced_parent_id,
                },
            })
    build_times["chunking"] = time.perf_counter() - started
    print(f"  ✓ {len(all_chunks)} child chunks from {len(documents)} documents ({build_times['chunking']:.1f}s)", flush=True)

    # M5: Enrich child text before indexing, while preserving structural metadata.
    started = time.perf_counter()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [
            {
                "text": item.enriched_text,
                "metadata": {
                    **item.auto_metadata,
                    **chunk["metadata"],
                },
            }
            for item, chunk in zip(enriched, all_chunks)
        ]
        print(f"  ✓ Enriched {len(enriched)} chunks", flush=True)
    else:
        print("  ⚠️  Enrichment returned no chunks; indexing original chunks", flush=True)
    build_times["enrichment"] = time.perf_counter() - started

    # M2: Hybrid BM25 + dense index.
    started = time.perf_counter()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    build_times["indexing"] = time.perf_counter() - started
    print(f"  ✓ Indexed ({build_times['indexing']:.1f}s)", flush=True)

    # M3: Initialize the reranker; its model is loaded on the first query.
    started = time.perf_counter()
    print("\n[4/4] Initializing reranker (model loads on first query)...", flush=True)
    reranker = CrossEncoderReranker()
    build_times["reranker_load"] = time.perf_counter() - started
    print("  ✓ Reranker initialized", flush=True)

    search.parent_text_by_id = parent_text_by_id
    search.latency_breakdown = build_times
    return search, reranker


def _contexts_from_results(results: list[Any], search: HybridSearch) -> list[str]:
    """Resolve retrieved children to original parent text and deduplicate parents."""
    parent_text_by_id = getattr(search, "parent_text_by_id", {}) or {}
    contexts: list[str] = []
    seen_parent_ids: set[str] = set()
    seen_text: set[str] = set()
    for result in results:
        metadata = getattr(result, "metadata", {}) or {}
        parent_id = metadata.get("parent_id")
        context = parent_text_by_id.get(parent_id) if parent_id else None
        if context:
            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)
        else:
            context = getattr(result, "text", "")
        if context and context not in seen_text:
            contexts.append(context)
            seen_text.add(context)
    return contexts


def run_query(
    query: str,
    search: HybridSearch,
    reranker: CrossEncoderReranker,
    timings: dict[str, float] | None = None,
) -> tuple[str, list[str]]:
    """Retrieve and rerank child chunks, then answer from their parent contexts."""
    query_started = time.perf_counter()
    started = time.perf_counter()
    results = search.search(query)
    results = _prefer_latest_policy_versions(results, query)
    retrieval_seconds = time.perf_counter() - started

    candidate_results = []
    seen_candidate_parents: set[str] = set()
    for result in results:
        metadata = getattr(result, "metadata", {}) or {}
        parent_id = metadata.get("parent_id")
        if parent_id and parent_id in seen_candidate_parents:
            continue
        if parent_id:
            seen_candidate_parents.add(parent_id)
        candidate_results.append(result)
    candidates = [
        {"text": result.text, "score": result.score, "metadata": result.metadata}
        for result in candidate_results
    ]
    started = time.perf_counter()
    model_load_before = getattr(reranker, "model_load_seconds", 0.0)
    reranked = reranker.rerank(query, candidates, top_k=RERANK_TOP_K)
    reranker_elapsed = time.perf_counter() - started
    model_load_delta = max(
        0.0,
        getattr(reranker, "model_load_seconds", model_load_before) - model_load_before,
    )
    reranking_seconds = max(0.0, reranker_elapsed - model_load_delta)
    selected = reranked if reranked else candidate_results[:RERANK_TOP_K]
    contexts = _contexts_from_results(selected, search)

    started = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI

            context_str = "\n\n".join(contexts)
            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là trợ lý hỏi đáp quy chế nội bộ, trả lời ngắn gọn bằng tiếng Việt. "
                            "Chỉ dùng dữ kiện được nêu rõ trong context; hãy đối chiếu đúng điều kiện "
                            "của câu hỏi (đặc biệt các mức ngày, ngưỡng tiền và chức danh), không thay "
                            "bằng một điều kiện gần giống. Trả lời đủ mọi ý trong câu hỏi; với câu hỏi "
                            "nhiều phần, nêu riêng từng kết quả. Giữ nguyên số liệu, vai trò và thời hạn "
                            "theo context; không tự suy diễn hay thêm dữ kiện. Nếu thiếu bằng chứng, nói "
                            "'Không tìm thấy thông tin.' Nếu câu hỏi không yêu cầu lịch sử, ưu tiên chính "
                            "sách hiện hành trong context."
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
                temperature=0,
                max_tokens=350,
            )
            answer = response.choices[0].message.content or contexts[0]
        except Exception as exc:
            # Avoid logging exception text, which can contain request details.
            print(f"  ⚠️  LLM generation failed ({type(exc).__name__})", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    answer_seconds = time.perf_counter() - started

    if timings is not None:
        timings.update({
            "retrieval_seconds": retrieval_seconds,
            "reranking_seconds": reranking_seconds,
            "reranker_load_seconds": model_load_delta,
            "answer_generation_seconds": answer_seconds,
            "query_processing_seconds": time.perf_counter() - query_started,
        })
    return answer, contexts


def save_latency_report(report: dict, path: str = "reports/latency_report.json") -> None:
    """Persist stage latency measurements as UTF-8 JSON."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)


def _latency_summary(values: list[float]) -> dict[str, float]:
    total = sum(values)
    return {
        "total_seconds": total,
        "mean_seconds": total / len(values) if values else 0.0,
    }


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run the 20-question evaluation, save score and latency reports."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    per_query_timings: list[dict[str, Any]] = []

    for index, item in enumerate(test_set):
        query_timings: dict[str, float] = {}
        answer, contexts = run_query(item["question"], search, reranker, timings=query_timings)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        per_query_timings.append({"question": item["question"], **query_timings})
        print(f"  [{index + 1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    started = time.perf_counter()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    ragas_seconds = time.perf_counter() - started
    print(f"  ✓ RAGAS done ({ragas_seconds:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        score = results.get(metric, 0)
        print(f"  {'✓' if score >= 0.75 else '✗'} {metric}: {score:.4f}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures)

    build_times = dict(getattr(search, "latency_breakdown", {}) or {})
    build_times["reranker_load"] = float(getattr(reranker, "model_load_seconds", 0.0))
    query_stages = {
        stage: [row.get(key, 0.0) for row in per_query_timings]
        for stage, key in (
            ("retrieval", "retrieval_seconds"),
            ("reranking", "reranking_seconds"),
            ("answer_generation", "answer_generation_seconds"),
            ("query_processing", "query_processing_seconds"),
        )
    }
    stages = {
        stage: _latency_summary([float(seconds)])
        for stage, seconds in build_times.items()
    }
    stages.update({stage: _latency_summary(values) for stage, values in query_stages.items()})
    stages["ragas_evaluation"] = _latency_summary([ragas_seconds])
    save_latency_report({
        "unit": "seconds",
        "num_queries": len(test_set),
        "stages": stages,
        "per_query": per_query_timings,
        "build_stages": build_times,
        "notes": [
            "query_processing is end-to-end and overlaps the retrieval, reranking, and answer_generation subtotals.",
            "reranker_load is reported separately; the first query's end-to-end time includes this cold start.",
        ],
    })
    return results


if __name__ == "__main__":
    started = time.perf_counter()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.perf_counter() - started:.1f}s")
