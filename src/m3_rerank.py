from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
import math
from functools import lru_cache
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@lru_cache(maxsize=2)
def _load_cross_encoder_model(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self.model_load_seconds = 0.0

    def _load_model(self):
        if self._model is None:
            started = time.perf_counter()
            try:
                self._model = _load_cross_encoder_model(self.model_name)
            finally:
                self.model_load_seconds += time.perf_counter() - started
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        candidates = [doc for doc in documents if isinstance(doc.get("text"), str) and doc["text"].strip()]
        if not candidates:
            return []
        try:
            model = self._load_model()
            pairs = [(query, document["text"]) for document in candidates]
            raw_scores = model.predict(pairs)
            if isinstance(raw_scores, (int, float)):
                scores = [float(raw_scores)]
            else:
                import numpy as np

                scores = np.asarray(raw_scores).reshape(-1).astype(float).tolist()
            if len(scores) != len(candidates):
                raise ValueError("CrossEncoder returned a different number of scores than documents")
        except Exception as exc:
            print(f"  ⚠️  CrossEncoder reranking unavailable: {exc}", flush=True)
            return []

        scored = []
        for score, document in zip(scores, candidates):
            if math.isfinite(score):
                scored.append((score, document))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RerankResult(
                text=document["text"],
                original_score=float(document.get("score", 0.0) or 0.0),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # Optional FlashRank adapter remains inactive unless installed and configured.
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
