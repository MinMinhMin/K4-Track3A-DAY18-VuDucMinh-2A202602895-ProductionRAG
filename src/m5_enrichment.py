from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


def _extractive_summary(text: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    return " ".join(sentences[:2]) if sentences else text.strip()


def _topic_from_text(text: str) -> str:
    lowered = text.lower()
    topics = [
        (r"nghỉ\s+phép", "nghỉ phép"),
        (r"mật\s+khẩu|password", "mật khẩu"),
        (r"bảo\s+hiểm", "bảo hiểm"),
        (r"lương|thu nhập", "lương"),
        (r"mua\s+sắm|mua\s+hàng", "mua sắm"),
        (r"tạm\s+ứng|hoàn\s+chi", "tạm ứng và hoàn chi"),
        (r"đào\s+tạo|khóa\s+học", "đào tạo"),
        (r"vpn|mã\s+độc|malware|cntt", "công nghệ thông tin"),
        (r"mentor|buddy", "mentor và buddy"),
    ]
    for pattern, topic in topics:
        if re.search(pattern, lowered):
            return topic
    first_line = next((line.strip("# -*\t ") for line in text.splitlines() if line.strip()), "quy định nội bộ")
    return first_line[:80] or "quy định nội bộ"


def _fallback_metadata(text: str, source: str = "") -> dict:
    lowered = text.lower()
    if any(word in lowered for word in ("lương", "nghỉ phép", "thử việc", "nhân viên", "bảo hiểm")):
        category = "hr"
    elif any(word in lowered for word in ("mật khẩu", "vpn", "cntt", "malware", "máy tính")):
        category = "it"
    elif any(word in lowered for word in ("chi phí", "tạm ứng", "ngân sách", "vnđ", "vnd")):
        category = "finance"
    else:
        category = "policy"
    return {
        "source": source,
        "topic": _topic_from_text(text),
        "entities": [],
        "category": category,
        "language": "vi" if re.search(r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", text, re.IGNORECASE) else "en",
    }


def _fallback_questions(text: str, n_questions: int = 3) -> list[str]:
    if n_questions <= 0 or not text.strip():
        return []
    topic = _topic_from_text(text)
    candidates = [
        f"Theo quy định, {topic} được áp dụng như thế nào?",
        f"Nhân viên cần biết gì về {topic}?",
        f"Những điều kiện hoặc mức áp dụng nào liên quan đến {topic}?",
    ]
    return candidates[:n_questions]


def _fallback_enrichment(text: str, source: str = "") -> dict:
    source_line = f"Trích từ {source}. " if source else ""
    context = f"{source_line}Nội dung thuộc chủ đề {_topic_from_text(text)}.".strip()
    return {
        "summary": _extractive_summary(text),
        "questions": _fallback_questions(text),
        "context": context,
        "metadata": _fallback_metadata(text, source),
    }


def _parse_json_object(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Enrichment response must be a JSON object")
    return value


def _normalize_enrichment(value: dict, text: str, source: str) -> dict:
    fallback = _fallback_enrichment(text, source)
    summary = value.get("summary")
    questions = value.get("questions")
    context = value.get("context")
    metadata = value.get("metadata")

    if isinstance(questions, str):
        questions = [line.strip(" -*\t0123456789.)") for line in questions.splitlines() if line.strip()]
    if not isinstance(questions, list):
        questions = fallback["questions"]
    questions = [str(question).strip() for question in questions if str(question).strip()][:3]
    if not isinstance(metadata, dict):
        metadata = {}
    normalized_metadata = {**fallback["metadata"], **metadata}
    if source:
        normalized_metadata["source"] = source
    return {
        "summary": summary.strip() if isinstance(summary, str) and summary.strip() else fallback["summary"],
        "questions": questions or fallback["questions"],
        "context": context.strip() if isinstance(context, str) and context.strip() else fallback["context"],
        "metadata": normalized_metadata,
    }


def _create_completion(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    from openai import OpenAI

    response = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenAI returned an empty response")
    return content.strip()


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    if OPENAI_API_KEY:
        try:
            return _create_completion(
                "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt, không thêm dữ kiện.",
                text,
                150,
            )
        except Exception as exc:
            print(f"  ⚠️  OpenAI summarize fallback ({type(exc).__name__})", flush=True)
    return _extractive_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0 or not text.strip():
        return []
    if OPENAI_API_KEY:
        try:
            content = _create_completion(
                f"Dựa trên đoạn văn, tạo tối đa {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về JSON array các câu hỏi bằng tiếng Việt.",
                text,
                200,
            )
            value = json.loads(content)
            if isinstance(value, list):
                questions = [str(question).strip() for question in value if str(question).strip()]
                if questions:
                    return questions[:n_questions]
        except Exception as exc:
            print(f"  ⚠️  OpenAI HyQA fallback ({type(exc).__name__})", flush=True)
    return _fallback_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text:
        return ""
    if OPENAI_API_KEY:
        try:
            context = _create_completion(
                "Viết một câu ngắn mô tả vị trí và chủ đề của đoạn văn trong tài liệu. Chỉ dựa trên nội dung đã cho.",
                f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                80,
            )
            return f"{context}\n\n{text}"
        except Exception as exc:
            print(f"  ⚠️  OpenAI contextual fallback ({type(exc).__name__})", flush=True)
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            content = _create_completion(
                'Trích xuất metadata từ đoạn văn. Trả về JSON object có topic, entities (array), category (policy|hr|it|finance), language (vi|en).',
                text,
                150,
            )
            value = _parse_json_object(content)
            if isinstance(value.get("entities", []), list):
                return {**_fallback_metadata(text), **value}
        except Exception as exc:
            print(f"  ⚠️  OpenAI metadata fallback ({type(exc).__name__})", flush=True)
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if not text.strip():
        return _fallback_enrichment(text, source)
    if not OPENAI_API_KEY:
        return _fallback_enrichment(text, source)

    system_prompt = (
        "Phân tích đoạn quy chế tiếng Việt và trả về đúng một JSON object với các trường: "
        '"summary" (2-3 câu ngắn, không thêm dữ kiện), "questions" (tối đa 3 câu hỏi), '
        '"context" (một câu về vị trí/chủ đề trong tài liệu), "metadata" '
        '(object gồm topic, entities array, category policy|hr|it|finance, language vi|en). '
        "Chỉ dùng thông tin trong đoạn văn."
    )
    try:
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        content = response.choices[0].message.content
        return _normalize_enrichment(_parse_json_object(content or ""), text, source)
    except Exception as exc:
        print(f"  ⚠️  Combined enrichment fallback ({type(exc).__name__})", flush=True)
        return _fallback_enrichment(text, source)


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        chunk_metadata = dict(chunk.get("metadata", {}))
        source = chunk_metadata.get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            contextual_text = contextual_prepend(text, source) if "contextual" in methods else ""
            context_line = contextual_text[:-len(text)].strip() if text and contextual_text.endswith(text) else ""
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched_parts = [context_line, summary, *questions, text]
        enriched_text = "\n\n".join(part.strip() for part in enriched_parts if isinstance(part, str) and part.strip())

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk_metadata, **auto_meta, **({"source": source} if source else {})},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
