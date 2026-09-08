from __future__ import annotations

from typing import Any

from .graph import NON_CIVIL_LAW_IDS
from .ljp_input import build_legal_issue_text
from .retrieval import LawRetriever


def build_law_query(processed_case: dict[str, Any]) -> str:
    return build_legal_issue_text(processed_case)


def retrieve_relevant_laws(
    processed_case: dict[str, Any],
    law_retriever: LawRetriever,
    *,
    top_k: int = 18,
) -> dict[str, Any]:
    law_query = build_law_query(processed_case)
    candidates = law_retriever.search(law_query, top_k)
    filtered = [
        item
        for item in candidates
        if str(item.get("law_id", "")) not in NON_CIVIL_LAW_IDS
    ]
    if len(filtered) >= 6:
        candidates = filtered
    return {
        "law_query": law_query,
        "relevant_laws": candidates,
    }


def compact_laws(
    laws: list[dict[str, Any]],
    *,
    limit: int = 8,
    content_chars: int = 600,
) -> list[dict[str, Any]]:
    return [
        {
            "law_id": str(item.get("law_id", "")),
            "aid": int(item.get("aid", 0)),
            "content": str(item.get("content", ""))[:content_chars],
            "score": float(item.get("score", 0.0)),
        }
        for item in laws[:limit]
    ]
