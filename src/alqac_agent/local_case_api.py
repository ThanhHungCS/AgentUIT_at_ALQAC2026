from __future__ import annotations

import re
from pathlib import Path

from .data import load_json
from .retrieval import BM25Index
from .schemas import CaseEvidence


def chunk_judgment(
    text: str,
    *,
    window_words: int = 220,
    overlap_words: int = 40,
) -> list[str]:
    """Create deterministic overlapping chunks for the local API simulator."""
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if window_words <= overlap_words:
        raise ValueError("window_words phải lớn hơn overlap_words")
    step = window_words - overlap_words
    return [
        " ".join(words[start : start + window_words])
        for start in range(0, len(words), step)
        if words[start : start + window_words]
    ]


class LocalCaseContentClient:
    """Offline drop-in replacement for the official per-case Top-1 API."""

    def __init__(
        self,
        public_path: str | Path,
        *,
        window_words: int = 220,
        overlap_words: int = 40,
    ) -> None:
        raw = load_json(public_path)
        self._indexes: dict[str, BM25Index] = {}
        self.call_count = 0
        self.query_log: list[dict[str, object]] = []
        for case in raw:
            case_id = str(case["case_id"])
            # judgment_text best approximates the hidden corpus. case_fact is only
            # a fallback, never concatenated as an extra privileged summary.
            text = str(case.get("judgment_text") or case.get("case_fact") or "")
            payloads = [
                {
                    "chunk_id": f"{case_id}_local_chunk_{index:04d}",
                    "text": chunk,
                }
                for index, chunk in enumerate(
                    chunk_judgment(
                        text,
                        window_words=window_words,
                        overlap_words=overlap_words,
                    ),
                    start=1,
                )
            ]
            self._indexes[case_id] = BM25Index(payloads, text_key="text")

    def retrieve(self, case_id: str, query: str) -> CaseEvidence | None:
        if case_id not in self._indexes:
            raise KeyError(
                f"{case_id} không có trong local public corpus; "
                "backend local-public chỉ dùng được cho public cases"
            )
        self.call_count += 1
        results = self._indexes[case_id].search(query, 1)
        if not results:
            self.query_log.append(
                {"case_id": case_id, "query": query, "chunk_id": None, "score": 0.0}
            )
            return None
        hit = results[0]
        evidence = CaseEvidence(
            chunk_id=str(hit["chunk_id"]),
            text=str(hit["text"]),
            score=float(hit["score"]),
            query=query,
            cached=True,
        )
        self.query_log.append(
            {
                "case_id": case_id,
                "query": query,
                "chunk_id": evidence.chunk_id,
                "score": evidence.score,
            }
        )
        return evidence

    def close(self) -> None:
        return None
