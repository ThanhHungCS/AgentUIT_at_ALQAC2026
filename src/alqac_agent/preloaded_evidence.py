from __future__ import annotations

import json
from pathlib import Path

from .schemas import CaseEvidence


class PreloadedEvidenceClient:
    """Load all pre-retrieved evidence from a JSONL file; no API calls needed."""

    def __init__(self, jsonl_path: str | Path) -> None:
        self._evidence: dict[str, list[CaseEvidence]] = {}
        self.call_count = 0
        self.query_log: list[dict[str, object]] = []
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Evidence JSONL not found: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                case_id = str(record["case_id"])
                chunks = []
                for chunk in record.get("chunks", []):
                    chunks.append(
                        CaseEvidence(
                            chunk_id=str(chunk["chunk_id"]),
                            text=str(chunk.get("text", "")),
                            score=float(chunk.get("score", 0.0)),
                            query=",".join(chunk.get("source_queries", [])),
                            cached=True,
                        )
                    )
                self._evidence[case_id] = chunks

    def retrieve(self, case_id: str, query: str) -> CaseEvidence | None:
        self.call_count += 1
        chunks = self._evidence.get(case_id, [])
        if not chunks:
            return None
        scored = sorted(chunks, key=lambda c: c.score, reverse=True)
        best = scored[0]
        self.query_log.append(
            {
                "case_id": case_id,
                "query": query,
                "chunk_id": best.chunk_id,
                "score": best.score,
            }
        )
        return CaseEvidence(
            chunk_id=best.chunk_id,
            text=best.text,
            score=best.score,
            query=query,
            cached=True,
        )

    def retrieve_all(self, case_id: str) -> list[CaseEvidence]:
        return list(self._evidence.get(case_id, []))

    def close(self) -> None:
        return None
