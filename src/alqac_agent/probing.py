from __future__ import annotations

import re
from pathlib import Path

from .case_api import CaseContentClient
from .config import Settings
from .data import load_json, write_json_atomic
from .fact_queries import build_fact_queries
from .local_case_api import LocalCaseContentClient
from .outcome import explicit_disposition_label, normalize_ascii
from .retrieval import tokenize


HEADING_RE = re.compile(
    r"vi cac le tren.{0,160}quyet dinh|quyet dinh\s*:|tuyen xu\s*:|"
    r"tuyen an\s*:|(?<!xet )\bxu\s*:"
)
VKS_RE = re.compile(r"vien kiem sat|de nghi hoi dong xet xu|quan diem giai quyet")
DISPOSITION_RE = re.compile(
    r"chap nhan(?: mot phan)?|khong chap nhan|bac yeu cau|buoc .* tra"
)


def _decision_segment(public_case: dict[str, object]) -> str:
    text = str(public_case.get("judgment_text") or "")
    normalized = normalize_ascii(text)
    matches = list(HEADING_RE.finditer(normalized))
    if not matches:
        return ""
    start = matches[-1].start()
    return normalized[start : start + 2200]


def _jaccard(left: str, right: str) -> float:
    a = set(tokenize(left))
    b = set(tokenize(right))
    return len(a & b) / len(a | b) if a and b else 0.0


def load_probe_queries(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    seen: set[str] = set()
    for line in lines:
        query = line.strip()
        if not query or query.startswith("#"):
            continue
        key = query.casefold()
        if key not in seen:
            seen.add(key)
            output.append(query)
    return output


def probe_evidence_queries(
    *,
    settings: Settings,
    public_path: str | Path,
    queries_path: str | Path,
    backend: str,
    output_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    public_cases = load_json(public_path)
    if limit is not None:
        public_cases = public_cases[:limit]
    queries = load_probe_queries(queries_path)
    if backend == "local-public":
        client = LocalCaseContentClient(public_path)
    elif backend == "official":
        client = CaseContentClient(
            url=settings.api_url,
            api_key=settings.api_key,
            cache_path=settings.cache_path,
            interval_seconds=settings.api_interval_seconds,
            timeout_seconds=settings.api_timeout_seconds,
        )
    else:
        raise ValueError(f"backend không hợp lệ: {backend}")

    rows: list[dict[str, object]] = []
    try:
        for query in queries:
            hits = headings = dispositions = vks = explicit = 0
            similarities: list[float] = []
            samples: list[dict[str, object]] = []
            for case in public_cases:
                case_id = str(case["case_id"])
                hit = client.retrieve(case_id, query)
                if hit is None:
                    continue
                hits += 1
                text = normalize_ascii(hit.text)
                has_heading = bool(HEADING_RE.search(text))
                has_disposition = bool(DISPOSITION_RE.search(text))
                has_vks = bool(VKS_RE.search(text))
                label = explicit_disposition_label([hit.model_dump()])
                headings += int(has_heading)
                dispositions += int(has_disposition)
                vks += int(has_vks)
                explicit += int(label is not None)
                segment = _decision_segment(case)
                similarity = _jaccard(text, segment)
                similarities.append(similarity)
                if len(samples) < 3 and (has_heading or label is not None):
                    samples.append(
                        {
                            "case_id": case_id,
                            "chunk_id": hit.chunk_id,
                            "label": label,
                            "similarity": round(similarity, 4),
                            "text": " ".join(hit.text.split())[:600],
                        }
                    )
            count = len(public_cases) or 1
            rows.append(
                {
                    "query": query,
                    "cases": len(public_cases),
                    "hits": hits,
                    "heading_rate": round(headings / count, 4),
                    "disposition_term_rate": round(dispositions / count, 4),
                    "vks_rate": round(vks / count, 4),
                    "explicit_label_rate": round(explicit / count, 4),
                    "avg_decision_overlap": round(
                        sum(similarities) / len(similarities), 4
                    )
                    if similarities
                    else 0.0,
                    "samples": samples,
                }
            )
    finally:
        client.close()

    rows.sort(
        key=lambda item: (
            float(item["explicit_label_rate"]),
            float(item["heading_rate"]),
            float(item["avg_decision_overlap"]),
            -float(item["vks_rate"]),
        ),
        reverse=True,
    )
    report = {"backend": backend, "queries": rows}
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report


def derive_fact_query_report(
    *,
    public_path: str | Path,
    output_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    public_cases = load_json(public_path)
    if limit is not None:
        public_cases = public_cases[:limit]

    rows: list[dict[str, object]] = []
    overlaps: list[float] = []
    for case in public_cases:
        queries = build_fact_queries(str(case["case_query"]), max_queries=2)
        fact = normalize_ascii(str(case.get("case_fact") or ""))
        query_text = normalize_ascii(" ".join(queries))
        overlap = _jaccard(query_text, fact)
        overlaps.append(overlap)
        rows.append(
            {
                "case_id": case["case_id"],
                "queries": queries,
                "case_fact_overlap": round(overlap, 4),
                "case_fact_preview": " ".join(str(case.get("case_fact") or "").split())[:600],
            }
        )

    report = {
        "cases": len(rows),
        "avg_case_fact_overlap": round(sum(overlaps) / len(overlaps), 4)
        if overlaps
        else 0.0,
        "items": rows,
    }
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report
