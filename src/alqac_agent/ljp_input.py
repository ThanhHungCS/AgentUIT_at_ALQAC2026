from __future__ import annotations

import re
from typing import Any

from .outcome import normalize_ascii


SECTION_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("plaintiff_claim", "plaintiff", "nguyên đơn"),
    ("plaintiff_claim", "plaintiff", "đơn khởi kiện"),
    ("defendant_argument", "defendant", "bị đơn"),
    ("related_party_statement", "related_party", "người có quyền lợi"),
    ("related_party_statement", "related_party", "người làm chứng"),
    ("procuracy_opinion", "procuracy", "đại diện viện kiểm sát"),
    ("procuracy_opinion", "procuracy", "kiểm sát viên"),
    ("court_reasoning", "court", "hội đồng xét xử"),
    ("court_disposition", "court", "vì các lẽ trên"),
    ("court_disposition", "court", "quyết định"),
    ("court_disposition", "court", "tuyên xử"),
)

DISPOSITION_CUES: tuple[str, ...] = (
    "vì các lẽ trên",
    "quyết định",
    "tuyên xử",
    "hội đồng xét xử",
    "ý kiến về việc giải quyết vụ án",
    "quan điểm giải quyết vụ án",
    "đề nghị hội đồng xét xử",
    "đề nghị hđxx",
    "chấp nhận một phần",
    "chấp nhận yêu cầu",
    "không chấp nhận yêu cầu",
    "bác yêu cầu",
    "bác toàn bộ",
    "rút yêu cầu",
    "rút toàn bộ",
)

LEGAL_ISSUE_TERMS = re.compile(
    r"\b(?:tranh chấp|hợp đồng|bồi thường|thiệt hại|quyền sử dụng đất|"
    r"thừa kế|vay|tín dụng|chuyển nhượng|mua bán|đòi tài sản|hụi|"
    r"công nợ|ly hôn|phản tố|yêu cầu độc lập|di sản|lối đi|ranh giới)\b",
    re.IGNORECASE,
)
MONEY_PATTERN = re.compile(
    r"\d{1,3}(?:[\. ]\d{3})+(?:,\d+)?\s*(?:đồng|vnđ|vnd|đ\b)",
    re.IGNORECASE,
)
AREA_PATTERN = re.compile(r"\d+(?:[.,]\d+)?\s*m\s*(?:2|²)", re.IGNORECASE)


def normalize_case_fact(text: str) -> str:
    return " ".join(text.split())


def _marker_hits(text: str) -> list[tuple[int, str, str, str]]:
    lowered = text.casefold()
    hits: list[tuple[int, str, str, str]] = []
    for section, speaker, marker in SECTION_MARKERS:
        start = lowered.find(marker)
        while start >= 0:
            hits.append((start, section, speaker, marker))
            start = lowered.find(marker, start + len(marker))
    hits.sort(key=lambda item: item[0])
    return hits


def segment_case_fact(case_fact: str) -> list[dict[str, Any]]:
    text = normalize_case_fact(case_fact)
    hits = _marker_hits(text)
    if not hits:
        return [
            {
                "segment_id": "seg_001",
                "section": "unknown",
                "speaker": "unknown",
                "marker": "",
                "text": text,
            }
        ]
    segments: list[dict[str, Any]] = []
    if hits[0][0] > 0:
        segments.append(
            {
                "segment_id": "seg_001",
                "section": "background",
                "speaker": "unknown",
                "marker": "",
                "text": text[: hits[0][0]].strip(),
            }
        )
    for index, (start, section, speaker, marker) in enumerate(hits, start=1):
        end = hits[index][0] if index < len(hits) else len(text)
        segment_text = text[start:end].strip()
        if not segment_text:
            continue
        segments.append(
            {
                "segment_id": f"seg_{len(segments) + 1:03d}",
                "section": section,
                "speaker": speaker,
                "marker": marker,
                "text": segment_text,
            }
        )
    return segments


def _window(text: str, position: int, *, before: int = 900, after: int = 2200) -> str:
    return text[max(0, position - before) : position + after].strip()


def _segment_score(segment: dict[str, Any]) -> int:
    normalized = normalize_ascii(str(segment.get("text", "")))
    section = str(segment.get("section", ""))
    speaker = str(segment.get("speaker", ""))
    score = 0
    score += 60 * (section == "court_disposition")
    score += 42 * (section == "court_reasoning")
    score += 28 * (section == "procuracy_opinion")
    score += 20 * (speaker == "court")
    score += 16 * (speaker == "procuracy")
    score -= 18 * (
        speaker in {"plaintiff", "defendant"} and "hoi dong xet xu" not in normalized
    )
    score += 24 * ("chap nhan mot phan" in normalized)
    score += 22 * ("khong chap nhan yeu cau" in normalized)
    score += 20 * ("bac yeu cau" in normalized)
    score += 18 * ("chap nhan yeu cau" in normalized)
    score += 16 * ("rut yeu cau" in normalized)
    score += 14 * ("de nghi hoi dong xet xu" in normalized)
    score += 10 * ("quan diem giai quyet vu an" in normalized)
    return score


def extract_main_claim(case_fact: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    text = normalize_case_fact(case_fact)
    candidate_text = " ".join(
        str(segment["text"])
        for segment in segments
        if segment.get("section") in {"plaintiff_claim", "background"}
    )
    if not candidate_text:
        candidate_text = text
    request_position = candidate_text.casefold().rfind("yêu cầu")
    claim = candidate_text[request_position:] if request_position >= 0 else candidate_text[:1600]
    claim = claim[:2200]
    return {
        "main_claim": claim,
        "requested_amounts": MONEY_PATTERN.findall(claim),
        "requested_areas": AREA_PATTERN.findall(claim),
    }


def extract_disposition_candidates(
    case_fact: str,
    segments: list[dict[str, Any]],
    *,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    text = normalize_case_fact(case_fact)
    lowered = text.casefold()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(source: dict[str, Any], snippet: str, score: int) -> None:
        key = normalize_ascii(snippet)[:260]
        if not snippet or key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "chunk_id": f"cf_{len(candidates) + 1:03d}",
                "segment_id": source.get("segment_id", ""),
                "section": source.get("section", "unknown"),
                "speaker": source.get("speaker", "unknown"),
                "query": source.get("marker", ""),
                "text": snippet,
                "score": score,
            }
        )

    for cue in DISPOSITION_CUES:
        start = lowered.find(cue)
        while start >= 0:
            add(
                {
                    "segment_id": "cue",
                    "section": "cue_window",
                    "speaker": "unknown",
                    "marker": cue,
                },
                _window(text, start),
                40,
            )
            start = lowered.find(cue, start + len(cue))
    for segment in segments:
        score = _segment_score(segment)
        if score > 0:
            add(segment, str(segment["text"])[:3200], score)
    candidates.sort(key=lambda item: int(item.get("score", 0)), reverse=True)
    return candidates[:max_candidates]


def build_legal_issue_text(processed_case: dict[str, Any]) -> str:
    main_claim = str(processed_case.get("main_claim", {}).get("main_claim", ""))
    segments = processed_case.get("segments", [])
    issue_sentences: list[str] = []
    for segment in segments:
        text = str(segment.get("text", ""))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if LEGAL_ISSUE_TERMS.search(sentence):
                issue_sentences.append(sentence)
            if len(issue_sentences) >= 10:
                break
        if len(issue_sentences) >= 10:
            break
    law_refs = re.findall(
        r"Điều\s+\d+[^\.;,]{0,80}",
        str(processed_case.get("case_fact", "")),
        flags=re.IGNORECASE,
    )
    return normalize_case_fact(" ".join([main_claim, *issue_sentences, *law_refs[:12]]))[:6500]


def process_case_fact(
    case_id: str,
    case_fact: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_case_fact(case_fact)
    segments = segment_case_fact(normalized)
    main_claim = extract_main_claim(normalized, segments)
    candidates = extract_disposition_candidates(normalized, segments)
    return {
        "case_id": case_id,
        "case_fact": normalized,
        "metadata": metadata or {},
        "segments": segments,
        "main_claim": main_claim,
        "disposition_candidates": candidates,
    }
