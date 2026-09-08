from __future__ import annotations

import re
from typing import Any, Protocol

from .ljp_retrieval import compact_laws
from .outcome import (
    SCOPE_TO_LABEL,
    apply_area_overlay,
    apply_money_overlay,
    extract_area_request,
    extract_money_request,
    normalize_ascii,
)
from .schemas import DispositionAssessment, JudgeDecision, VerdictLabel


class ZeroShotJudge(Protocol):
    def extract_disposition(
        self,
        case_query: str,
        evidence: list[dict[str, object]],
        laws: list[dict[str, object]] | None = None,
    ) -> DispositionAssessment:
        ...

    def judge(
        self,
        case_query: str,
        evidence: list[dict[str, object]],
        laws: list[dict[str, object]],
        examples: list[dict[str, object]],
        advocate_a: dict[str, object],
        advocate_b: dict[str, object],
    ) -> JudgeDecision:
        ...


LABEL_PATTERNS: tuple[tuple[VerdictLabel, re.Pattern[str], int], ...] = (
    (
        "PARTIAL_A_WIN",
        re.compile(r"chap nhan (?:mot|1) phan (?:yeu cau|don khoi kien)"),
        34,
    ),
    (
        "B_WIN",
        re.compile(
            r"(?:khong chap nhan|bac)(?: toan bo)? "
            r"(?:cac )?(?:yeu cau|don khoi kien)(?: cua nguyen don)?"
        ),
        36,
    ),
    (
        "A_WIN",
        re.compile(
            r"(?<!khong )chap nhan(?: toan bo)? "
            r"(?:cac )?(?:yeu cau|don khoi kien)(?: cua nguyen don)?"
        ),
        30,
    ),
)


def _source_weight(candidate: dict[str, Any]) -> int:
    section = str(candidate.get("section", ""))
    speaker = str(candidate.get("speaker", ""))
    text = normalize_ascii(str(candidate.get("text", "")))
    weight = 0
    weight += 55 * (section == "court_disposition")
    weight += 42 * (section == "court_reasoning")
    weight += 28 * (section == "procuracy_opinion")
    weight += 18 * (speaker == "court")
    weight += 12 * (speaker == "procuracy")
    weight += 12 * ("hoi dong xet xu" in text)
    weight += 10 * ("kiem sat vien" in text and "de nghi" in text)
    weight -= 25 * ("luat su bao ve" in text)
    weight -= 18 * (speaker in {"plaintiff", "defendant", "related_party"})
    return weight


def _assessment_for_label(
    label: VerdictLabel,
    candidate: dict[str, Any],
    score: int,
    phrase: str,
) -> dict[str, Any]:
    scope = {
        "A_WIN": "ALL",
        "PARTIAL_A_WIN": "MAJORITY",
        "PARTIAL_B_WIN": "MINORITY",
        "B_WIN": "NONE",
    }[label]
    confidence = 0.9 if score >= 80 else 0.76 if score >= 55 else 0.6
    return {
        "source_quality": "OPERATIVE_ORDER" if score >= 80 else "COURT_REASONING",
        "court_wording": (
            "PARTIAL" if label.startswith("PARTIAL") else "ALL" if label == "A_WIN" else "NONE"
        ),
        "claim_scope": scope,
        "confidence": confidence,
        "requested_relief": "",
        "granted_relief": phrase,
        "rejected_relief": "",
        "decisive_chunk_ids": [str(candidate.get("chunk_id", ""))],
        "followup_queries": [],
    }


def heuristic_disposition(processed_case: dict[str, Any]) -> dict[str, Any]:
    best: tuple[int, VerdictLabel, dict[str, Any], str] | None = None
    for rank, candidate in enumerate(processed_case.get("disposition_candidates", [])):
        text = normalize_ascii(str(candidate.get("text", "")))
        source_score = _source_weight(candidate)
        for label, pattern, pattern_score in LABEL_PATTERNS:
            for match in pattern.finditer(text):
                preceding = text[max(0, match.start() - 240) : match.start()]
                if (
                    source_score < 20
                    and re.search(r"(?:bi don|luat su|nguoi bao ve)[^.;]{0,200}$", preceding)
                ):
                    continue
                score = source_score + pattern_score - min(match.start() // 180, 10) - rank
                item = (score, label, candidate, match.group(0))
                if best is None or item[0] > best[0]:
                    best = item
    if best is None:
        return {
            "source_quality": "MISSING",
            "court_wording": "NOT_EXPLICIT",
            "claim_scope": "UNCLEAR",
            "confidence": 0.0,
            "requested_relief": str(processed_case.get("main_claim", {}).get("main_claim", ""))[:600],
            "granted_relief": "",
            "rejected_relief": "",
            "decisive_chunk_ids": [],
            "followup_queries": [],
        }
    score, label, candidate, phrase = best
    return _assessment_for_label(label, candidate, score, phrase)


def _label_from_scope(scope: object) -> VerdictLabel | None:
    value = str(scope)
    if value in SCOPE_TO_LABEL:
        return SCOPE_TO_LABEL[value]
    return None


def _safe_confidence(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def llm_disposition(
    processed_case: dict[str, Any],
    relevant_laws: list[dict[str, Any]],
    judge: ZeroShotJudge | None,
) -> dict[str, Any] | None:
    if judge is None:
        return None
    try:
        evidence = [
            {
                "chunk_id": str(item.get("chunk_id", "")),
                "text": (
                    f"section={item.get('section', 'unknown')}; "
                    f"speaker={item.get('speaker', 'unknown')}; "
                    f"text={str(item.get('text', ''))[:1400]}"
                ),
            }
            for item in processed_case.get("disposition_candidates", [])[:7]
        ]
        prompt = (
            "Zero-shot legal judgment prediction. Không dùng ví dụ có nhãn. "
            "Hãy trích xuất phạm vi yêu cầu chính của nguyên đơn được chấp nhận "
            "từ các đoạn case_fact đã xử lý."
        )
        return judge.extract_disposition(
            prompt,
            evidence,
            compact_laws(relevant_laws, limit=6),
        ).model_dump()
    except Exception as error:
        print(
            f"[warn] zero-shot disposition LLM fallback ({type(error).__name__}): {error}",
            flush=True,
        )
        return None


def resolve_judgment(
    processed_case: dict[str, Any],
    relevant_laws: list[dict[str, Any]],
    *,
    judge: ZeroShotJudge | None = None,
) -> dict[str, Any]:
    heuristic = heuristic_disposition(processed_case)
    model_assessment = llm_disposition(processed_case, relevant_laws, judge)
    assessment = heuristic
    if model_assessment and _safe_confidence(model_assessment.get("confidence")) >= _safe_confidence(
        heuristic.get("confidence")
    ):
        assessment = model_assessment

    prediction = _label_from_scope(assessment.get("claim_scope")) or "PARTIAL_A_WIN"
    evidence = list(processed_case.get("disposition_candidates", []))
    money_signal = extract_money_request(str(processed_case.get("case_fact", "")))
    prediction, money_source = apply_money_overlay(
        prediction,
        evidence,
        money_signal.as_dict() if money_signal else None,
    )
    area_signal = extract_area_request(str(processed_case.get("case_fact", "")))
    prediction, area_source = apply_area_overlay(
        prediction,
        evidence,
        area_signal.as_dict() if area_signal else None,
    )

    source_parts = [str(assessment.get("source_quality", "UNKNOWN"))]
    if money_source:
        source_parts.append(money_source)
    if area_source:
        source_parts.append(area_source)
    return {
        "case_id": processed_case["case_id"],
        "prediction": prediction,
        "trace": {
            "claim_scope": assessment.get("claim_scope", "UNCLEAR"),
            "confidence": assessment.get("confidence", 0.0),
            "source": " + ".join(source_parts),
            "main_claim": processed_case.get("main_claim", {}),
            "disposition_candidates": [
                {
                    "chunk_id": str(item.get("chunk_id", "")),
                    "section": str(item.get("section", "")),
                    "speaker": str(item.get("speaker", "")),
                    "text": str(item.get("text", ""))[:1200],
                    "score": item.get("score", 0.0),
                }
                for item in processed_case.get("disposition_candidates", [])[:6]
            ],
            "disposition": assessment,
            "law_query": processed_case.get("law_query", ""),
            "relevant_laws": compact_laws(relevant_laws, limit=8),
        },
    }
