from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .schemas import VerdictLabel

# Minimal Top-1 probes. They target the operative order and avoid broad party/VKS
# terms that tend to pull argument summaries from the official API.
DISPOSITION_PROBES: tuple[str, ...] = (
    "Vì các lẽ trên QUYẾT ĐỊNH Tuyên xử",
    "Tuyên xử: Chấp nhận Không chấp nhận yêu cầu khởi kiện",
)

SCOPE_TO_LABEL: dict[str, VerdictLabel] = {
    "ALL": "A_WIN",
    "MAJORITY": "PARTIAL_A_WIN",
    "MINORITY": "PARTIAL_B_WIN",
    "NONE": "B_WIN",
}


def disposition_score(text: str) -> int:
    value = text.casefold()
    score = 0
    score += 30 * ("vì các lẽ trên" in value)
    score += 25 * ("quyết định" in value)
    score += 18 * ("tuyên xử" in value)
    score += 15 * ("chấp nhận một phần yêu cầu" in value)
    score += 12 * ("không chấp nhận yêu cầu" in value)
    score += 10 * ("chấp nhận yêu cầu khởi kiện" in value)
    if "đại diện viện kiểm sát" in value and "quyết định" not in value:
        score -= 20
    return score


def select_disposition_evidence(
    evidence: list[dict[str, object]], top_k: int = 4
) -> list[dict[str, object]]:
    ranked = sorted(
        enumerate(evidence),
        key=lambda item: (disposition_score(str(item[1].get("text", ""))), -item[0]),
        reverse=True,
    )
    output: list[dict[str, object]] = []
    for _, item in ranked[:top_k]:
        text = str(item.get("text", ""))
        position = text.casefold().rfind("quyết định")
        if position >= 0:
            text = text[position:]
        output.append({"chunk_id": item["chunk_id"], "text": text})
    return output


def normalize_ascii(text: str) -> str:
    value = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(value.split())


_STRONG_HEADING = re.compile(r"vi cac le tren.{0,80}quyet dinh")
_OTHER_HEADING = re.compile(
    r"quyet dinh\s*:|tuyen xu\s*:|tuyen an\s*:|(?<!xet )\bxu\s*:"
)
_DISPOSITION_PATTERNS: tuple[tuple[VerdictLabel, re.Pattern[str]], ...] = (
    ("PARTIAL_A_WIN", re.compile(r"chap nhan (?:mot|1) phan")),
    (
        "B_WIN",
        re.compile(r"(?:khong chap nhan|bac)(?: toan bo)? (?:cac )?yeu cau"),
    ),
    (
        "A_WIN",
        re.compile(r"(?<!khong )chap nhan(?: toan bo)? (?:cac )?yeu cau"),
    ),
    (
        "B_WIN",
        re.compile(r"(?:khong chap nhan|bac)(?: toan bo)? don khoi kien"),
    ),
    (
        "A_WIN",
        re.compile(r"(?<!khong )chap nhan(?: toan bo)? don khoi kien"),
    ),
)
_NON_COURT_CUES = re.compile(r"de nghi|vien kiem sat|luat su|quan diem")


def explicit_disposition_label(
    evidence: list[dict[str, object]],
) -> VerdictLabel | None:
    """Read the earliest disposition after the strongest operative heading.

    A chunk may repeat the plaintiff's request, the VKS recommendation and the
    court order.  Trimming at the *last* operative heading prevents those earlier
    statements from being mistaken for the judgment.
    """

    best: tuple[float, int, VerdictLabel] | None = None
    for chunk_index, item in enumerate(evidence):
        text = normalize_ascii(str(item.get("text", "")))
        headings = list(_STRONG_HEADING.finditer(text))
        if headings:
            start = headings[-1].end()
            quality = 30
            segment = text[start : start + 1600]
        else:
            headings = list(_OTHER_HEADING.finditer(text))
            if headings:
                start = headings[-1].end()
                quality = 20
                segment = text[start : start + 1600]
            else:
                quality = 0
                segment = text[:1600]

        for label, pattern in _DISPOSITION_PATTERNS:
            for match in pattern.finditer(segment):
                score = quality + 10 - min(match.start() / 200, 8)
                if quality == 0:
                    preceding = segment[max(0, match.start() - 250) : match.start()]
                    if _NON_COURT_CUES.search(preceding):
                        score -= 15
                candidate = (score, -chunk_index, label)
                if best is None or candidate > best:
                    best = candidate
    # A lone party/VKS proposal receives a negative score and is not a court
    # disposition; let the grounded extractor/query prior handle that case.
    return best[2] if best is not None and best[0] > 0 else None


_MONEY_CORE = r"\d{1,3}(?:[\. ]\d{3})+|\d{4,}"
_MONEY = (
    rf"(?:{_MONEY_CORE})(?:,\d+)?\s*"
    r"(?:đồng|vnđ|vnd|đ(?:\b|(?=\W)))"
)
_MONEY_RE = re.compile(rf"(?<!\d)({_MONEY})", re.IGNORECASE)
_PAYMENT_VERB = r"(?:hoàn trả|thanh toán|bồi thường|\btrả\b)"


def _money_value(raw: str) -> int:
    return int(re.sub(r"\D", "", raw.split(",")[0]))


@dataclass(frozen=True, slots=True)
class MoneyRequest:
    requested_amount: int
    incomplete: bool
    alternative: bool
    raw_amounts: tuple[str, ...]
    followup_queries: tuple[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_amount": self.requested_amount,
            "incomplete": self.incomplete,
            "alternative": self.alternative,
            "raw_amounts": list(self.raw_amounts),
            "followup_queries": list(self.followup_queries),
        }


def extract_money_request(case_query: str) -> MoneyRequest | None:
    """Extract a monetary main/alternative claim from the final request clause."""

    lowered = case_query.casefold()
    request_position = lowered.rfind("yêu cầu")
    tail = " ".join(case_query[request_position:].split()) if request_position >= 0 else ""
    payment = re.search(
        rf"{_PAYMENT_VERB}[^.!?]*?{_MONEY}", tail, re.IGNORECASE
    )
    if payment is None:
        return None
    segment = tail[payment.start() :]
    raw_amounts = tuple(match.group(1) for match in _MONEY_RE.finditer(segment))
    if not raw_amounts:
        return None
    totals = [
        _money_value(match.group(1))
        for match in re.finditer(
            rf"tổng(?: cộng)?[^.!?]{{0,50}}?({_MONEY})",
            segment,
            re.IGNORECASE,
        )
    ]
    requested = totals[-1] if totals else sum(_money_value(x) for x in raw_amounts)
    if requested <= 0:
        return None
    incomplete = bool(
        re.search(
            re.escape(raw_amounts[0]) + r"\s*(?:và|,)\s*lãi",
            segment,
            re.IGNORECASE,
        )
    )
    alternative = bool(
        re.search(
            rf"nếu không[^.!?]{{0,100}}{_PAYMENT_VERB}",
            tail,
            re.IGNORECASE,
        )
    )
    amounts_for_query = " ".join(raw_amounts)
    queries = (
        f"QUYẾT ĐỊNH buộc bị đơn trả cho nguyên đơn số tiền {amounts_for_query}",
        f"Tòa án tuyên xử yêu cầu của nguyên đơn số tiền {amounts_for_query}",
    )
    return MoneyRequest(requested, incomplete, alternative, raw_amounts, queries)


_GRANT_PATTERNS: tuple[tuple[int, re.Pattern[str], str], ...] = (
    (
        3,
        re.compile(
            rf"buộc[^.;]{{0,240}}?{_PAYMENT_VERB}[^.;]{{0,140}}?(?P<m>{_MONEY})",
            re.IGNORECASE,
        ),
        "ordered",
    ),
    (
        2,
        re.compile(
            rf"phải\s+(?:có\s+nghĩa\s+vụ\s+)?{_PAYMENT_VERB}"
            rf"[^.;]{{0,140}}?(?P<m>{_MONEY})",
            re.IGNORECASE,
        ),
        "payment",
    ),
    (
        1,
        re.compile(
            rf"\((?P<m>{_MONEY})\s*x\s*5\s*%\s*=\s*{_MONEY}\)\s*"
            r"đối với nghĩa vụ phải thực hiện",
            re.IGNORECASE,
        ),
        "fee-inferred",
    ),
)
_DIRECT_PERCENT = re.compile(
    rf"(?P<g>{_MONEY})\s*/\s*(?P<r>{_MONEY})\s*=\s*[\d,.]+\s*%",
    re.IGNORECASE,
)


def _grant_candidates(
    evidence: list[dict[str, object]],
) -> list[tuple[int, int, str]]:
    output: list[tuple[int, int, str]] = []
    for chunk in evidence:
        text = " ".join(str(chunk.get("text", "")).split())
        for rank, pattern, kind in _GRANT_PATTERNS:
            for match in pattern.finditer(text):
                preceding = text[max(0, match.start() - 80) : match.start()].casefold()
                following = text[match.end() : match.end() + 100].casefold()
                if (
                    kind in {"ordered", "payment"}
                    and "yêu cầu" in preceding[-65:]
                    and re.search(r"không có căn cứ|không chấp nhận", following)
                ):
                    continue
                output.append((rank, _money_value(match.group("m")), kind))
    return output


def apply_money_overlay(
    prediction: VerdictLabel,
    evidence: list[dict[str, object]],
    signal: dict[str, object] | None,
) -> tuple[VerdictLabel, str | None]:
    """Conservatively refine full A/B outcomes using requested vs ordered money."""

    if not signal:
        return prediction, None
    requested = int(signal.get("requested_amount", 0))
    if requested <= 0:
        return prediction, None
    candidates = [
        candidate
        for candidate in _grant_candidates(evidence)
        if 0 < candidate[1] <= 1.01 * requested
    ]
    exact_alternative = any(
        kind in {"ordered", "payment"}
        and abs(amount / requested - 1) <= 0.02
        for _, amount, kind in candidates
    )
    best_grant = max(candidates, key=lambda item: (item[0], item[1]), default=None)

    dynamic_queries = {str(x) for x in signal.get("followup_queries", [])}
    dynamic_text = " ".join(
        str(item.get("text", ""))
        for item in evidence
        if str(item.get("query", "")) in dynamic_queries
    )
    operative_reject = bool(
        re.search(
            r"tuyên\s*xử[\s\S]{0,100}?(?:không chấp nhận|bác)",
            dynamic_text,
            re.IGNORECASE,
        )
    )

    if bool(signal.get("alternative")) and exact_alternative and prediction == "B_WIN":
        return "PARTIAL_B_WIN", "exact alternative monetary relief"
    if operative_reject:
        return "B_WIN", "operative monetary rejection"
    if (
        best_grant is not None
        and not bool(signal.get("incomplete"))
        and prediction == "A_WIN"
        and best_grant[1] / requested < 0.995
    ):
        return "PARTIAL_A_WIN", "requested-versus-ordered monetary relief"

    # Some judgments state the computed ratio directly.  Keep this independent
    # of inferred grant candidates and apply it only to an otherwise full win.
    if prediction == "A_WIN":
        fixed_evidence = [
            item
            for item in evidence
            if str(item.get("query", "")) in DISPOSITION_PROBES
        ]
        for item in fixed_evidence:
            for match in _DIRECT_PERCENT.finditer(str(item.get("text", ""))):
                granted = _money_value(match.group("g"))
                asked = _money_value(match.group("r"))
                if 0 < granted <= asked and granted / asked < 0.995:
                    label: VerdictLabel = (
                        "PARTIAL_A_WIN" if granted / asked > 0.5 else "PARTIAL_B_WIN"
                    )
                    return label, "explicit granted/requested percentage"
    return prediction, None


_AREA_VALUE_PATTERN = (
    r"\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:[.,]\d+)?"
)
_AREA_RE = re.compile(
    rf"(?<!\d)(?P<a>{_AREA_VALUE_PATTERN})\s*m\s*(?:2|²)",
    re.IGNORECASE,
)


def _area_value(raw: str) -> float:
    value = re.sub(r"\s|m\s*(?:2|²)", "", raw, flags=re.IGNORECASE)
    if "." in value and "," in value:
        value = value.replace(".", "").replace(",", ".")
    elif "." in value:
        groups = value.split(".")
        if all(len(group) == 3 for group in groups[1:]):
            value = "".join(groups)
    elif "," in value:
        value = value.replace(",", ".")
    return float(value)


@dataclass(frozen=True, slots=True)
class AreaRequest:
    requested_area: float
    followup_queries: tuple[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_area": self.requested_area,
            "followup_queries": list(self.followup_queries),
        }


def extract_area_request(case_query: str) -> AreaRequest | None:
    """Extract the main requested land area for conditional area retrieval."""

    position = case_query.casefold().rfind("yêu cầu")
    tail = case_query[position:] if position >= 0 else case_query
    match = _AREA_RE.search(tail) or _AREA_RE.search(case_query)
    if match is None:
        return None
    requested = _area_value(match.group("a"))
    if requested <= 0:
        return None
    parcel_ids = re.findall(
        r"thửa(?: đất)?(?: số)?\s*([0-9]+)", case_query, re.IGNORECASE
    )
    parcel_id = parcel_ids[0] if parcel_ids else ""
    parcel_query = (
        "QUYẾT ĐỊNH Chấp nhận yêu cầu khởi kiện đòi lại đất "
        f"Buộc trả lại thửa đất {parcel_id}"
        if parcel_id
        else "QUYẾT ĐỊNH nguyên đơn được quyền quản lý sử dụng diện tích đất"
    )
    queries = (
        "Tòa án tuyên phần diện tích giao cho nguyên đơn",
        parcel_query,
    )
    return AreaRequest(requested, queries)


_AREA_RETURN = re.compile(
    rf"buộc[^.;]{{0,300}}?(?:trả lại|giao trả|trả phần đất)"
    rf"[^.;]{{0,180}}?(?:diện tích(?: đất)?|phần đất)"
    rf"[^.;]{{0,80}}?(?P<a>{_AREA_VALUE_PATTERN})\s*m\s*(?:2|²)",
    re.IGNORECASE,
)
_AREA_SHARE = re.compile(
    rf"chia cho[^.;]{{0,260}}?(?:thửa đất|phần đất)"
    rf"[^.;]{{0,160}}?diện tích[^.;]{{0,80}}?"
    rf"(?P<a>{_AREA_VALUE_PATTERN})\s*m\s*(?:2|²)",
    re.IGNORECASE,
)
_AREA_PLAINTIFF_AWARD = re.compile(
    rf"nguyên đơn[^.;]{{0,150}}?được (?:quyền )?(?:quản lý|sở hữu)"
    rf"[^.;]{{0,180}}?(?:nằm trong )?diện tích(?: đất)?\s*"
    rf"(?P<a>{_AREA_VALUE_PATTERN})\s*m\s*(?:2|²)",
    re.IGNORECASE,
)


def apply_area_overlay(
    prediction: VerdictLabel,
    evidence: list[dict[str, object]],
    signal: dict[str, object] | None,
) -> tuple[VerdictLabel, str | None]:
    """Refine only land outcomes with area explicitly granted in dynamic hits."""

    if not signal:
        return prediction, None
    requested = float(signal.get("requested_area", 0.0))
    if requested <= 0:
        return prediction, None
    dynamic_queries = {str(x) for x in signal.get("followup_queries", [])}
    granted_parcels: set[float] = set()
    strong_plaintiff_awards: set[float] = set()
    for item in evidence:
        if str(item.get("query", "")) not in dynamic_queries:
            continue
        text = " ".join(str(item.get("text", "")).split())
        for pattern in (_AREA_RETURN, _AREA_SHARE, _AREA_PLAINTIFF_AWARD):
            for match in pattern.finditer(text):
                preceding = text[max(0, match.start() - 120) : match.start()]
                if re.search(
                    r"(?:không chấp nhận|bác)[^.;]{0,100}$",
                    preceding,
                    re.IGNORECASE,
                ):
                    continue
                area = _area_value(match.group("a"))
                granted_parcels.add(area)
                if pattern is _AREA_PLAINTIFF_AWARD:
                    strong_plaintiff_awards.add(area)
    granted = sum(granted_parcels)
    strong_granted = sum(strong_plaintiff_awards)
    if (
        prediction == "PARTIAL_A_WIN"
        and 0.995 <= strong_granted / requested <= 1.01
    ):
        return "A_WIN", "substantially full plaintiff land award"
    if not (0 < granted / requested < 0.995):
        return prediction, None
    if prediction == "A_WIN":
        return "PARTIAL_A_WIN", "requested-versus-granted land area"
    if prediction == "PARTIAL_A_WIN" and granted / requested <= 0.5:
        return "PARTIAL_B_WIN", "minority of requested land area"
    return prediction, None


def resolve_prediction(
    *,
    assessment: dict[str, object],
    evidence: list[dict[str, object]],
    prior: dict[str, object],
    confidence_threshold: float,
) -> tuple[VerdictLabel, str]:
    explicit = explicit_disposition_label(evidence)
    scope = str(assessment.get("claim_scope", "UNCLEAR"))
    quality = str(assessment.get("source_quality", "MISSING"))
    confidence = float(assessment.get("confidence", 0.0))
    allowed_ids = {str(item["chunk_id"]) for item in evidence}
    cited_ids = {str(item) for item in assessment.get("decisive_chunk_ids", [])}
    grounded = bool(cited_ids) and cited_ids <= allowed_ids

    if explicit == "PARTIAL_A_WIN" and scope == "MINORITY" and grounded:
        return "PARTIAL_B_WIN", "operative parser + grounded minority assessment"
    if explicit == "PARTIAL_A_WIN" and prior.get("partial_prediction") == "PARTIAL_B_WIN":
        return "PARTIAL_B_WIN", "operative parser + partial query kNN"
    if explicit is not None:
        return explicit, "operative-order parser"

    prediction = str(prior.get("prediction", "PARTIAL_A_WIN"))
    neighbors = prior.get("neighbors", [])
    if neighbors and prediction in {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}:
        return prediction, "BM25 query kNN prior"

    if (
        scope in SCOPE_TO_LABEL
        and quality in {"OPERATIVE_ORDER", "COURT_REASONING"}
        and grounded
        and confidence >= confidence_threshold
    ):
        return SCOPE_TO_LABEL[scope], "grounded disposition assessment"
    return "PARTIAL_A_WIN", "majority fallback"
