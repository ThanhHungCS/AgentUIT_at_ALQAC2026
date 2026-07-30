from __future__ import annotations

import re


_AGENT_PROMPT_RE = re.compile(r"\bAgent\s+dự\s+đoán\b.*$", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean(text: str) -> str:
    return " ".join(text.split())


def _trim(text: str, limit: int) -> str:
    text = _clean(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut if cut else text[:limit]


def _sentences(case_query: str) -> list[str]:
    body = _AGENT_PROMPT_RE.sub("", case_query)
    return [_clean(item) for item in _SENTENCE_SPLIT_RE.split(body) if _clean(item)]


def build_fact_queries(case_query: str, *, max_queries: int = 1) -> list[str]:
    """Create compact fact-retrieval queries from the private-safe case_query.

    Public `case_fact` usually starts with court-process wording such as
    "Tại đơn khởi kiện..." and then restates the dispute and plaintiff's relief.
    These queries intentionally combine that boilerplate with the concrete
    parties/claim extracted from `case_query`, so the same generator can be used
    on private cases without reading public-only fact text.
    """

    sentences = _sentences(case_query)
    if not sentences:
        return []

    dispute = next(
        (
            sentence
            for sentence in sentences
            if re.search(r"khởi kiện|tranh chấp|yêu cầu", sentence, re.IGNORECASE)
        ),
        sentences[0],
    )
    relief = next(
        (
            sentence
            for sentence in reversed(sentences)
            if re.search(r"\byêu cầu\b|đề nghị|buộc|công nhận|hủy", sentence, re.IGNORECASE)
        ),
        "",
    )

    queries: list[str] = []
    primary = _trim(
        "Tại đơn khởi kiện quá trình giải quyết nguyên đơn trình bày "
        + dispute,
        260,
    )
    if primary:
        queries.append(primary)

    if relief and relief.casefold() != dispute.casefold():
        secondary = _trim(
            "Nguyên đơn yêu cầu Tòa án " + relief,
            240,
        )
        queries.append(secondary)

    output: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.casefold()
        if key not in seen:
            seen.add(key)
            output.append(query)
    return output[:max_queries]
