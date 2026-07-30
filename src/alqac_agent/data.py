from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import CaseInput


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_cases(path: str | Path) -> list[CaseInput]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: root phải là JSON array")
    cases = [CaseInput.model_validate(item) for item in raw]
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: case_id bị trùng")
    return cases


def load_law_articles(path: str | Path) -> list[dict[str, object]]:
    laws = load_json(path)
    if not isinstance(laws, list):
        raise ValueError(f"{path}: root phải là JSON array")
    articles: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for law in laws:
        law_id = str(law["law_id"])
        for article in law["content"]:
            key = (law_id, int(article["aid"]))
            if key in seen:
                raise ValueError(f"Trùng law evidence: {key}")
            seen.add(key)
            articles.append(
                {
                    "law_id": law_id,
                    "aid": key[1],
                    "content": str(article["content_Article"]),
                }
            )
    return articles


def load_public_examples(path: str | Path) -> list[dict[str, object]]:
    raw = load_json(path)
    required = {"case_id", "case_query", "verdict_label"}
    return [item for item in raw if isinstance(item, dict) and required <= item.keys()]


def write_json_atomic(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(destination)

