from __future__ import annotations

"""Extract important keywords directly from ALQAC case_query text with a local LLM.

This module intentionally does not read or match against the keyword corpora.
The extracted query keywords are meant to be combined with corpus keywords in a
later query-design step.
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from llama_cpp import Llama

from llm_case_type_classifier import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    find_case_query,
    load_cases,
    load_llm,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTRUCTION = ROOT / "prompts" / "case_query_keyword_extractor_instruction.txt"
DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_DIR.parent / "llm_case_query_keywords"
DEFAULT_PUBLIC_BATCH_OUTPUT = DEFAULT_OUTPUT_ROOT / "public_llm_case_query_keywords.json"
DEFAULT_PRIVATE_BATCH_OUTPUT = DEFAULT_OUTPUT_ROOT / "private_llm_case_query_keywords.json"

VALID_KEYWORD_ID_PREFIXES = {
    "name",
    "noun",
    "verb",
    "adjective",
    "location",
    "number",
}
BASIC_PREFIX_BY_OLD_TYPE = {
    "party_or_org": "name",
    "dispute_relation": "noun",
    "claim_or_request": "verb",
    "property_or_object": "noun",
    "land_detail": "noun",
    "money_amount": "number",
    "date_time": "number",
    "document_or_evidence": "noun",
    "location": "location",
    "distinctive_fact": "noun",
}
BANNED_NOISE_PHRASES = {
    "agent du doan",
    "du doan",
    "theo ban",
    "nguyen don thang",
    "bi don thang",
    "thang kien",
    "kha nang thang kien",
    "nguyen don hay bi don",
    "nguyen don thang kien",
    "bi don thang kien",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def build_user_prompt(case_query: str) -> str:
    return "case_query:\n" + case_query


def extract_json_object(text: str) -> dict[str, Any]:
    text = strip_code_fences(text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model output does not contain a JSON object: {text}")
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    repaired = repair_common_json_errors(candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        loose_keywords = extract_loose_keyword_objects(candidate)
        if loose_keywords:
            return {"keywords": loose_keywords}
        raise


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def repair_common_json_errors(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"}\s*{", "},{", repaired)
    repaired = re.sub(r"]\s*\"", "],\"", repaired)
    repaired = re.sub(r'"\s*\n\s*"', '",\n"', repaired)
    return repaired


def extract_json_string_field(text: str, field: str) -> str | None:
    pattern = re.compile(rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"', flags=re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1)


def extract_loose_keyword_objects(text: str) -> list[dict[str, Any]]:
    """Recover keyword items when the model forgets commas in the JSON array."""

    keyword_blocks = re.findall(r"\{[^{}]*?\"keyword_text\"[^{}]*?\}", text, flags=re.DOTALL)
    keywords: list[dict[str, Any]] = []
    for block in keyword_blocks:
        repaired_block = repair_common_json_errors(block)
        try:
            item = json.loads(repaired_block)
        except json.JSONDecodeError:
            keyword_text = extract_json_string_field(block, "keyword_text")
            if not keyword_text:
                continue
            item = {
                "keyword_id": extract_json_string_field(block, "keyword_id")
                or extract_json_string_field(block, "id")
                or "",
                "type": extract_json_string_field(block, "type") or "",
                "keyword_text": keyword_text,
            }
        if isinstance(item, dict) and item.get("keyword_text"):
            keywords.append(item)
    return keywords


def normalize_keyword_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" ,.;:()[]{}\"'")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    no_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return no_marks.replace("\u0111", "d").replace("\u0110", "D")
    return no_marks.replace("đ", "d").replace("Đ", "D")


def is_noise_keyword(text: str) -> bool:
    normalized = strip_accents(text).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return any(phrase in normalized for phrase in BANNED_NOISE_PHRASES)


def keyword_prefix(keyword_id: str) -> str | None:
    match = re.match(r"^([a-z_]+)\d+$", str(keyword_id or "").strip())
    if not match:
        return None
    prefix = match.group(1)
    if prefix == "date":
        return "number"
    return prefix if prefix in VALID_KEYWORD_ID_PREFIXES else None


def infer_prefix_from_text(text: str, current_prefix: str) -> str:
    normalized = strip_accents(text).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if re.search(r"^(ong|ba|anh|chi|cu)\s+", normalized):
        return "name"
    if re.search(r"^(ngan hang|quy tin dung|cong ty|ubnd|uy ban nhan dan|benh vien)\s+", normalized):
        return "name"
    if re.search(r"\d", normalized):
        return "number"
    if re.search(r"\b(tinh|thanh pho|tp\.?|huyen|quan|xa|phuong|thi tran|ap|thon)\b", normalized):
        return "location"
    return current_prefix


def validate_keywords(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_keywords = payload.get("keywords")
    if not isinstance(raw_keywords, list):
        raise ValueError(f"Model output must contain a keywords list: {payload}")

    keywords: list[dict[str, Any]] = []
    seen = set()
    prefix_counts: Counter[str] = Counter()
    for item in raw_keywords:
        if not isinstance(item, dict):
            continue
        text = normalize_keyword_text(item.get("keyword_text", ""))
        if not text:
            continue
        if is_noise_keyword(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)

        raw_keyword_id = item.get("keyword_id") or item.get("id")
        prefix = keyword_prefix(raw_keyword_id)
        if prefix is None and item.get("type") == "date":
            prefix = "number"
        if prefix is None:
            prefix = BASIC_PREFIX_BY_OLD_TYPE.get(item.get("keyword_type"), "noun")
        prefix = infer_prefix_from_text(text, prefix)
        prefix_counts[prefix] += 1

        keywords.append(
            {
                "keyword_id": f"{prefix}{prefix_counts[prefix]}",
                "type": prefix,
                "keyword_text": text,
            }
        )
    return keywords


def extract_keywords_from_query(
    llm: Llama,
    case_query: str,
    instruction: str,
    max_tokens: int = 512,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": build_user_prompt(case_query)},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
    )
    elapsed_seconds = time.perf_counter() - started_at
    content = response["choices"][0]["message"]["content"]
    try:
        parsed = extract_json_object(content)
    except Exception as exc:
        raw_preview = content[:2500].replace("\n", "\\n")
        raise ValueError(f"Cannot parse model JSON output: {exc}. raw_output_preview={raw_preview}") from exc
    keywords = validate_keywords(parsed)
    return {
        "keywords": keywords,
        "keyword_count": len(keywords),
        "keyword_id_prefix_counts": dict(
            Counter(item.get("type") or keyword_prefix(item["keyword_id"]) or "unknown" for item in keywords).most_common()
        ),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "raw_output": content,
    }


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["case_id"]: row
        for row in rows
        if row.get("case_id") and not row.get("error")
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("keywords")]
    errors = [row for row in rows if row.get("error")]
    prefix_counter: Counter[str] = Counter()
    for row in valid_rows:
        for keyword in row.get("keywords", []):
            prefix_counter[keyword.get("type") or keyword_prefix(keyword["keyword_id"]) or "unknown"] += 1
    return {
        "num_cases": len(rows),
        "num_valid": len(valid_rows),
        "num_errors": len(errors),
        "total_keywords": sum(len(row.get("keywords", [])) for row in valid_rows),
        "keyword_id_prefix_counts": dict(prefix_counter.most_common()),
        "avg_elapsed_seconds": round(
            sum(float(row.get("elapsed_seconds", 0.0)) for row in valid_rows) / max(1, len(valid_rows)),
            3,
        ),
    }


def default_batch_output(split: str) -> Path:
    if split == "public":
        return DEFAULT_PUBLIC_BATCH_OUTPUT
    if split == "private":
        return DEFAULT_PRIVATE_BATCH_OUTPUT
    return DEFAULT_OUTPUT_ROOT / f"{split}_llm_case_query_keywords.json"


def cmd_extract(args: argparse.Namespace) -> None:
    if not args.query and not args.case_id:
        raise ValueError("Provide either --query or --case-id")
    case_query = args.query or find_case_query(args.case_id, args.split)
    instruction = args.instruction.read_text(encoding="utf-8")
    llm = load_llm(args.model, args.n_ctx, args.n_threads, args.verbose)
    result = extract_keywords_from_query(
        llm=llm,
        case_query=case_query,
        instruction=instruction,
        max_tokens=args.max_tokens,
    )
    payload = {
        "case_id": args.case_id,
        "split": args.split,
        "case_query": case_query,
        **result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_batch(args: argparse.Namespace) -> None:
    instruction = args.instruction.read_text(encoding="utf-8")
    cases = load_cases(args.split)
    if args.limit is not None:
        cases = cases[: args.limit]

    output = args.output or default_batch_output(args.split)
    existing = load_existing_rows(output) if args.resume else {}
    rows_by_case_id: dict[str, dict[str, Any]] = dict(existing)
    llm = load_llm(args.model, args.n_ctx, args.n_threads, args.verbose)
    started_at = time.perf_counter()

    for index, case in enumerate(cases, 1):
        case_id = case["case_id"]
        if case_id in rows_by_case_id:
            print(f"[{index}/{len(cases)}] {case_id} -> skipped")
            continue

        try:
            result = extract_keywords_from_query(
                llm=llm,
                case_query=case["case_query"],
                instruction=instruction,
                max_tokens=args.max_tokens,
            )
            rows_by_case_id[case_id] = {
                "case_id": case_id,
                "case_query": case["case_query"],
                **result,
            }
            print(
                f"[{index}/{len(cases)}] {case_id} -> "
                f"{result['keyword_count']} keywords ({result['elapsed_seconds']}s)"
            )
        except Exception as exc:
            rows_by_case_id[case_id] = {
                "case_id": case_id,
                "case_query": case["case_query"],
                "error": str(exc),
            }
            print(f"[{index}/{len(cases)}] {case_id} -> ERROR: {exc}")

        ordered_rows = [rows_by_case_id[case["case_id"]] for case in cases if case["case_id"] in rows_by_case_id]
        write_json(output, ordered_rows)

    ordered_rows = [rows_by_case_id[case["case_id"]] for case in cases if case["case_id"] in rows_by_case_id]
    summary = {
        "output": str(output),
        "elapsed_total_seconds": round(time.perf_counter() - started_at, 3),
        **summarize(ordered_rows),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract important keywords from ALQAC case_query using Qwen local.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--instruction", type=Path, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--query")
    extract_parser.add_argument("--case-id")
    extract_parser.add_argument("--split", choices=["public", "private"], default="public")
    extract_parser.set_defaults(func=cmd_extract)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("--split", choices=["public", "private"], default="public")
    batch_parser.add_argument("--limit", type=int)
    batch_parser.add_argument("--resume", action="store_true")
    batch_parser.add_argument("--output", type=Path)
    batch_parser.set_defaults(func=cmd_batch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
