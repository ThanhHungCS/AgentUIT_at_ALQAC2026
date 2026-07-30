from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from llm_case_type_classifier import (
    DEFAULT_INSTRUCTION,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SIMPLE_CORPUS,
    classify_query,
    load_case_types,
    load_cases,
    load_llm,
    write_json,
)


DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "public_llm_case_type_predictions.json"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


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
    valid_rows = [row for row in rows if row.get("primary_case_type_id")]
    errors = [row for row in rows if row.get("error")]
    return {
        "num_cases": len(rows),
        "num_valid": len(valid_rows),
        "num_errors": len(errors),
        "type_counts": dict(Counter(row["primary_case_type_id"] for row in valid_rows).most_common()),
        "avg_elapsed_seconds": round(
            sum(float(row.get("elapsed_seconds", 0.0)) for row in valid_rows) / max(1, len(valid_rows)),
            3,
        ),
    }


def run_public_batch(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output
    instruction = args.instruction.read_text(encoding="utf-8")
    case_types = load_case_types(args.case_types)
    cases = load_cases("public")
    if args.limit is not None:
        cases = cases[: args.limit]

    existing = load_existing_rows(output_path) if args.resume else {}
    rows_by_case_id: dict[str, dict[str, Any]] = dict(existing)

    llm = load_llm(args.model, args.n_ctx, args.n_threads, args.verbose)
    started_at = time.perf_counter()

    for index, case in enumerate(cases, 1):
        case_id = case["case_id"]
        if case_id in rows_by_case_id:
            print(f"[{index}/{len(cases)}] {case_id} -> skipped")
            continue

        try:
            result = classify_query(
                llm=llm,
                case_query=case["case_query"],
                instruction=instruction,
                case_types=case_types,
                max_tokens=args.max_tokens,
            )
            rows_by_case_id[case_id] = {
                "case_id": case_id,
                "case_query": case["case_query"],
                **result,
            }
            print(
                f"[{index}/{len(cases)}] {case_id} -> "
                f"{result['primary_case_type_id']} ({result['elapsed_seconds']}s)"
            )
        except Exception as exc:
            rows_by_case_id[case_id] = {
                "case_id": case_id,
                "case_query": case["case_query"],
                "error": str(exc),
            }
            print(f"[{index}/{len(cases)}] {case_id} -> ERROR: {exc}")

        ordered_rows = [rows_by_case_id[case["case_id"]] for case in cases if case["case_id"] in rows_by_case_id]
        write_json(output_path, ordered_rows)

    ordered_rows = [rows_by_case_id[case["case_id"]] for case in cases if case["case_id"] in rows_by_case_id]
    summary = {
        "output": str(output_path),
        "elapsed_total_seconds": round(time.perf_counter() - started_at, 3),
        **summarize(ordered_rows),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen local case-type classifier on all public test cases.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--instruction", type=Path, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--case-types", type=Path, default=DEFAULT_SIMPLE_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="skip cases already present in output")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_public_batch(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
