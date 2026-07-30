from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_case_query_keyword_extractor import (
    DEFAULT_INSTRUCTION,
    DEFAULT_MODEL,
    extract_keywords_from_query,
)
from llm_case_type_classifier import find_case_query, load_llm, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "llm_case_query_keywords" / "public_tests"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Qwen keyword extraction on one ALQAC public-test case_id."
    )
    parser.add_argument("--case-id", required=True, help="case_id from data/ALQAC2026_public_test.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--instruction", type=Path, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_query = find_case_query(args.case_id, "public")
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
        "split": "public",
        "case_query": case_query,
        **result,
    }

    if not args.no_save:
        output = args.output or DEFAULT_OUTPUT_DIR / f"{args.case_id}.json"
        write_json(output, payload)
        payload = {"saved_to": str(output), **payload}

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
