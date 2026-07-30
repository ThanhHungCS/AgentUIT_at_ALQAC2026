from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from retrieve_public_evidence import DEFAULT_CONFIG, TASK_ROOT, retrieve_cases


DEFAULT_OUTPUT_DIR = TASK_ROOT / "outputs" / "per_case"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "case"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve evidence chunks for one ALQAC public-test case_id."
    )
    parser.add_argument("case_id", help="Public-test case_id, for example: case_4588")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--blueprints", type=Path)
    parser.add_argument("--output", type=Path, help="Output JSONL file. Defaults to outputs/per_case/<case_id>_api_evidence.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--token", help="Override ALQAC_TOKEN from api_public_evidence_retrieval/.env")
    parser.add_argument("--dry-run", action="store_true", help="Instantiate queries only; do not call the API.")
    parser.add_argument("--append", action="store_true", help="Append to the output file instead of replacing it.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output
    if output is None:
        output = args.output_dir / f"{safe_filename(args.case_id)}_api_evidence.jsonl"

    retrieve_args = argparse.Namespace(
        config=args.config,
        blueprints=args.blueprints,
        output=output,
        case_id=args.case_id,
        start=None,
        limit=None,
        token=args.token,
        dry_run=args.dry_run,
        overwrite=not args.append,
    )
    retrieve_cases(retrieve_args)


if __name__ == "__main__":
    main()
