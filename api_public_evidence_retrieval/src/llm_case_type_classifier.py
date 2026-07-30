from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from llama_cpp import Llama


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_HF_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_MODEL = ROOT / "models" / "qwen2.5-3b-instruct-gguf" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
DEFAULT_INSTRUCTION = ROOT / "prompts" / "case_type_classifier_instruction.txt"
DEFAULT_SIMPLE_CORPUS = ROOT / "data" / "keyword_corpora_by_case_type.simple.json"
DEFAULT_PUBLIC = ROOT / "data" / "ALQAC2026_public_test.json"
DEFAULT_PRIVATE = ROOT / "data" / "ALQAC_private_test.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "llm_case_type_classification"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(split: str) -> list[dict[str, Any]]:
    path = DEFAULT_PUBLIC if split == "public" else DEFAULT_PRIVATE
    return load_json(path)


def find_case_query(case_id: str, split: str) -> str:
    for case in load_cases(split):
        if case.get("case_id") == case_id:
            return case["case_query"]
    raise KeyError(f"{case_id} not found in {split} data")


def load_case_types(path: Path) -> list[dict[str, str]]:
    rows = load_json(path)
    return [
        {
            "case_type_id": row["case_type_id"],
            "case_type_text": row["case_type_text"],
        }
        for row in rows
    ]


def build_user_prompt(case_query: str, case_types: list[dict[str, str]]) -> str:
    return (
        "case_types:\n"
        + json.dumps(case_types, ensure_ascii=False, indent=2)
        + "\n\ncase_query:\n"
        + case_query
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model output does not contain a JSON object: {text}")
    return json.loads(match.group(0))


def validate_prediction(prediction: dict[str, Any], case_types: list[dict[str, str]]) -> dict[str, Any]:
    valid_by_id = {item["case_type_id"]: item for item in case_types}
    valid_by_text = {item["case_type_text"]: item for item in case_types}

    case_type_id = prediction.get("primary_case_type_id")
    case_type_text = prediction.get("primary_case_type_text")

    if case_type_id in valid_by_id:
        canonical = valid_by_id[case_type_id]
    elif case_type_text in valid_by_text:
        canonical = valid_by_text[case_type_text]
    else:
        raise ValueError(f"Invalid case type returned by model: {prediction}")

    return {
        "primary_case_type_id": canonical["case_type_id"],
        "primary_case_type_text": canonical["case_type_text"],
    }


def classify_query(
    llm: Llama,
    case_query: str,
    instruction: str,
    case_types: list[dict[str, str]],
    max_tokens: int = 96,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": build_user_prompt(case_query, case_types)},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
    )
    elapsed_seconds = time.perf_counter() - started_at
    content = response["choices"][0]["message"]["content"]
    parsed = validate_prediction(extract_json_object(content), case_types)
    return {
        **parsed,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "raw_output": content,
    }


def resolve_model_path(model_path: Path) -> Path:
    if not model_path.exists():
        if model_path == DEFAULT_MODEL:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=DEFAULT_HF_REPO_ID,
                filename=DEFAULT_HF_FILENAME,
            )
            return Path(downloaded)
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path


def load_llm(model_path: Path, n_ctx: int, n_threads: int, verbose: bool, n_gpu_layers: int = -1) -> Llama:
    return Llama(
        model_path=str(resolve_model_path(model_path)),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        chat_format="chatml",
        verbose=verbose,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_classify(args: argparse.Namespace) -> None:
    if not args.query and not args.case_id:
        raise ValueError("Provide either --query or --case-id")
    case_query = args.query or find_case_query(args.case_id, args.split)
    instruction = args.instruction.read_text(encoding="utf-8")
    case_types = load_case_types(args.case_types)
    llm = load_llm(args.model, args.n_ctx, args.n_threads, args.verbose)
    result = classify_query(llm, case_query, instruction, case_types, max_tokens=args.max_tokens)
    payload = {
        "case_id": args.case_id,
        "split": args.split,
        "case_query": case_query,
        **result,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_batch(args: argparse.Namespace) -> None:
    instruction = args.instruction.read_text(encoding="utf-8")
    case_types = load_case_types(args.case_types)
    llm = load_llm(args.model, args.n_ctx, args.n_threads, args.verbose)
    rows = []
    selected_cases = load_cases(args.split)
    if args.limit is not None:
        selected_cases = selected_cases[: args.limit]

    for index, case in enumerate(selected_cases, 1):
        try:
            result = classify_query(
                llm,
                case["case_query"],
                instruction,
                case_types,
                max_tokens=args.max_tokens,
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "case_query": case["case_query"],
                    **result,
                }
            )
            print(
                f"[{index}/{len(selected_cases)}] {case['case_id']} -> "
                f"{result['primary_case_type_id']} ({result['elapsed_seconds']}s)"
            )
        except Exception as exc:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "case_query": case["case_query"],
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(selected_cases)}] {case['case_id']} -> ERROR: {exc}")

    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.split}_llm_case_type_predictions.json"
    write_json(output, rows)
    print(json.dumps({"output": str(output), "num_cases": len(rows)}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-based ALQAC case type classifier.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--instruction", type=Path, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--case-types", type=Path, default=DEFAULT_SIMPLE_CORPUS)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(required=True)

    classify_parser = subparsers.add_parser("classify")
    classify_parser.add_argument("--query")
    classify_parser.add_argument("--case-id")
    classify_parser.add_argument("--split", choices=["public", "private"], default="public")
    classify_parser.set_defaults(func=cmd_classify)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("--split", choices=["public", "private"], default="public")
    batch_parser.add_argument("--limit", type=int)
    batch_parser.add_argument("--output", type=Path)
    batch_parser.set_defaults(func=cmd_batch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
