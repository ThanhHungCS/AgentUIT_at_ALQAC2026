from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .data import load_cases, load_law_articles, load_public_examples
from .evaluation import evaluate_outcomes, prepare_public_input
from .ljp_graph import run_ljp_public
from .llm import LegalAgents
from .probing import derive_fact_query_report, probe_evidence_queries
from .retrieval import LawRetriever
from .runner import run_batch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ALQAC 2026 LangGraph agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="kiểm tra dữ liệu, không gọi API")
    validate.add_argument("--input", default="ALQAC_private_test.json")
    validate.add_argument("--laws", default="corpus_law_pub.json")
    validate.add_argument("--examples", default="ALQAC2026_public_test.json")

    inspect = subparsers.add_parser("inspect-law", help="thử BM25 local, không gọi API")
    inspect.add_argument("query")
    inspect.add_argument("--laws", default="corpus_law_pub.json")
    inspect.add_argument("--top-k", type=int, default=5)

    prepare = subparsers.add_parser(
        "prepare-public", help="tạo input public giống private, không gọi API"
    )
    prepare.add_argument("--source", default="ALQAC2026_public_test.json")
    prepare.add_argument("--output", default="ALQAC2026_public_input.json")

    evaluate = subparsers.add_parser(
        "evaluate", help="chấm prediction public bằng verdict_label"
    )
    evaluate.add_argument("--gold", default="ALQAC2026_public_test.json")
    evaluate.add_argument("--predictions", default="submission.public.json")

    probe = subparsers.add_parser(
        "probe-evidence", help="thử nhiều query evidence trên public cases"
    )
    probe.add_argument("--public", default="ALQAC2026_public_test.json")
    probe.add_argument("--queries", default="queries.evidence.txt")
    probe.add_argument(
        "--case-backend",
        choices=("official", "local-public"),
        default="local-public",
    )
    probe.add_argument("--output", default="probe.evidence.json")
    probe.add_argument("--limit", type=int)

    derive_fact = subparsers.add_parser(
        "derive-fact-queries",
        help="tạo query fact-style từ case_query và đo overlap với public case_fact",
    )
    derive_fact.add_argument("--public", default="ALQAC2026_public_test.json")
    derive_fact.add_argument("--output", default="fact_queries.public.json")
    derive_fact.add_argument("--limit", type=int)

    run = subparsers.add_parser("run", help="chạy agent và tạo submission")
    run.add_argument("--input", default="ALQAC_private_test.json")
    run.add_argument("--laws", default="corpus_law_pub.json")
    run.add_argument(
        "--examples",
        default=None,
        help="tùy chọn: public examples có label để calibration; mặc định không dùng",
    )
    run.add_argument(
        "--no-few-shot",
        action="store_true",
        help="không đưa public label nào vào agent",
    )
    run.add_argument("--output", default="submission.json")
    run.add_argument(
        "--case-backend",
        choices=("official", "local-public", "preloaded"),
        default="official",
        help="nguồn case evidence",
    )
    run.add_argument(
        "--case-corpus",
        default="ALQAC2026_public_test.json",
        help="public JSON dùng bởi backend local-public",
    )
    run.add_argument(
        "--evidence-jsonl",
        default=None,
        help="JSONL file containing pre-retrieved evidence (for preloaded backend)",
    )
    run.add_argument("--limit", type=int)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--explanation", action="store_true")
    run.add_argument(
        "--evidence-output",
        help=(
            "file dev trace chứa full retrieved evidence; mặc định là "
            "<output>.evidence.json"
        ),
    )

    ljp = subparsers.add_parser(
        "run-ljp",
        help="chạy workflow Legal Judgment Prediction từ case_fact, không gọi ALQAC API",
    )
    ljp.add_argument("--input", default="ALQAC2026_public_test.json")
    ljp.add_argument("--laws", default="corpus_law_pub.json")
    ljp.add_argument("--output", default="runs/submission.ljp.public.json")
    ljp.add_argument(
        "--model",
        default="qwen3.5-9b",
        help="model alias served by the OpenAI-compatible LLM server",
    )
    ljp.add_argument("--llm-base-url", default=None)
    ljp.add_argument("--llm-provider", default=None, choices=("vllm", "ollama", "groq"))
    ljp.add_argument(
        "--structured-method",
        default=None,
        choices=("json_schema", "prompt_json"),
    )
    ljp.add_argument("--limit", type=int)
    ljp.add_argument("--no-resume", action="store_true")
    ljp.add_argument("--no-llm", action="store_true")
    ljp.add_argument("--law-candidates", type=int, default=18)
    ljp.add_argument(
        "--trace-output",
        default=None,
        help="optional debug trace path with processed input, law query, laws and reasoning",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validate":
        cases = load_cases(args.input)
        articles = load_law_articles(args.laws)
        examples = load_public_examples(args.examples)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "input_cases": len(cases),
                    "law_articles": len(articles),
                    "public_examples": len(examples),
                    "network_calls": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "inspect-law":
        retriever = LawRetriever(load_law_articles(args.laws))
        print(json.dumps(retriever.search(args.query, args.top_k), ensure_ascii=False, indent=2))
        return
    if args.command == "prepare-public":
        inputs = prepare_public_input(args.source, args.output)
        print(
            json.dumps(
                {"status": "ok", "cases": len(inputs), "output": args.output, "network_calls": 0},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "evaluate":
        print(
            json.dumps(
                evaluate_outcomes(args.gold, args.predictions),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "probe-evidence":
        settings = Settings.from_env()
        settings.validate_for_run(require_api_key=args.case_backend == "official")
        print(
            json.dumps(
                probe_evidence_queries(
                    settings=settings,
                    public_path=args.public,
                    queries_path=args.queries,
                    backend=args.case_backend,
                    output_path=args.output,
                    limit=args.limit,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "derive-fact-queries":
        print(
            json.dumps(
                derive_fact_query_report(
                    public_path=args.public,
                    output_path=args.output,
                    limit=args.limit,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "run-ljp":
        settings = Settings.from_env()
        overrides = {"llm_structured_method": "prompt_json"}
        if args.model is not None:
            overrides["llm_model"] = args.model
        if args.llm_base_url is not None:
            overrides["llm_base_url"] = args.llm_base_url
        if args.llm_provider is not None:
            overrides["llm_provider"] = args.llm_provider
        if args.structured_method is not None:
            overrides["llm_structured_method"] = args.structured_method
        if overrides:
            settings = replace(settings, **overrides)
        settings.validate_for_run(require_api_key=False)
        agents = None if args.no_llm else LegalAgents(settings)
        run_ljp_public(
            input_path=args.input,
            law_path=args.laws,
            output_path=args.output,
            judge=agents,
            limit=args.limit,
            resume=not args.no_resume,
            law_candidates=args.law_candidates,
            trace_output_path=args.trace_output,
        )
        return
    run_batch(
        input_path=args.input,
        law_path=args.laws,
        examples_path=None if args.no_few_shot else args.examples,
        output_path=args.output,
        settings=Settings.from_env(),
        limit=args.limit,
        include_explanation=args.explanation,
        resume=not args.no_resume,
        case_backend=args.case_backend,
        case_corpus_path=args.case_corpus,
        evidence_output_path=args.evidence_output,
        evidence_jsonl_path=args.evidence_jsonl,
    )


if __name__ == "__main__":
    main()
