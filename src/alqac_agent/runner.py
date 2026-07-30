from __future__ import annotations

from pathlib import Path

from .case_api import CaseContentClient
from .config import Settings
from .data import (
    load_cases,
    load_json,
    load_law_articles,
    load_public_examples,
    write_json_atomic,
)
from .graph import ALQACWorkflow
from .llm import LegalAgents
from .local_case_api import LocalCaseContentClient
from .preloaded_evidence import PreloadedEvidenceClient
from .retrieval import ExampleRetriever, LawRetriever
from .schemas import SubmissionItem


SUBMISSION_FIELDS = ("case_id", "prediction", "case_evidence", "law_evidence")


def _submission_payload(raw: dict[str, object]) -> dict[str, object]:
    item = SubmissionItem.model_validate(raw)
    payload = item.model_dump(exclude_none=True)
    return {key: payload[key] for key in SUBMISSION_FIELDS}


def _default_evidence_output(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_suffix(output.suffix + ".evidence.json")


def _evidence_trace(state: dict[str, object]) -> dict[str, object]:
    submission = _submission_payload(dict(state["submission"]))
    return {
        "case_id": submission["case_id"],
        "prediction": submission["prediction"],
        "api_queries": state.get("api_queries", []),
        "fact_queries": state.get("fact_queries", []),
        "case_type_info": state.get("case_type_info", {}),
        "keywords": state.get("keywords", []),
        "submitted_case_evidence": submission["case_evidence"],
        "retrieved_case_evidence": state.get("case_evidence", []),
        "disposition": state.get("disposition", {}),
        "money_signal": state.get("money_signal", {}),
        "area_signal": state.get("area_signal", {}),
        "outcome_prior": state.get("outcome_prior", {}),
        "decision": state.get("decision", {}),
        "law_candidates": state.get("law_candidates", []),
        "law_context_for_outcome": state.get("law_candidates", [])[:8],
        "law_evidence": state.get("law_evidence", []),
    }


def build_workflow(
    settings: Settings,
    law_path: str | Path,
    examples_path: str | Path | None,
    case_backend: str = "official",
    case_corpus_path: str | Path = "ALQAC2026_public_test.json",
    evidence_jsonl_path: str | Path | None = None,
) -> tuple[object, object]:
    articles = load_law_articles(law_path)
    examples = load_public_examples(examples_path) if examples_path is not None else []
    if case_backend == "local-public":
        client = LocalCaseContentClient(case_corpus_path)
    elif case_backend == "preloaded":
        if evidence_jsonl_path is None:
            raise ValueError("preloaded backend requires evidence_jsonl_path")
        client = PreloadedEvidenceClient(evidence_jsonl_path)
    elif case_backend == "official":
        client = CaseContentClient(
            url=settings.api_url,
            api_key=settings.api_key,
            cache_path=settings.cache_path,
            interval_seconds=settings.api_interval_seconds,
            timeout_seconds=settings.api_timeout_seconds,
        )
    else:
        raise ValueError(f"Case backend không hợp lệ: {case_backend}")
    workflow = ALQACWorkflow(
        settings=settings,
        agents=LegalAgents(settings),
        case_client=client,
        law_retriever=LawRetriever(articles),
        example_retriever=ExampleRetriever(examples),
    ).compile()
    return workflow, client


def run_batch(
    *,
    input_path: str | Path,
    law_path: str | Path,
    examples_path: str | Path | None,
    output_path: str | Path,
    settings: Settings,
    limit: int | None = None,
    include_explanation: bool = False,
    resume: bool = True,
    case_backend: str = "official",
    case_corpus_path: str | Path = "ALQAC2026_public_test.json",
    evidence_output_path: str | Path | None = None,
    evidence_jsonl_path: str | Path | None = None,
) -> list[dict[str, object]]:
    settings.validate_for_run(require_api_key=case_backend == "official")
    cases = load_cases(input_path)
    if limit is not None:
        cases = cases[:limit]
    output = Path(output_path)
    evidence_output = (
        _default_evidence_output(output)
        if evidence_output_path is None
        else Path(evidence_output_path)
    )
    completed: dict[str, dict[str, object]] = {}
    evidence_traces: dict[str, dict[str, object]] = {}
    if resume and output.exists():
        previous = load_json(output)
        for raw in previous:
            restored = _submission_payload(raw)
            completed[str(restored["case_id"])] = restored
    if resume and evidence_output.exists():
        previous_traces = load_json(evidence_output)
        if isinstance(previous_traces, list):
            for raw in previous_traces:
                if isinstance(raw, dict) and "case_id" in raw:
                    evidence_traces[str(raw["case_id"])] = raw

    graph, client = build_workflow(
        settings,
        law_path,
        examples_path,
        case_backend=case_backend,
        case_corpus_path=case_corpus_path,
        evidence_jsonl_path=evidence_jsonl_path,
    )
    try:
        for index, case in enumerate(cases, start=1):
            if case.case_id in completed:
                continue
            print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
            state = graph.invoke(case.model_dump())
            result = _submission_payload(dict(state["submission"]))
            completed[case.case_id] = result
            evidence_traces[case.case_id] = _evidence_trace(dict(state))
            ordered = [completed[item.case_id] for item in cases if item.case_id in completed]
            write_json_atomic(output, ordered)
            ordered_traces = [
                evidence_traces[item.case_id]
                for item in cases
                if item.case_id in evidence_traces
            ]
            write_json_atomic(evidence_output, ordered_traces)
    finally:
        client.close()
    return [completed[item.case_id] for item in cases if item.case_id in completed]
