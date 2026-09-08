from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .data import load_json, load_law_articles, write_json_atomic
from .ljp_input import process_case_fact
from .ljp_reasoning import ZeroShotJudge, resolve_judgment
from .ljp_retrieval import retrieve_relevant_laws
from .retrieval import LawRetriever


class LJPState(TypedDict, total=False):
    case_id: str
    case_fact: str
    metadata: dict[str, Any]
    processed_case: dict[str, Any]
    law_query: str
    relevant_laws: list[dict[str, Any]]
    judgment: dict[str, Any]
    prediction_item: dict[str, str]


class ZeroShotLJPWorkflow:
    """Three-module zero-shot LJP workflow: input processing, law retrieval, reasoning."""

    def __init__(
        self,
        *,
        law_retriever: LawRetriever,
        judge: ZeroShotJudge | None = None,
        law_candidates: int = 18,
    ) -> None:
        self.law_retriever = law_retriever
        self.judge = judge
        self.law_candidates = law_candidates

    def process_input(self, state: LJPState) -> dict[str, Any]:
        processed = process_case_fact(
            state["case_id"],
            state["case_fact"],
            state.get("metadata", {}),
        )
        return {"processed_case": processed}

    def retrieve_laws(self, state: LJPState) -> dict[str, Any]:
        retrieval = retrieve_relevant_laws(
            state["processed_case"],
            self.law_retriever,
            top_k=self.law_candidates,
        )
        processed = dict(state["processed_case"])
        processed["law_query"] = retrieval["law_query"]
        return {
            "processed_case": processed,
            "law_query": retrieval["law_query"],
            "relevant_laws": retrieval["relevant_laws"],
        }

    def reason_judgment(self, state: LJPState) -> dict[str, Any]:
        judgment = resolve_judgment(
            state["processed_case"],
            state.get("relevant_laws", []),
            judge=self.judge,
        )
        return {
            "judgment": judgment,
            "prediction_item": {
                "case_id": str(judgment["case_id"]),
                "prediction": str(judgment["prediction"]),
            },
        }

    def compile(self):
        builder = StateGraph(LJPState)
        builder.add_node("process_input", self.process_input)
        builder.add_node("retrieve_laws", self.retrieve_laws)
        builder.add_node("reason_judgment", self.reason_judgment)
        builder.add_edge(START, "process_input")
        builder.add_edge("process_input", "retrieve_laws")
        builder.add_edge("retrieve_laws", "reason_judgment")
        builder.add_edge("reason_judgment", END)
        return builder.compile()


def _load_ljp_cases(path: str | Path) -> list[dict[str, Any]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: root must be a JSON array")
    cases: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id", "")).strip()
        case_fact = str(item.get("case_fact", "")).strip()
        if not case_id or not case_fact:
            raise ValueError(f"{path}: missing case_id or case_fact")
        metadata = {
            key: item[key]
            for key in ("court", "case_type", "A_role", "B_role", "A_description", "B_description")
            if key in item
        }
        cases.append({"case_id": case_id, "case_fact": case_fact, "metadata": metadata})
    ids = [str(item["case_id"]) for item in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate case_id")
    return cases


def run_ljp_public(
    *,
    input_path: str | Path,
    law_path: str | Path,
    output_path: str | Path,
    judge: ZeroShotJudge | None = None,
    limit: int | None = None,
    resume: bool = True,
    law_candidates: int = 18,
    trace_output_path: str | Path | None = None,
) -> list[dict[str, str]]:
    cases = _load_ljp_cases(input_path)
    if limit is not None:
        cases = cases[:limit]
    output = Path(output_path)
    trace_output = Path(trace_output_path) if trace_output_path else None
    completed: dict[str, dict[str, str]] = {}
    traces: dict[str, dict[str, Any]] = {}

    if resume and output.exists():
        previous = load_json(output)
        if isinstance(previous, list):
            for raw in previous:
                if isinstance(raw, dict) and "case_id" in raw and "prediction" in raw:
                    completed[str(raw["case_id"])] = {
                        "case_id": str(raw["case_id"]),
                        "prediction": str(raw["prediction"]),
                    }
    if resume and trace_output and trace_output.exists():
        previous_trace = load_json(trace_output)
        if isinstance(previous_trace, list):
            for raw in previous_trace:
                if isinstance(raw, dict) and "case_id" in raw:
                    traces[str(raw["case_id"])] = raw

    workflow = ZeroShotLJPWorkflow(
        law_retriever=LawRetriever(load_law_articles(law_path)),
        judge=judge,
        law_candidates=law_candidates,
    ).compile()

    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        if case_id in completed:
            continue
        print(f"[ljp {index}/{len(cases)}] {case_id}", flush=True)
        state = workflow.invoke(case)
        completed[case_id] = dict(state["prediction_item"])
        traces[case_id] = {
            "case_id": case_id,
            **dict(state["judgment"].get("trace", {})),
        }
        ordered = [
            completed[str(item["case_id"])]
            for item in cases
            if str(item["case_id"]) in completed
        ]
        write_json_atomic(output, ordered)
        if trace_output:
            ordered_trace = [
                traces[str(item["case_id"])]
                for item in cases
                if str(item["case_id"]) in traces
            ]
            write_json_atomic(trace_output, ordered_trace)
    return [
        completed[str(item["case_id"])]
        for item in cases
        if str(item["case_id"]) in completed
    ]
