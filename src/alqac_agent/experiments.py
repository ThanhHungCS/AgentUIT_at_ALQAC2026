from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .data import load_json, load_law_articles, write_json_atomic
from .evaluation import LABELS, evaluate_outcomes
from .ljp_input import normalize_case_fact, process_case_fact
from .ljp_reasoning import ZeroShotJudge, resolve_judgment
from .ljp_retrieval import compact_laws, retrieve_relevant_laws
from .retrieval import LawRetriever
from .schemas import LJPLabelPrediction, VerdictLabel

ExperimentMode = Literal[
    "prompt_only",
    "method",
    "no_law_retrieval",
    "no_input_processing",
    "no_law_retrieval_no_input_processing",
]

MAIN_METHODS = {"prompt_only", "method"}
ABLATION_METHODS = {
    "method",
    "no_law_retrieval",
    "no_input_processing",
    "no_law_retrieval_no_input_processing",
}
METHOD_LABELS = {
    "prompt_only": "Prompt-only",
    "method": "+ Method",
    "no_law_retrieval": "w/o Law Retrieval",
    "no_input_processing": "w/o Input Processing",
    "no_law_retrieval_no_input_processing": "w/o Law Retrieval + w/o Input Processing",
}
METRIC_COLUMNS = [
    "Accuracy",
    "Macro-F1",
    "A_WIN F1",
    "PARTIAL_A_WIN F1",
    "PARTIAL_B_WIN F1",
    "B_WIN F1",
    "Coverage",
]


@dataclass(frozen=True, slots=True)
class ExperimentInfo:
    experiment_id: str
    model_type: str
    backbone: str
    params: str
    domain: str
    mode: ExperimentMode


def _slug(value: str) -> str:
    cleaned = []
    for char in value.casefold():
        cleaned.append(char if char.isalnum() else "_")
    slug = "_".join("".join(cleaned).split("_"))
    return slug or "experiment"


def default_experiment_id(backbone: str, params: str, mode: ExperimentMode) -> str:
    return _slug(f"{backbone}_{params}_{mode}")


def _load_cases(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
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
    return cases[:limit] if limit is not None else cases


def _raw_processed_case(case: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_case_fact(str(case["case_fact"]))
    return {
        "case_id": str(case["case_id"]),
        "case_fact": normalized,
        "metadata": dict(case.get("metadata", {})),
        "segments": [
            {
                "segment_id": "raw_001",
                "section": "raw_case_fact",
                "speaker": "unknown",
                "marker": "",
                "text": normalized,
            }
        ],
        "main_claim": {
            "main_claim": normalized[:2200],
            "requested_amounts": [],
            "requested_areas": [],
        },
        "disposition_candidates": [
            {
                "chunk_id": "raw_001",
                "segment_id": "raw_001",
                "section": "raw_case_fact",
                "speaker": "unknown",
                "query": "raw_case_fact",
                "text": normalized[:5000],
                "score": 0,
            }
        ],
    }


def _validate_prompt_judge(judge: ZeroShotJudge | None) -> Any:
    if judge is None or not hasattr(judge, "predict_ljp_label"):
        raise ValueError("Prompt-only experiments require an LLM judge with predict_ljp_label().")
    return judge


def _run_prompt_only(case: dict[str, Any], judge: ZeroShotJudge | None) -> dict[str, Any]:
    predictor = _validate_prompt_judge(judge)
    try:
        decision: LJPLabelPrediction = predictor.predict_ljp_label(str(case["case_fact"]))
    except Exception as error:
        prediction: VerdictLabel = "PARTIAL_A_WIN"
        return {
            "case_id": str(case["case_id"]),
            "prediction": prediction,
            "trace": {
                "mode": "prompt_only",
                "case_fact_excerpt": normalize_case_fact(str(case["case_fact"]))[:1800],
                "llm_error": {
                    "type": type(error).__name__,
                    "message": str(error)[:1200],
                },
                "fallback_used": True,
                "fallback_prediction": prediction,
            },
        }
    return {
        "case_id": str(case["case_id"]),
        "prediction": decision.prediction,
        "trace": {
            "mode": "prompt_only",
            "case_fact_excerpt": normalize_case_fact(str(case["case_fact"]))[:1800],
            "decision": decision.model_dump(),
            "fallback_used": False,
        },
    }


def _run_architecture_mode(
    case: dict[str, Any],
    *,
    mode: ExperimentMode,
    law_retriever: LawRetriever | None,
    law_candidates: int,
    judge: ZeroShotJudge | None,
) -> dict[str, Any]:
    use_input_processing = mode not in {
        "no_input_processing",
        "no_law_retrieval_no_input_processing",
    }
    use_law_retrieval = mode not in {
        "no_law_retrieval",
        "no_law_retrieval_no_input_processing",
    }
    processed = (
        process_case_fact(str(case["case_id"]), str(case["case_fact"]), dict(case.get("metadata", {})))
        if use_input_processing
        else _raw_processed_case(case)
    )
    laws: list[dict[str, Any]] = []
    law_query = ""
    if use_law_retrieval:
        if law_retriever is None:
            raise ValueError("Law retrieval mode requires --laws.")
        retrieval = retrieve_relevant_laws(processed, law_retriever, top_k=law_candidates)
        laws = list(retrieval["relevant_laws"])
        law_query = str(retrieval["law_query"])
        processed = dict(processed)
        processed["law_query"] = law_query
    else:
        processed = dict(processed)
        processed["law_query"] = ""
    judgment = resolve_judgment(processed, laws, judge=judge)
    trace = dict(judgment.get("trace", {}))
    trace.update(
        {
            "mode": mode,
            "input_processing_enabled": use_input_processing,
            "law_retrieval_enabled": use_law_retrieval,
            "law_query": law_query,
            "relevant_laws": compact_laws(laws, limit=8),
        }
    )
    return {
        "case_id": str(judgment["case_id"]),
        "prediction": str(judgment["prediction"]),
        "trace": trace,
    }


def _paths(result_dir: Path, experiment_id: str) -> dict[str, Path]:
    return {
        "submission": result_dir / "submissions" / f"{experiment_id}.json",
        "metrics": result_dir / "metrics" / f"{experiment_id}.metrics.json",
        "trace": result_dir / "traces" / f"{experiment_id}.trace.json",
        "confusion_matrix": result_dir
        / "confusion_matrices"
        / f"{experiment_id}.confusion_matrix.json",
        "retrieval": result_dir / "qualitative" / f"{experiment_id}.retrieval_outputs.json",
        "input": result_dir / "qualitative" / f"{experiment_id}.input_processing_outputs.json",
        "special": result_dir / "qualitative" / f"{experiment_id}.special_cases.json",
    }


def _metric_row(info: ExperimentInfo, metrics: dict[str, Any]) -> dict[str, str]:
    per_label = metrics.get("per_label", {})
    row = {
        "Type": info.model_type,
        "Backbone": info.backbone,
        "Params": info.params,
        "Domain": info.domain,
        "Method": METHOD_LABELS[info.mode],
        "Accuracy": str(metrics.get("accuracy", "")),
        "Macro-F1": str(metrics.get("macro_f1", "")),
        "Coverage": str(metrics.get("coverage", "")),
        "Confusion Matrix": f"confusion_matrices/{info.experiment_id}.confusion_matrix.json",
    }
    for label in LABELS:
        row[f"{label} F1"] = str(per_label.get(label, {}).get("f1", ""))
    return row


def _delta_row(base: dict[str, str], method: dict[str, str]) -> dict[str, str]:
    row = dict(method)
    row["Method"] = "Delta"
    row["Confusion Matrix"] = ""
    for column in METRIC_COLUMNS:
        try:
            row[column] = f"{float(method[column]) - float(base[column]):+.6f}"
        except (TypeError, ValueError, KeyError):
            row[column] = ""
    return row


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Type",
        "Backbone",
        "Params",
        "Domain",
        "Method",
        *METRIC_COLUMNS,
        "Confusion Matrix",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "Type",
        "Backbone",
        "Params",
        "Domain",
        "Method",
        *METRIC_COLUMNS,
        "Confusion Matrix",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_experiment_tables(result_dir: str | Path = "result") -> None:
    root = Path(result_dir)
    metrics_dir = root / "metrics"
    if not metrics_dir.exists():
        return
    main_rows: list[dict[str, str]] = []
    ablation_rows: list[dict[str, str]] = []
    all_metrics: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*.metrics.json")):
        payload = load_json(path)
        if not isinstance(payload, dict) or "experiment" not in payload:
            continue
        info_raw = payload["experiment"]
        info = ExperimentInfo(
            experiment_id=str(info_raw["experiment_id"]),
            model_type=str(info_raw["type"]),
            backbone=str(info_raw["backbone"]),
            params=str(info_raw["params"]),
            domain=str(info_raw["domain"]),
            mode=str(info_raw["mode"]),  # type: ignore[arg-type]
        )
        row = _metric_row(info, payload)
        all_metrics.append(payload)
        if info.mode in MAIN_METHODS:
            main_rows.append(row)
        if info.mode in ABLATION_METHODS:
            ablation_rows.append(row)

    indexed: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = {}
    for row in main_rows:
        key = (row["Type"], row["Backbone"], row["Params"], row["Domain"])
        indexed.setdefault(key, {})[row["Method"]] = row
    rows_with_delta: list[dict[str, str]] = []
    for key in indexed:
        group = indexed[key]
        for method_name in ("Prompt-only", "+ Method"):
            if method_name in group:
                rows_with_delta.append(group[method_name])
        if "Prompt-only" in group and "+ Method" in group:
            rows_with_delta.append(_delta_row(group["Prompt-only"], group["+ Method"]))
    _write_csv(root / "tables" / "main_comparison.csv", rows_with_delta)
    _write_markdown(root / "tables" / "main_comparison.md", rows_with_delta)
    _write_csv(root / "tables" / "ablation_study.csv", ablation_rows)
    _write_markdown(root / "tables" / "ablation_study.md", ablation_rows)
    write_json_atomic(root / "tables" / "all_metrics.json", all_metrics)


def _write_qualitative_files(
    paths: dict[str, Path],
    traces: list[dict[str, Any]],
    gold_path: str | Path,
    predictions: list[dict[str, str]],
) -> None:
    gold_raw = load_json(gold_path)
    gold = {str(item["case_id"]): str(item.get("verdict_label", "")) for item in gold_raw}
    predicted = {str(item["case_id"]): str(item["prediction"]) for item in predictions}

    retrieval_rows = []
    input_rows = []
    special_rows = []
    for trace in traces:
        case_id = str(trace["case_id"])
        laws = trace.get("relevant_laws", [])
        retrieval_rows.append(
            {
                "case_id": case_id,
                "law_query": trace.get("law_query", ""),
                "retrieved_laws": laws,
            }
        )
        input_rows.append(
            {
                "case_id": case_id,
                "main_claim": trace.get("main_claim", {}),
                "disposition_candidates": trace.get("disposition_candidates", []),
                "claim_scope": trace.get("claim_scope", "UNCLEAR"),
                "source": trace.get("source", ""),
            }
        )
        reasons: list[str] = []
        if predicted.get(case_id) != gold.get(case_id):
            reasons.append("wrong_prediction")
        if _float_or_zero(trace.get("confidence", 0.0)) < 0.7:
            reasons.append("low_confidence")
        if not laws and trace.get("law_retrieval_enabled"):
            reasons.append("empty_law_retrieval")
        if "PARTIAL" in predicted.get(case_id, "") or "PARTIAL" in gold.get(case_id, ""):
            reasons.append("partial_label_boundary")
        if trace.get("source") in {"MISSING", "UNKNOWN"}:
            reasons.append("weak_disposition_signal")
        if reasons:
            special_rows.append(
                {
                    "case_id": case_id,
                    "gold": gold.get(case_id, ""),
                    "prediction": predicted.get(case_id, ""),
                    "reasons": sorted(set(reasons)),
                    "retrieval": retrieval_rows[-1],
                    "input_processing": input_rows[-1],
                    "disposition": trace.get("disposition", {}),
                }
            )
    write_json_atomic(paths["retrieval"], retrieval_rows)
    write_json_atomic(paths["input"], input_rows)
    write_json_atomic(paths["special"], special_rows)


def run_ljp_experiment(
    *,
    input_path: str | Path,
    law_path: str | Path,
    result_dir: str | Path,
    info: ExperimentInfo,
    judge: ZeroShotJudge | None,
    limit: int | None = None,
    resume: bool = True,
    law_candidates: int = 18,
) -> dict[str, Any]:
    paths = _paths(Path(result_dir), info.experiment_id)
    cases = _load_cases(input_path, limit=limit)
    law_retriever = (
        LawRetriever(load_law_articles(law_path))
        if info.mode not in {"prompt_only", "no_law_retrieval_no_input_processing"}
        else None
    )

    completed: dict[str, dict[str, str]] = {}
    traces: dict[str, dict[str, Any]] = {}
    if resume and paths["submission"].exists():
        previous = load_json(paths["submission"])
        if isinstance(previous, list):
            for raw in previous:
                if isinstance(raw, dict) and "case_id" in raw and "prediction" in raw:
                    completed[str(raw["case_id"])] = {
                        "case_id": str(raw["case_id"]),
                        "prediction": str(raw["prediction"]),
                    }
    if resume and paths["trace"].exists():
        previous_trace = load_json(paths["trace"])
        if isinstance(previous_trace, list):
            for raw in previous_trace:
                if isinstance(raw, dict) and "case_id" in raw:
                    traces[str(raw["case_id"])] = raw

    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        if case_id in completed:
            continue
        print(f"[experiment {info.experiment_id} {index}/{len(cases)}] {case_id}", flush=True)
        if info.mode == "prompt_only":
            result = _run_prompt_only(case, judge)
        else:
            result = _run_architecture_mode(
                case,
                mode=info.mode,
                law_retriever=law_retriever,
                law_candidates=law_candidates,
                judge=judge,
            )
        prediction = str(result["prediction"])
        if prediction not in LABELS:
            raise ValueError(f"Invalid prediction label for {case_id}: {prediction}")
        completed[case_id] = {"case_id": case_id, "prediction": prediction}
        traces[case_id] = {"case_id": case_id, **dict(result["trace"])}
        ordered = [
            completed[str(item["case_id"])]
            for item in cases
            if str(item["case_id"]) in completed
        ]
        ordered_trace = [
            traces[str(item["case_id"])]
            for item in cases
            if str(item["case_id"]) in traces
        ]
        write_json_atomic(paths["submission"], ordered)
        write_json_atomic(paths["trace"], ordered_trace)

    predictions = [
        completed[str(item["case_id"])]
        for item in cases
        if str(item["case_id"]) in completed
    ]
    ordered_traces = [
        traces[str(item["case_id"])]
        for item in cases
        if str(item["case_id"]) in traces
    ]
    metrics = evaluate_outcomes(input_path, paths["submission"])
    fallback_count = sum(1 for trace in ordered_traces if trace.get("fallback_used"))
    llm_error_count = sum(1 for trace in ordered_traces if trace.get("llm_error"))
    metrics = {
        "experiment": {
            "experiment_id": info.experiment_id,
            "type": info.model_type,
            "backbone": info.backbone,
            "params": info.params,
            "domain": info.domain,
            "mode": info.mode,
            "method": METHOD_LABELS[info.mode],
        },
        "fallback_cases": fallback_count,
        "fallback_rate": round(fallback_count / len(cases), 6) if cases else 0.0,
        "llm_error_cases": llm_error_count,
        "llm_error_rate": round(llm_error_count / len(cases), 6) if cases else 0.0,
        **metrics,
    }
    write_json_atomic(paths["metrics"], metrics)
    write_json_atomic(paths["confusion_matrix"], metrics["confusion_matrix"])
    _write_qualitative_files(paths, ordered_traces, input_path, predictions)
    refresh_experiment_tables(result_dir)
    return {
        "experiment": metrics["experiment"],
        "paths": {key: str(path) for key, path in paths.items()},
        "metrics": metrics,
    }
