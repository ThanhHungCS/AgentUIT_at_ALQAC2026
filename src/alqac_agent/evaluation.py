from __future__ import annotations

from collections import Counter
from pathlib import Path

from .data import load_json, write_json_atomic
from .schemas import SubmissionItem, VerdictLabel

LABELS: tuple[VerdictLabel, ...] = (
    "A_WIN",
    "PARTIAL_A_WIN",
    "PARTIAL_B_WIN",
    "B_WIN",
)


def prepare_public_input(source: str | Path, output: str | Path) -> list[dict[str, str]]:
    """Strip all gold/full-judgment fields so public input matches private input."""
    raw = load_json(source)
    inputs = [
        {"case_id": str(item["case_id"]), "case_query": str(item["case_query"])}
        for item in raw
    ]
    ids = [item["case_id"] for item in inputs]
    if len(ids) != len(set(ids)):
        raise ValueError("Public source có case_id trùng")
    write_json_atomic(output, inputs)
    return inputs


def evaluate_outcomes(
    gold_path: str | Path, predictions_path: str | Path
) -> dict[str, object]:
    """Evaluate only the outcome component available in the distributed public JSON."""
    gold_raw = load_json(gold_path)
    prediction_raw = load_json(predictions_path)
    gold = {str(item["case_id"]): str(item["verdict_label"]) for item in gold_raw}
    predictions: dict[str, str] = {}
    for raw in prediction_raw:
        item = SubmissionItem.model_validate(raw)
        if item.case_id in predictions:
            raise ValueError(f"Prediction trùng case_id: {item.case_id}")
        predictions[item.case_id] = item.prediction

    unknown = sorted(set(predictions) - set(gold))
    if unknown:
        raise ValueError(f"Prediction chứa case_id không có trong gold: {unknown[:5]}")
    evaluated_ids = [case_id for case_id in gold if case_id in predictions]
    missing = [case_id for case_id in gold if case_id not in predictions]
    confusion: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
    correct = 0
    for case_id in evaluated_ids:
        expected = gold[case_id]
        predicted = predictions[case_id]
        if expected not in confusion:
            raise ValueError(f"Gold label không hợp lệ: {expected}")
        confusion[expected][predicted] += 1
        correct += int(expected == predicted)

    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        true_positive = confusion[label][label]
        false_negative = sum(confusion[label].values()) - true_positive
        false_positive = sum(confusion[other][label] for other in LABELS) - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "support": sum(confusion[label].values()),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    count = len(evaluated_ids)
    return {
        "gold_cases": len(gold),
        "evaluated_cases": count,
        "missing_cases": missing,
        "coverage": round(count / len(gold), 6) if gold else 0.0,
        "accuracy": round(correct / count, 6) if count else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "per_label": per_label,
        "confusion_matrix": {
            "rows_are_gold": list(LABELS),
            "columns_are_prediction": list(LABELS),
            "values": [
                [confusion[gold_label][predicted] for predicted in LABELS]
                for gold_label in LABELS
            ],
        },
    }
