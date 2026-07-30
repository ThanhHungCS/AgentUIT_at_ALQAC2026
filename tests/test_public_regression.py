import hashlib
import json
from pathlib import Path

from alqac_agent.local_case_api import LocalCaseContentClient
from alqac_agent.outcome import (
    DISPOSITION_PROBES,
    apply_area_overlay,
    apply_money_overlay,
    extract_area_request,
    extract_money_request,
    resolve_prediction,
)
from alqac_agent.retrieval import ExampleRetriever


ROOT = Path(__file__).resolve().parents[1]


def test_public_leave_one_out_outcome_regression():
    """Replay the outcome head without any LLM/network call."""

    public_path = ROOT / "ALQAC2026_public_test.json"
    cases = json.loads(public_path.read_text(encoding="utf-8"))
    client = LocalCaseContentClient(public_path)
    examples = ExampleRetriever(cases)
    predictions: list[str] = []

    for case in cases:
        money = extract_money_request(case["case_query"])
        area = None if money else extract_area_request(case["case_query"])
        relief = money or area
        queries = list(DISPOSITION_PROBES)
        if relief:
            queries.extend(relief.followup_queries)
        evidence: list[dict[str, object]] = []
        seen: set[str] = set()
        for query in queries:
            hit = client.retrieve(case["case_id"], query)
            if hit is not None and hit.chunk_id not in seen:
                evidence.append(hit.model_dump())
                seen.add(hit.chunk_id)

        prior = examples.predict_prior(
            case["case_query"], 7, exclude_case_id=case["case_id"]
        )
        partial = examples.predict_partial_prior(
            case["case_query"], exclude_case_id=case["case_id"]
        )
        prior["partial_prediction"] = partial["prediction"]
        prediction, _ = resolve_prediction(
            assessment={},
            evidence=evidence,
            prior=prior,
            confidence_threshold=0.65,
        )
        prediction, _ = apply_money_overlay(
            prediction, evidence, money.as_dict() if money else None
        )
        prediction, _ = apply_area_overlay(
            prediction, evidence, area.as_dict() if area else None
        )
        predictions.append(prediction)

    accuracy = sum(
        prediction == case["verdict_label"]
        for prediction, case in zip(predictions, cases)
    ) / len(cases)
    checksum_input = "".join(
        f"{case['case_id']}={prediction}\n"
        for case, prediction in zip(cases, predictions)
    )
    assert accuracy == 0.78
    assert hashlib.sha256(checksum_input.encode()).hexdigest() == (
        "3c2e6c3e99ba8db1f5f2e7ab31832d8da3cf57b57bd750241174c47a8894bfbb"
    )
