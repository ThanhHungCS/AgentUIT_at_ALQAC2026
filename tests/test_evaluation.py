import json

from alqac_agent.evaluation import evaluate_outcomes, prepare_public_input


def test_prepare_and_evaluate_public(tmp_path):
    source = tmp_path / "public.json"
    inputs = tmp_path / "inputs.json"
    predictions = tmp_path / "predictions.json"
    source.write_text(
        json.dumps(
            [
                {"case_id": "x", "case_query": "q1", "verdict_label": "A_WIN", "court_verdict": "secret"},
                {"case_id": "y", "case_query": "q2", "verdict_label": "B_WIN", "court_verdict": "secret"},
            ]
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            [
                {"case_id": "x", "prediction": "A_WIN", "case_evidence": [], "law_evidence": []},
                {"case_id": "y", "prediction": "A_WIN", "case_evidence": [], "law_evidence": []},
            ]
        ),
        encoding="utf-8",
    )
    prepared = prepare_public_input(source, inputs)
    assert prepared == [
        {"case_id": "x", "case_query": "q1"},
        {"case_id": "y", "case_query": "q2"},
    ]
    metrics = evaluate_outcomes(source, predictions)
    assert metrics["accuracy"] == 0.5
    assert metrics["coverage"] == 1.0


def test_evaluate_minimal_ljp_predictions(tmp_path):
    source = tmp_path / "public.json"
    predictions = tmp_path / "predictions.json"
    source.write_text(
        json.dumps(
            [
                {"case_id": "x", "case_query": "q1", "verdict_label": "A_WIN"},
                {"case_id": "y", "case_query": "q2", "verdict_label": "B_WIN"},
            ]
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            [
                {"case_id": "x", "prediction": "A_WIN"},
                {"case_id": "y", "prediction": "B_WIN"},
            ]
        ),
        encoding="utf-8",
    )

    metrics = evaluate_outcomes(source, predictions)

    assert metrics["accuracy"] == 1.0
    assert metrics["coverage"] == 1.0
