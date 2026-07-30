import json

import pytest

from alqac_agent.data import load_cases, load_law_articles
from alqac_agent.schemas import AdvocateOpinion, JudgeDecision


def test_duplicate_cases_rejected(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {"case_id": "x", "case_query": "q"},
                {"case_id": "x", "case_query": "q2"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trùng"):
        load_cases(path)


def test_law_schema(tmp_path):
    path = tmp_path / "laws.json"
    path.write_text(
        json.dumps(
            [{"law_id": "L", "content": [{"aid": 7, "content_Article": "Nội dung"}]}]
        ),
        encoding="utf-8",
    )
    assert load_law_articles(path) == [{"law_id": "L", "aid": 7, "content": "Nội dung"}]


def test_confidence_percentage_is_normalized():
    advocate = AdvocateOpinion(
        side="A",
        analysis="Lập luận",
        proposed_label="A_WIN",
        confidence=98,
    )
    judge = JudgeDecision(
        prediction="A_WIN",
        explanation="Kết luận",
        confidence="85%",
    )
    assert advocate.confidence == 0.98
    assert judge.confidence == 0.85


def test_confidence_can_be_omitted_by_model():
    advocate = AdvocateOpinion(
        side="B",
        analysis="Lập luận",
        proposed_label="B_WIN",
    )
    judge = JudgeDecision(
        prediction="B_WIN",
        explanation="Bác yêu cầu",
    )
    assert advocate.confidence == 0.5
    assert judge.confidence == 0.5


def test_advocate_missing_label_defaults_to_its_side():
    advocate_a = AdvocateOpinion(side="A", analysis="Lập luận")
    advocate_b = AdvocateOpinion(side="B", analysis="Lập luận")
    assert advocate_a.proposed_label == "A_WIN"
    assert advocate_b.proposed_label == "B_WIN"
