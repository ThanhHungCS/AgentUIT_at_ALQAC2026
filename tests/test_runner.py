import json

from alqac_agent.config import Settings
from alqac_agent import runner


class _FakeGraph:
    def invoke(self, state):
        return {
            "submission": {
                "case_id": state["case_id"],
                "prediction": "B_WIN",
                "case_evidence": [],
                "law_evidence": [],
                "explanation": "new diagnostic",
            }
        }


class _FakeClient:
    def close(self):
        return None


def test_resume_does_not_reintroduce_explanation(monkeypatch, tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "submission.json"
    input_path.write_text(
        json.dumps(
            [
                {"case_id": "old", "case_query": "q1"},
                {"case_id": "new", "case_query": "q2"},
            ]
        ),
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "old",
                    "prediction": "A_WIN",
                    "case_evidence": [],
                    "law_evidence": [],
                    "explanation": "old diagnostic",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "build_workflow",
        lambda *args, **kwargs: (_FakeGraph(), _FakeClient()),
    )

    runner.run_batch(
        input_path=input_path,
        law_path=tmp_path / "unused-laws.json",
        examples_path=None,
        output_path=output_path,
        settings=Settings(),
        include_explanation=False,
        resume=True,
        case_backend="local-public",
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item["case_id"] for item in saved] == ["old", "new"]
    assert all("explanation" not in item for item in saved)
