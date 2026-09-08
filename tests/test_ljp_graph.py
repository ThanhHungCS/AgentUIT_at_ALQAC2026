from __future__ import annotations

from alqac_agent.ljp_graph import ZeroShotLJPWorkflow
from alqac_agent.ljp_input import process_case_fact
from alqac_agent.ljp_reasoning import heuristic_disposition
from alqac_agent.ljp_retrieval import build_law_query


class DummyLawRetriever:
    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        assert "bồi thường" in query.casefold()
        return [
            {
                "law_id": "91/2015/QH13",
                "aid": 584,
                "content": "Người gây thiệt hại phải bồi thường.",
                "score": 2.0,
            },
            {
                "law_id": "92/2015/QH13",
                "aid": 147,
                "content": "Đương sự có nghĩa vụ chứng minh yêu cầu của mình.",
                "score": 0.8,
            },
        ][:top_k]


def test_input_processing_extracts_claim_and_disposition_candidates() -> None:
    processed = process_case_fact(
        "case_x",
        """
        Nguyên đơn yêu cầu bị đơn bồi thường thiệt hại ngoài hợp đồng.
        Bị đơn không chấp nhận yêu cầu của nguyên đơn.
        Kiểm sát viên phát biểu quan điểm giải quyết vụ án: Đề nghị Hội đồng xét xử
        chấp nhận một phần yêu cầu khởi kiện của nguyên đơn.
        """,
    )

    assert "bồi thường" in processed["main_claim"]["main_claim"].casefold()
    assert processed["disposition_candidates"]


def test_legal_retrieval_query_uses_processed_case_fact() -> None:
    processed = process_case_fact(
        "case_x",
        "Nguyên đơn khởi kiện tranh chấp bồi thường thiệt hại ngoài hợp đồng.",
    )
    query = build_law_query(processed)

    assert "bồi thường" in query.casefold()


def test_reasoning_maps_disposition_to_prediction_scope() -> None:
    processed = process_case_fact(
        "case_x",
        "Ý kiến về việc giải quyết vụ án: Đề nghị Hội đồng xét xử "
        "không chấp nhận yêu cầu khởi kiện của nguyên đơn.",
    )
    assessment = heuristic_disposition(processed)

    assert assessment["claim_scope"] == "NONE"


def test_zero_shot_ljp_workflow_outputs_prediction_only() -> None:
    workflow = ZeroShotLJPWorkflow(
        law_retriever=DummyLawRetriever(),  # type: ignore[arg-type]
        judge=None,
        law_candidates=2,
    ).compile()

    state = workflow.invoke(
        {
            "case_id": "case_x",
            "case_fact": (
                "Nguyên đơn yêu cầu bị đơn bồi thường thiệt hại ngoài hợp đồng. "
                "Ý kiến về việc giải quyết vụ án: Đề nghị Hội đồng xét xử "
                "không chấp nhận yêu cầu khởi kiện của nguyên đơn."
            ),
            "metadata": {},
        }
    )

    assert state["prediction_item"] == {"case_id": "case_x", "prediction": "B_WIN"}
    assert "case_evidence" not in state["prediction_item"]
    assert "law_evidence" not in state["prediction_item"]
