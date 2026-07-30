from alqac_agent.config import Settings
from alqac_agent.graph import ALQACWorkflow
from alqac_agent.retrieval import ExampleRetriever, LawRetriever
from alqac_agent.schemas import (
    CaseEvidence,
    DispositionAssessment,
    LawSelection,
    SearchPlan,
)


class FakeCaseClient:
    def retrieve(self, case_id, query):
        return CaseEvidence(
            chunk_id=f"{case_id}_chunk_1",
            text="Tuyên xử chấp nhận toàn bộ yêu cầu khởi kiện của nguyên đơn.",
            score=1.0,
            query=query,
        )


class FakeAgents:
    def plan(self, case_query):
        return SearchPlan(
            dispute_summary="tranh chấp",
            main_claim="công nhận yêu cầu",
            decisive_issues=["hiệu lực"],
            case_queries=["tuyên xử"],
            law_query="quy định hiệu lực",
        )

    def extract_disposition(self, case_query, evidence, laws=None):
        return DispositionAssessment(
            source_quality="OPERATIVE_ORDER",
            court_wording="ALL",
            claim_scope="ALL",
            confidence=0.9,
            decisive_chunk_ids=[evidence[0]["chunk_id"]],
        )

    def select_laws(self, case_query, case_evidence, candidates):
        return LawSelection(selected=[{"law_id": "L", "aid": 1}])

def test_graph_runs_and_keeps_only_retrieved_law():
    workflow = ALQACWorkflow(
        settings=Settings(initial_case_queries=1, max_case_queries=1),
        agents=FakeAgents(),
        case_client=FakeCaseClient(),
        law_retriever=LawRetriever([{"law_id": "L", "aid": 1, "content": "quy định hiệu lực"}]),
        example_retriever=ExampleRetriever([]),
    ).compile()
    result = workflow.invoke(
        {"case_id": "case_x", "case_query": "quy định hiệu lực"}
    )
    assert result["submission"]["prediction"] == "A_WIN"
    assert result["submission"]["case_evidence"] == ["case_x_chunk_1"]
    assert result["submission"]["law_evidence"] == [{"law_id": "L", "aid": 1}]


class FailingOptionalAgents(FakeAgents):
    def plan(self, case_query):
        raise ConnectionError("LLM is temporarily unavailable")

    def select_laws(self, case_query, case_evidence, candidates):
        raise ConnectionError("LLM is temporarily unavailable")


def test_graph_falls_back_when_optional_llm_calls_fail():
    workflow = ALQACWorkflow(
        settings=Settings(initial_case_queries=1, max_case_queries=1),
        agents=FailingOptionalAgents(),
        case_client=FakeCaseClient(),
        law_retriever=LawRetriever(
            [{"law_id": "L", "aid": 1, "content": "quy định hiệu lực"}]
        ),
        example_retriever=ExampleRetriever([]),
    ).compile()
    result = workflow.invoke(
        {"case_id": "case_x", "case_query": "quy định hiệu lực"}
    )
    assert result["submission"]["prediction"] == "A_WIN"
    assert result["submission"]["law_evidence"] == [{"law_id": "L", "aid": 1}]
