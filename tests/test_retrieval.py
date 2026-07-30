from alqac_agent.retrieval import BM25Index, ExampleRetriever


def test_bm25_prefers_matching_document():
    index = BM25Index(
        [
            {"id": 1, "text": "bồi thường thiệt hại do súc vật gây ra"},
            {"id": 2, "text": "hợp đồng chuyển nhượng quyền sử dụng đất"},
        ],
        text_key="text",
    )
    assert index.search("trách nhiệm bồi thường do chó", 1)[0]["id"] == 1


def test_zero_match_returns_empty():
    index = BM25Index([{"id": 1, "text": "đất đai"}], text_key="text")
    assert index.search("xyzabc", 3) == []


def test_public_example_excludes_current_case_label():
    retriever = ExampleRetriever(
        [
            {"case_id": "x", "case_query": "tranh chấp đất", "verdict_label": "A_WIN"},
            {"case_id": "y", "case_query": "tranh chấp đất đai", "verdict_label": "B_WIN"},
        ]
    )
    results = retriever.search("tranh chấp đất", 1, exclude_case_id="x")
    assert [item["case_id"] for item in results] == ["y"]


def test_prior_excludes_current_case():
    retriever = ExampleRetriever(
        [
            {"case_id": "x", "case_query": "tranh chấp đất", "verdict_label": "A_WIN"},
            {"case_id": "y", "case_query": "tranh chấp đất đai", "verdict_label": "B_WIN"},
        ]
    )
    prior = retriever.predict_prior("tranh chấp đất", 1, exclude_case_id="x")
    assert prior["prediction"] == "B_WIN"
    assert prior["neighbors"][0]["case_id"] == "y"


def test_partial_query_prior_excludes_current_case():
    retriever = ExampleRetriever(
        [
            {
                "case_id": "current",
                "case_query": "bồi thường do tai nạn",
                "verdict_label": "PARTIAL_B_WIN",
            },
            {
                "case_id": "near_a",
                "case_query": "bồi thường thiệt hại do tai nạn",
                "verdict_label": "PARTIAL_A_WIN",
            },
            {
                "case_id": "far_b",
                "case_query": "tranh chấp thừa kế đất",
                "verdict_label": "PARTIAL_B_WIN",
            },
        ]
    )
    prior = retriever.predict_partial_prior(
        "bồi thường do tai nạn", exclude_case_id="current", top_k=1
    )
    assert prior["prediction"] == "PARTIAL_A_WIN"
    assert prior["neighbors"][0]["case_id"] == "near_a"
