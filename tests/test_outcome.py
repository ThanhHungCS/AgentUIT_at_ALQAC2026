from alqac_agent.outcome import (
    apply_area_overlay,
    apply_money_overlay,
    explicit_disposition_label,
    extract_area_request,
    extract_money_request,
    resolve_prediction,
)


def evidence(text):
    return [{"chunk_id": "c1", "text": text}]


def test_explicit_partial_order_wins_over_noisy_prior():
    result, source = resolve_prediction(
        assessment={
            "claim_scope": "MAJORITY",
            "source_quality": "OPERATIVE_ORDER",
            "confidence": 0.9,
            "decisive_chunk_ids": ["c1"],
        },
        evidence=evidence("QUYẾT ĐỊNH: Chấp nhận một phần yêu cầu khởi kiện."),
        prior={"prediction": "A_WIN"},
        confidence_threshold=0.65,
    )
    assert result == "PARTIAL_A_WIN"
    assert source == "operative-order parser"


def test_grounded_minority_refines_partial_wording():
    result, _ = resolve_prediction(
        assessment={
            "claim_scope": "MINORITY",
            "source_quality": "OPERATIVE_ORDER",
            "confidence": 0.8,
            "decisive_chunk_ids": ["c1"],
        },
        evidence=evidence("QUYẾT ĐỊNH: Chấp nhận một phần yêu cầu khởi kiện."),
        prior={"prediction": "PARTIAL_A_WIN"},
        confidence_threshold=0.65,
    )
    assert result == "PARTIAL_B_WIN"


def test_ungrounded_assessment_falls_back_to_prior():
    result, source = resolve_prediction(
        assessment={
            "claim_scope": "ALL",
            "source_quality": "OPERATIVE_ORDER",
            "confidence": 0.99,
            "decisive_chunk_ids": ["hallucinated"],
        },
        evidence=evidence("Nội dung chưa nêu quyết định của Tòa."),
        prior={
            "prediction": "B_WIN",
            "neighbors": [{"case_id": "case_neighbor"}],
        },
        confidence_threshold=0.65,
    )
    assert result == "B_WIN"
    assert source == "BM25 query kNN prior"


def test_reject_order_detected():
    assert (
        explicit_disposition_label(
            evidence("Vì các lẽ trên, QUYẾT ĐỊNH: Không chấp nhận yêu cầu khởi kiện.")
        )
        == "B_WIN"
    )


def test_vks_proposal_is_not_treated_as_court_order():
    assert (
        explicit_disposition_label(
            evidence(
                "Đại diện Viện kiểm sát đề nghị Hội đồng xét xử "
                "chấp nhận yêu cầu khởi kiện của nguyên đơn."
            )
        )
        is None
    )


def test_vks_proposal_after_xet_xu_colon_is_not_a_court_order():
    assert (
        explicit_disposition_label(
            evidence(
                "Ý kiến về việc giải quyết vụ án: Đề nghị Hội đồng xét xử: "
                "Chấp nhận một phần yêu cầu khởi kiện của nguyên đơn."
            )
        )
        is None
    )


def test_final_order_overrides_earlier_vks_proposal():
    assert explicit_disposition_label(
        evidence(
            "Viện kiểm sát đề nghị chấp nhận toàn bộ yêu cầu. "
            "Vì các lẽ trên, QUYẾT ĐỊNH: Không chấp nhận yêu cầu "
            "khởi kiện của nguyên đơn."
        )
    ) == "B_WIN"


def test_money_overlay_demotes_full_win_when_award_is_smaller():
    signal = extract_money_request(
        "Nguyên đơn yêu cầu bị đơn thanh toán 86.400.000 đồng."
    )
    assert signal is not None
    result, source = apply_money_overlay(
        "A_WIN",
        evidence("QUYẾT ĐỊNH: Buộc bị đơn thanh toán 84.800.000 đồng."),
        signal.as_dict(),
    )
    assert result == "PARTIAL_A_WIN"
    assert source == "requested-versus-ordered monetary relief"


def test_area_overlay_refines_partial_to_minority():
    signal = extract_area_request(
        "Nguyên đơn yêu cầu trả lại 2.500m2 thuộc thửa đất số 14."
    )
    assert signal is not None
    dynamic_query = signal.followup_queries[0]
    result, source = apply_area_overlay(
        "PARTIAL_A_WIN",
        [
            {
                "chunk_id": "c1",
                "query": dynamic_query,
                "text": "Tuyên xử chia cho nguyên đơn phần đất có diện tích 150m2.",
            }
        ],
        signal.as_dict(),
    )
    assert result == "PARTIAL_B_WIN"
    assert source == "minority of requested land area"


def test_area_overlay_accepts_nearly_identical_survey_area_as_full():
    signal = extract_area_request(
        "Nguyên đơn yêu cầu công nhận khoảng 10.083m2 đất."
    )
    assert signal is not None
    result, source = apply_area_overlay(
        "PARTIAL_A_WIN",
        [
            {
                "chunk_id": "c1",
                "query": signal.followup_queries[1],
                "text": (
                    "QUYẾT ĐỊNH: Nguyên đơn được quyền quản lý, "
                    "sử dụng diện tích đất 10.038m2."
                ),
            }
        ],
        signal.as_dict(),
    )
    assert result == "A_WIN"
    assert source == "substantially full plaintiff land award"
