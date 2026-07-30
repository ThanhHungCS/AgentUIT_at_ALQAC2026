import json

import pytest

from alqac_agent.local_case_api import LocalCaseContentClient, chunk_judgment


def test_chunk_judgment_has_overlap():
    chunks = chunk_judgment(" ".join(str(i) for i in range(12)), window_words=5, overlap_words=2)
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_local_case_api_is_scoped_by_case_id(tmp_path):
    corpus = tmp_path / "public.json"
    corpus.write_text(
        json.dumps(
            [
                {
                    "case_id": "case_a",
                    "judgment_text": "Tòa án tuyên chấp nhận toàn bộ yêu cầu nguyên đơn.",
                    "verdict_label": "A_WIN",
                    "court_verdict": "gold must not be indexed separately",
                },
                {
                    "case_id": "case_b",
                    "judgment_text": "Tòa án bác toàn bộ yêu cầu nguyên đơn.",
                    "verdict_label": "B_WIN",
                },
            ]
        ),
        encoding="utf-8",
    )
    client = LocalCaseContentClient(corpus, window_words=8, overlap_words=2)
    hit = client.retrieve("case_a", "chấp nhận toàn bộ")
    assert hit is not None
    assert hit.chunk_id.startswith("case_a_")
    assert "chấp nhận" in hit.text
    with pytest.raises(KeyError):
        client.retrieve("case_unknown", "tuyên xử")
