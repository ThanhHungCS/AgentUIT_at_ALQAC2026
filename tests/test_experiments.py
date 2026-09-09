from __future__ import annotations

import json

from alqac_agent.experiments import ExperimentInfo, run_ljp_experiment
from alqac_agent.schemas import LJPLabelPrediction


class DummyPromptJudge:
    def predict_ljp_label(
        self,
        case_fact: str,
        laws: list[dict[str, object]] | None = None,
        processed_input: dict[str, object] | None = None,
    ) -> LJPLabelPrediction:
        return LJPLabelPrediction(
            prediction="A_WIN",
            confidence=0.9,
            explanation="Dummy prompt-only decision.",
        )


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_prompt_only_experiment_writes_result_layout(tmp_path) -> None:
    benchmark = tmp_path / "public.json"
    laws = tmp_path / "laws.json"
    result_dir = tmp_path / "result"
    _write_json(
        benchmark,
        [
            {
                "case_id": "case_x",
                "case_fact": "Nguyên đơn yêu cầu bị đơn thanh toán toàn bộ khoản nợ.",
                "verdict_label": "A_WIN",
            }
        ],
    )
    _write_json(laws, [])

    summary = run_ljp_experiment(
        input_path=benchmark,
        law_path=laws,
        result_dir=result_dir,
        info=ExperimentInfo(
            experiment_id="dummy_prompt",
            model_type="General Domain",
            backbone="Dummy",
            params="1B",
            domain="General",
            mode="prompt_only",
        ),
        judge=DummyPromptJudge(),  # type: ignore[arg-type]
        resume=False,
    )

    assert summary["metrics"]["accuracy"] == 1.0
    assert (result_dir / "submissions" / "dummy_prompt.json").exists()
    assert (result_dir / "metrics" / "dummy_prompt.metrics.json").exists()
    assert (result_dir / "traces" / "dummy_prompt.trace.json").exists()
    assert (result_dir / "qualitative" / "dummy_prompt.input_processing_outputs.json").exists()
    assert (result_dir / "tables" / "main_comparison.csv").exists()


def test_method_experiment_writes_retrieval_outputs(tmp_path) -> None:
    benchmark = tmp_path / "public.json"
    laws = tmp_path / "laws.json"
    result_dir = tmp_path / "result"
    _write_json(
        benchmark,
        [
            {
                "case_id": "case_x",
                "case_fact": (
                    "Nguyên đơn yêu cầu bị đơn bồi thường thiệt hại. "
                    "Hội đồng xét xử chấp nhận yêu cầu khởi kiện của nguyên đơn."
                ),
                "verdict_label": "A_WIN",
            }
        ],
    )
    _write_json(
        laws,
        [
            {
                "law_id": "91/2015/QH13",
                "content": [
                    {
                        "aid": 1,
                        "content_Article": "Người gây thiệt hại phải bồi thường.",
                    }
                ],
            }
        ],
    )

    run_ljp_experiment(
        input_path=benchmark,
        law_path=laws,
        result_dir=result_dir,
        info=ExperimentInfo(
            experiment_id="dummy_method",
            model_type="General Domain",
            backbone="Dummy",
            params="1B",
            domain="General",
            mode="method",
        ),
        judge=None,
        resume=False,
    )

    retrieval = json.loads(
        (result_dir / "qualitative" / "dummy_method.retrieval_outputs.json").read_text(
            encoding="utf-8"
        )
    )
    assert retrieval[0]["retrieved_laws"][0]["law_id"] == "91/2015/QH13"
