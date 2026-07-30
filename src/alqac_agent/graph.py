from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .fact_queries import build_fact_queries
from .llm import LegalAgents
from .outcome import (
    DISPOSITION_PROBES,
    apply_area_overlay,
    apply_money_overlay,
    explicit_disposition_label,
    extract_area_request,
    extract_money_request,
    resolve_prediction,
    select_disposition_evidence,
)
from .retrieval import ExampleRetriever, LawRetriever
from .schemas import AgentState, CaseTypeClassification, KeywordExtraction, SubmissionItem, SubmissionLawEvidence


CASE_TYPE_LAW_PRIORITY: dict[str, list[str]] = {
    "land_use_dispute": ["45/2013/QH13", "91/2015/QH13", "92/2015/QH13"],
    "compensation_damage": ["91/2015/QH13", "92/2015/QH13"],
    "inheritance": ["91/2015/QH13", "92/2015/QH13"],
    "credit_contract": ["47/2010/QH12", "91/2015/QH13", "92/2015/QH13"],
    "land_transfer_contract": ["45/2013/QH13", "91/2015/QH13", "92/2015/QH13"],
    "loan_contract": ["91/2015/QH13", "92/2015/QH13"],
    "sale_goods_debt": ["91/2015/QH13", "92/2015/QH13"],
    "service_labor_training": ["91/2015/QH13", "92/2015/QH13"],
    "hui_dispute": ["91/2015/QH13", "92/2015/QH13"],
    "cooperation_contract": ["91/2015/QH13", "92/2015/QH13"],
    "property_claim": ["91/2015/QH13", "92/2015/QH13"],
}

NON_CIVIL_LAW_IDS = {
    "93/2015/QH13",
    "100/2015/QH13",
    "50/2014/QH13",
    "52/2014/QH13",
    "52/2010/QH12",
    "39/2009/QH12",
    "02/2011/QH13",
    "60/2014/QH13",
    "19/2011/NĐ-CP",
    "24/2012/NĐ-CP",
    "66/2014/QH13",
}


def _deduplicate_strings(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def _usable_bm25_query(value: str) -> bool:
    lowered = value.casefold().strip()
    if len(lowered.split()) < 3 or "_" in lowered:
        return False
    forbidden = ("find ", "search ", "locate ", "tìm kiếm ", "hãy tìm ")
    return not lowered.startswith(forbidden)


def _error_summary(error: Exception, limit: int = 420) -> str:
    message = " ".join(str(error).split())
    return message if len(message) <= limit else message[:limit] + "..."


DISPOSITION_FOLLOWUP_QUERIES: tuple[str, str] = (
    "QUYẾT ĐỊNH: Căn cứ vào Điều Nghị quyết Tuyên xử",
)


class ALQACWorkflow:
    def __init__(
        self,
        *,
        settings: Settings,
        agents: LegalAgents,
        case_client: object,
        law_retriever: LawRetriever,
        example_retriever: ExampleRetriever,
    ) -> None:
        self.settings = settings
        self.agents = agents
        self.case_client = case_client
        self.law_retriever = law_retriever
        self.example_retriever = example_retriever

    def plan(self, state: AgentState) -> dict[str, object]:
        if hasattr(self.case_client, "retrieve_all"):
            plan = {
                "dispute_summary": state["case_query"],
                "main_claim": state["case_query"],
                "decisive_issues": ["phạm vi yêu cầu được Tòa chấp nhận"],
                "case_queries": list(DISPOSITION_PROBES),
                "law_query": state["case_query"],
            }
            planned_queries = []
        else:
            try:
                plan_model = self.agents.plan(state["case_query"])
                plan = plan_model.model_dump()
                planned_queries = plan_model.case_queries
            except Exception as error:
                print(
                    f"[warn] planner fallback ({type(error).__name__}): "
                    f"{_error_summary(error)}",
                    flush=True,
                )
                plan = {
                    "dispute_summary": state["case_query"],
                    "main_claim": state["case_query"],
                    "decisive_issues": ["phạm vi yêu cầu được Tòa chấp nhận"],
                    "case_queries": list(DISPOSITION_PROBES),
                    "law_query": state["case_query"],
                }
                planned_queries = []
        money_request = extract_money_request(state["case_query"])
        area_request = (
            None if money_request is not None else extract_area_request(state["case_query"])
        )
        relief_request = money_request or area_request
        fact_queries = build_fact_queries(state["case_query"], max_queries=1)
        dynamic_queries = list(relief_request.followup_queries[:1]) if relief_request else []
        if self.settings.query_mode == "disposition":
            queries = list(DISPOSITION_PROBES)
            query_budget = 2
        elif self.settings.query_mode == "fact":
            queries = fact_queries or [DISPOSITION_PROBES[0]]
            query_budget = 1
        else:
            queries = _deduplicate_strings(
                list(DISPOSITION_PROBES) + dynamic_queries
            )
            query_budget = (
                self.settings.max_case_queries
                if relief_request is not None
                else self.settings.initial_case_queries
            )
        queries = queries[:query_budget]
        return {
            "plan": plan,
            "api_queries": queries,
            "fact_queries": fact_queries,
            "money_signal": money_request.as_dict() if money_request else {},
            "area_signal": area_request.as_dict() if area_request else {},
        }

    def classify_case_type(self, state: AgentState) -> dict[str, object]:
        try:
            result = self.agents.classify_case_type(state["case_query"])
            return {"case_type_info": result.model_dump()}
        except Exception as error:
            print(
                f"[warn] case-type classifier fallback ({type(error).__name__}): "
                f"{_error_summary(error)}",
                flush=True,
            )
            return {
                "case_type_info": {
                    "primary_case_type_id": "unknown",
                    "primary_case_type_text": "",
                    "predicted_difficulty": "medium",
                    "key_facts": [],
                }
            }

    def extract_keywords(self, state: AgentState) -> dict[str, object]:
        try:
            result = self.agents.extract_keywords(state["case_query"])
            return {"keywords": [kw.model_dump() for kw in result.keywords]}
        except Exception as error:
            print(
                f"[warn] keyword extraction fallback ({type(error).__name__}): "
                f"{_error_summary(error)}",
                flush=True,
            )
            return {"keywords": []}

    def _retrieve_queries(
        self,
        state: AgentState,
        queries: list[str],
    ) -> dict[str, object]:
        existing = {
            item["chunk_id"]: item for item in state.get("case_evidence", [])
        }
        executed = list(state.get("api_queries", []))
        for query in queries:
            if len(executed) >= self.settings.max_case_queries:
                break
            hit = self.case_client.retrieve(state["case_id"], query)
            if hit is not None:
                existing[hit.chunk_id] = hit.model_dump()
            executed.append(query)
        return {"case_evidence": list(existing.values()), "api_queries": executed}

    def retrieve_initial(self, state: AgentState) -> dict[str, object]:
        if hasattr(self.case_client, "retrieve_all"):
            all_chunks = self.case_client.retrieve_all(state["case_id"])
            if all_chunks:
                return {
                    "case_evidence": [chunk.model_dump() for chunk in all_chunks],
                    "api_queries": ["preloaded"],
                }
        planned = list(state["api_queries"])
        temporary = dict(state)
        temporary["api_queries"] = []
        return self._retrieve_queries(temporary, planned)

    def retrieve_law_context(self, state: AgentState) -> dict[str, object]:
        plan = state["plan"]
        evidence_text = "\n".join(
            str(item["text"]) for item in state.get("case_evidence", [])
        )
        query = f'{state["case_query"]}\n{plan["law_query"]}\n{evidence_text}'
        candidates = self.law_retriever.search(query, self.settings.law_candidates)

        filtered = [
            item
            for item in candidates
            if str(item.get("law_id", "")) not in NON_CIVIL_LAW_IDS
        ]
        if len(filtered) < 6:
            filtered = candidates

        return {"law_candidates": filtered}

    def assess_disposition(self, state: AgentState) -> dict[str, object]:
        selected = select_disposition_evidence(state["case_evidence"], top_k=4)
        explicit = explicit_disposition_label(state["case_evidence"])
        if explicit is not None:
            scope = {
                "A_WIN": "ALL",
                "PARTIAL_A_WIN": "MAJORITY",
                "PARTIAL_B_WIN": "MINORITY",
                "B_WIN": "NONE",
            }[explicit]
            return {
                "disposition": {
                    "source_quality": "OPERATIVE_ORDER",
                    "court_wording": (
                        "PARTIAL" if explicit.startswith("PARTIAL") else
                        "ALL" if explicit == "A_WIN" else "NONE"
                    ),
                    "claim_scope": scope,
                    "confidence": 1.0,
                    "requested_relief": "",
                    "granted_relief": "",
                    "rejected_relief": "",
                    "decisive_chunk_ids": [item["chunk_id"] for item in selected[:1]],
                    "followup_queries": [],
                }
            }
        try:
            compact_selected = [
                {**item, "text": str(item.get("text", ""))[:500]}
                for item in selected
            ]
            assessment = self.agents.extract_disposition(
                state["case_query"],
                compact_selected,
                state.get("law_candidates", []),
            )
        except Exception as error:
            print(
                f"[warn] disposition fallback ({type(error).__name__}): "
                f"{_error_summary(error)}",
                flush=True,
            )
            return {
                "disposition": {
                    "source_quality": "MISSING",
                    "court_wording": "NOT_EXPLICIT",
                    "claim_scope": "UNCLEAR",
                    "confidence": 0.0,
                    "requested_relief": "",
                    "granted_relief": "",
                    "rejected_relief": "",
                    "decisive_chunk_ids": [],
                    "followup_queries": [],
                }
            }
        allowed = {str(item["chunk_id"]) for item in selected}
        cleaned = assessment.model_dump()
        cleaned["decisive_chunk_ids"] = [
            str(item)
            for item in cleaned["decisive_chunk_ids"]
            if str(item) in allowed
        ]
        if (
            cleaned["source_quality"] in {"MISSING", "PARTY_STATEMENT"}
            or not cleaned["decisive_chunk_ids"]
        ):
            cleaned["source_quality"] = "MISSING"
            cleaned["court_wording"] = "NOT_EXPLICIT"
            cleaned["claim_scope"] = "UNCLEAR"
            cleaned["confidence"] = 0.0
            model_followups = list(cleaned["followup_queries"])
            existing = {str(query).casefold() for query in model_followups}
            cleaned["followup_queries"] = [
                query
                for query in DISPOSITION_FOLLOWUP_QUERIES
                if query.casefold() not in existing
            ]
        return {"disposition": cleaned}

    def route_after_disposition(
        self, state: AgentState
    ) -> Literal["followup", "laws"]:
        room = self.settings.max_case_queries - len(state.get("api_queries", []))
        followups = state["disposition"].get("followup_queries", [])
        return (
            "followup"
            if state["disposition"].get("claim_scope") == "UNCLEAR"
            and room > 0
            and followups
            else "laws"
        )

    def followup(self, state: AgentState) -> dict[str, object]:
        previous = {query.casefold() for query in state.get("api_queries", [])}
        queries = [
            str(query)
            for query in state["disposition"].get("followup_queries", [])
            if str(query).casefold() not in previous
        ]
        return self._retrieve_queries(state, queries)

    def reassess_disposition(self, state: AgentState) -> dict[str, object]:
        return self.assess_disposition(state)

    def laws(self, state: AgentState) -> dict[str, object]:
        candidates = list(state.get("law_candidates", []))
        if not candidates:
            plan = state["plan"]
            evidence_text = "\n".join(
                str(item["text"]) for item in state["case_evidence"]
            )
            query = f'{state["case_query"]}\n{plan["law_query"]}\n{evidence_text}'
            candidates = self.law_retriever.search(query, self.settings.law_candidates)
        try:
            selected = self.agents.select_laws(
                state["case_query"], state["case_evidence"], candidates
            )
            raw_selected = selected.model_dump()["selected"]
        except Exception as error:
            print(
                f"[warn] law-selector fallback ({type(error).__name__}): "
                f"{_error_summary(error)}",
                flush=True,
            )
            raw_selected = []
        allowed = {(str(x["law_id"]), int(x["aid"])): x for x in candidates}
        chosen: list[dict[str, object]] = []
        seen: set[tuple[str, int]] = set()
        for raw in raw_selected:
            try:
                key = (str(raw["law_id"]), int(raw["aid"]))
            except (KeyError, TypeError, ValueError):
                continue
            if key in allowed and key not in seen:
                chosen.append(allowed[key])
                seen.add(key)
        if not chosen:
            chosen = candidates[: min(5, len(candidates))]
        return {
            "law_candidates": candidates,
            "law_evidence": chosen[: self.settings.max_law_evidence],
            "examples": self.example_retriever.search(
                state["case_query"],
                self.settings.few_shots,
                exclude_case_id=state["case_id"],
            ),
        }

    def resolve_outcome(self, state: AgentState) -> dict[str, object]:
        prior = self.example_retriever.predict_prior(
            state["case_query"],
            self.settings.outcome_knn_k,
            exclude_case_id=state["case_id"],
        )
        partial_prior = self.example_retriever.predict_partial_prior(
            state["case_query"],
            exclude_case_id=state["case_id"],
        )
        prior["partial_prediction"] = partial_prior["prediction"]
        prior["partial_scores"] = partial_prior["scores"]
        prior["partial_neighbors"] = partial_prior.get("neighbors", [])
        prediction, source = resolve_prediction(
            assessment=state["disposition"],
            evidence=state["case_evidence"],
            prior=prior,
            confidence_threshold=self.settings.disposition_confidence,
        )
        prediction, money_source = apply_money_overlay(
            prediction,
            state["case_evidence"],
            state.get("money_signal"),
        )
        if money_source:
            source = f"{source} + {money_source}"
        prediction, area_source = apply_area_overlay(
            prediction,
            state["case_evidence"],
            state.get("area_signal"),
        )
        if area_source:
            source = f"{source} + {area_source}"
        selected_laws = [
            {"law_id": item["law_id"], "aid": item["aid"]}
            for item in state["law_evidence"]
        ]
        disposition = state["disposition"]
        explanation = (
            f"Nguồn quyết định: {source}. "
            f"Yêu cầu: {disposition.get('requested_relief', '')} "
            f"Được chấp nhận: {disposition.get('granted_relief', '')} "
            f"Bị bác: {disposition.get('rejected_relief', '')}"
        ).strip()
        decision = {
            "prediction": prediction,
            "explanation": explanation,
            "selected_law_evidence": selected_laws,
            "confidence": disposition.get("confidence", 0.5),
            "source": source,
        }
        return {"outcome_prior": prior, "decision": decision}

    def debate_judge(self, state: AgentState) -> dict[str, object]:
        decision = dict(state.get("decision", {}))
        current_prediction = decision.get("prediction", "PARTIAL_A_WIN")
        current_source = decision.get("source", "")

        if current_source.startswith("operative-order parser"):
            return {}

        evidence = state.get("case_evidence", [])
        laws = state.get("law_evidence", [])
        examples = state.get("examples", [])
        case_query = state["case_query"]

        compact_evidence = [
            {"chunk_id": str(item.get("chunk_id", "")), "text": str(item.get("text", ""))[:600]}
            for item in evidence[:5]
        ]
        compact_laws = [
            {"law_id": str(item.get("law_id", "")), "aid": int(item.get("aid", 0)),
             "content": str(item.get("content", ""))[:400]}
            for item in laws[:4]
        ]
        compact_examples = [
            {"case_id": str(item.get("case_id", "")), "verdict_label": str(item.get("verdict_label", ""))}
            for item in examples[:3]
        ]

        try:
            judge_decision = self.agents.judge(
                case_query,
                compact_evidence,
                compact_laws,
                compact_examples,
                {"side": "A", "proposed_label": current_prediction, "confidence": 0.5, "analysis": ""},
                {"side": "B", "proposed_label": current_prediction, "confidence": 0.5, "analysis": ""},
            )
            judge_label = judge_decision.prediction
            judge_confidence = float(judge_decision.confidence)

            operative_label = explicit_disposition_label(evidence)
            if operative_label is not None:
                final_prediction = operative_label
                final_source = f"operative-order parser (judge confirmed: {judge_label})"
            elif judge_confidence >= 0.7 and judge_label != current_prediction:
                final_prediction = judge_label
                final_source = f"judge (conf={judge_confidence:.2f})"
            else:
                final_prediction = current_prediction
                final_source = current_source + f" + judge (judge={judge_label})"

            decision["prediction"] = final_prediction
            decision["source"] = final_source
            decision["debate_judge_prediction"] = judge_label
            decision["debate_judge_confidence"] = judge_confidence
            return {
                "decision": decision,
            }
        except Exception as error:
            print(
                f"[warn] debate_judge fallback ({type(error).__name__}): "
                f"{_error_summary(error)}",
                flush=True,
            )
            return {}

    def finalize(self, state: AgentState) -> dict[str, object]:
        allowed = {
            (str(item["law_id"]), int(item["aid"])) for item in state["law_evidence"]
        }
        selected: list[SubmissionLawEvidence] = []
        seen: set[tuple[str, int]] = set()
        for raw in state["decision"].get("selected_law_evidence", []):
            try:
                key = (str(raw["law_id"]), int(raw["aid"]))
            except (KeyError, TypeError, ValueError):
                continue
            if key in allowed and key not in seen:
                selected.append(SubmissionLawEvidence(law_id=key[0], aid=key[1]))
                seen.add(key)
        if not selected:
            selected = []
            for item in state["law_evidence"][: self.settings.max_law_evidence]:
                key = (str(item["law_id"]), int(item["aid"]))
                if key not in seen:
                    selected.append(SubmissionLawEvidence(law_id=key[0], aid=key[1]))
                    seen.add(key)
        decisive_ids = [
            str(item)
            for item in state.get("disposition", {}).get("decisive_chunk_ids", [])
        ]
        fact_query_keys = {
            str(query).casefold() for query in state.get("fact_queries", [])
        }
        fact_ids = [
            str(item["chunk_id"])
            for item in state["case_evidence"]
            if str(item.get("query", "")).casefold() in fact_query_keys
        ]
        ranked_case_ids = list(
            dict.fromkeys(
                decisive_ids
                + [
                    str(item["chunk_id"])
                    for item in select_disposition_evidence(
                        state["case_evidence"],
                        top_k=1,
                    )
                ]
                + fact_ids[:1]
            )
        )
        item = SubmissionItem(
            case_id=state["case_id"],
            prediction=state["decision"]["prediction"],
            case_evidence=ranked_case_ids[:2],
            law_evidence=selected,
        )
        return {"submission": item.model_dump(exclude_none=True)}

    def compile(self):
        builder = StateGraph(AgentState)
        builder.add_node("plan", self.plan)
        builder.add_node("retrieve_initial", self.retrieve_initial)
        builder.add_node("retrieve_law_context", self.retrieve_law_context)
        builder.add_node("assess_disposition", self.assess_disposition)
        builder.add_node("followup", self.followup)
        builder.add_node("reassess_disposition", self.reassess_disposition)
        builder.add_node("laws", self.laws)
        builder.add_node("resolve_outcome", self.resolve_outcome)
        builder.add_node("debate_judge", self.debate_judge)
        builder.add_node("finalize", self.finalize)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "retrieve_initial")
        builder.add_edge("retrieve_initial", "retrieve_law_context")
        builder.add_edge("retrieve_law_context", "assess_disposition")
        builder.add_conditional_edges(
            "assess_disposition", self.route_after_disposition
        )
        builder.add_edge("followup", "reassess_disposition")
        builder.add_edge("reassess_disposition", "laws")
        builder.add_edge("laws", "resolve_outcome")
        builder.add_edge("resolve_outcome", "debate_judge")
        builder.add_edge("debate_judge", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()
