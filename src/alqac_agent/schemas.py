from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VerdictLabel = Literal["A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaseInput(StrictModel):
    case_id: str
    case_query: str


class SearchPlan(StrictModel):
    dispute_summary: str
    main_claim: str
    decisive_issues: list[str] = Field(min_length=1, max_length=6)
    case_queries: list[str] = Field(min_length=1, max_length=8)
    law_query: str

    @field_validator("case_queries")
    @classmethod
    def nonempty_queries(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = " ".join(value.split())
            if value and value.casefold() not in {x.casefold() for x in cleaned}:
                cleaned.append(value)
        if not cleaned:
            raise ValueError("case_queries rỗng")
        return cleaned


class CaseEvidence(StrictModel):
    chunk_id: str
    text: str
    score: float = 0.0
    query: str = ""
    cached: bool = False


class EvidenceAudit(StrictModel):
    sufficient: bool
    missing_information: list[str] = Field(default_factory=list, max_length=4)
    followup_queries: list[str] = Field(default_factory=list, max_length=4)


class DispositionAssessment(StrictModel):
    source_quality: Literal[
        "OPERATIVE_ORDER", "COURT_REASONING", "PARTY_STATEMENT", "MISSING"
    ] = "MISSING"
    court_wording: Literal["ALL", "PARTIAL", "NONE", "NOT_EXPLICIT"] = (
        "NOT_EXPLICIT"
    )
    claim_scope: Literal["ALL", "MAJORITY", "MINORITY", "NONE", "UNCLEAR"] = (
        "UNCLEAR"
    )
    confidence: float = Field(default=0.5, ge=0, le=1)
    requested_relief: str = ""
    granted_relief: str = ""
    rejected_relief: str = ""
    decisive_chunk_ids: list[str] = Field(default_factory=list, max_length=6)
    followup_queries: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().removesuffix("%").strip()
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value
        return number / 100 if 1 < number <= 100 else number


class LawEvidence(StrictModel):
    law_id: str
    aid: int
    content: str = ""
    score: float = 0.0


class LawIdentifier(StrictModel):
    law_id: str
    aid: int


class LawSelection(StrictModel):
    selected: list[LawIdentifier] = Field(default_factory=list, max_length=12)


class AdvocateOpinion(StrictModel):
    side: Literal["A", "B"]
    proposed_label: VerdictLabel = "A_WIN"
    confidence: float = Field(default=0.5, ge=0, le=1)
    strongest_facts: list[str] = Field(default_factory=list, max_length=6)
    applicable_rules: list[str] = Field(default_factory=list, max_length=6)
    analysis: str = ""

    @model_validator(mode="before")
    @classmethod
    def supply_missing_advocacy_label(cls, value: object) -> object:
        if isinstance(value, dict) and "proposed_label" not in value:
            value = dict(value)
            value["proposed_label"] = "A_WIN" if value.get("side") == "A" else "B_WIN"
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> object:
        """Accept common model output in either 0..1 or percentage 0..100."""
        if isinstance(value, str):
            value = value.strip().removesuffix("%").strip()
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value
        return number / 100 if 1 < number <= 100 else number

class JudgeDecision(StrictModel):
    prediction: VerdictLabel
    explanation: str = ""
    selected_law_evidence: list[LawIdentifier] = Field(
        default_factory=list, max_length=12
    )
    confidence: float = Field(default=0.5, ge=0, le=1)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().removesuffix("%").strip()
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value
        return number / 100 if 1 < number <= 100 else number


class SubmissionLawEvidence(LawIdentifier):
    pass


class SubmissionItem(StrictModel):
    case_id: str
    prediction: VerdictLabel
    case_evidence: list[str]
    law_evidence: list[SubmissionLawEvidence]
    explanation: str | None = None


class CaseTypeClassification(StrictModel):
    primary_case_type_id: str
    primary_case_type_text: str
    predicted_difficulty: Literal["easy", "medium", "hard"] = "medium"
    key_facts: list[str] = Field(default_factory=list, max_length=6)


class KeywordItem(StrictModel):
    keyword_id: str
    type: str
    keyword_text: str


class KeywordExtraction(StrictModel):
    keywords: list[KeywordItem] = Field(default_factory=list, max_length=20)


class AgentState(TypedDict, total=False):
    case_id: str
    case_query: str
    plan: dict[str, object]
    case_type_info: dict[str, object]
    keywords: list[dict[str, object]]
    case_evidence: list[dict[str, object]]
    api_queries: list[str]
    fact_queries: list[str]
    audit: dict[str, object]
    disposition: dict[str, object]
    outcome_prior: dict[str, object]
    money_signal: dict[str, object]
    area_signal: dict[str, object]
    law_candidates: list[dict[str, object]]
    law_evidence: list[dict[str, object]]
    examples: list[dict[str, object]]
    advocate_a: dict[str, object]
    advocate_b: dict[str, object]
    decision: dict[str, object]
    submission: dict[str, object]
