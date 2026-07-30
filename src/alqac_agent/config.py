from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _integer(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _floating(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings, intentionally loaded without contacting any service."""

    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "Qwen/Qwen3-8B"
    llm_provider: str = "vllm"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 180.0
    llm_max_tokens: int = 2048
    llm_structured_method: str = "json_schema"
    groq_api_key: str = ""
    api_url: str = "https://alqac-api.ngrok.pro/retrieve"
    api_key: str = ""
    api_interval_seconds: float = 5.1
    api_timeout_seconds: float = 30.0
    initial_case_queries: int = 2
    max_case_queries: int = 3
    query_mode: str = "minimal"
    law_candidates: int = 18
    max_law_evidence: int = 8
    few_shots: int = 3
    outcome_knn_k: int = 7
    disposition_confidence: float = 0.65
    cache_path: Path = Path(".cache/case_api.json")

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", defaults.llm_base_url),
            llm_model=os.getenv("LLM_MODEL", defaults.llm_model),
            llm_provider=os.getenv("LLM_PROVIDER", defaults.llm_provider),
            llm_api_key=os.getenv("LLM_API_KEY", defaults.llm_api_key),
            llm_temperature=_floating("LLM_TEMPERATURE", defaults.llm_temperature),
            llm_timeout_seconds=_floating(
                "LLM_TIMEOUT_SECONDS", defaults.llm_timeout_seconds
            ),
            llm_max_tokens=_integer("LLM_MAX_TOKENS", defaults.llm_max_tokens),
            llm_structured_method=os.getenv(
                "LLM_STRUCTURED_METHOD", defaults.llm_structured_method
            ),
            api_url=os.getenv("ALQAC_API_URL", defaults.api_url),
            api_key=os.getenv("ALQAC_API_KEY", ""),
            api_interval_seconds=_floating(
                "ALQAC_API_INTERVAL_SECONDS", defaults.api_interval_seconds
            ),
            api_timeout_seconds=_floating(
                "ALQAC_API_TIMEOUT_SECONDS", defaults.api_timeout_seconds
            ),
            initial_case_queries=_integer(
                "ALQAC_INITIAL_CASE_QUERIES", defaults.initial_case_queries
            ),
            max_case_queries=_integer(
                "ALQAC_MAX_CASE_QUERIES", defaults.max_case_queries
            ),
            query_mode=os.getenv("ALQAC_QUERY_MODE", defaults.query_mode),
            law_candidates=_integer("ALQAC_LAW_CANDIDATES", defaults.law_candidates),
            max_law_evidence=_integer(
                "ALQAC_MAX_LAW_EVIDENCE", defaults.max_law_evidence
            ),
            few_shots=_integer("ALQAC_FEW_SHOTS", defaults.few_shots),
            outcome_knn_k=_integer("ALQAC_OUTCOME_KNN_K", defaults.outcome_knn_k),
            disposition_confidence=_floating(
                "ALQAC_DISPOSITION_CONFIDENCE", defaults.disposition_confidence
            ),
            groq_api_key=os.getenv("GROQ_KEY", os.getenv("GROQ_API_KEY", "")),
            cache_path=Path(os.getenv("ALQAC_CACHE_PATH", str(defaults.cache_path))),
        )

    def validate_for_run(self, *, require_api_key: bool = True) -> None:
        if require_api_key and not self.api_key:
            raise ValueError("Thiếu ALQAC_API_KEY; chưa thể gọi Case Content API.")
        if self.initial_case_queries < 1:
            raise ValueError("ALQAC_INITIAL_CASE_QUERIES phải >= 1")
        if self.max_case_queries < self.initial_case_queries:
            raise ValueError(
                "ALQAC_MAX_CASE_QUERIES phải >= ALQAC_INITIAL_CASE_QUERIES"
            )
        if self.query_mode not in {"minimal", "disposition", "fact"}:
            raise ValueError(
                "ALQAC_QUERY_MODE phải là minimal, disposition hoặc fact"
            )
        if self.llm_structured_method not in {"json_schema", "prompt_json"}:
            raise ValueError(
                "LLM_STRUCTURED_METHOD phải là json_schema hoặc prompt_json"
            )
        if self.llm_provider not in {"vllm", "ollama", "groq"}:
            raise ValueError("LLM_PROVIDER phải là vllm, ollama hoặc groq")
