from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "evolution_query_optimizer.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


STOPWORDS = {
    "agent",
    "ai",
    "anh",
    "ba",
    "ban",
    "bang",
    "bi",
    "cac",
    "cho",
    "co",
    "con",
    "cua",
    "du",
    "duoc",
    "don",
    "doan",
    "hay",
    "kien",
    "la",
    "mot",
    "nay",
    "nguyen",
    "nhung",
    "ong",
    "tai",
    "theo",
    "thang",
    "thi",
    "trong",
    "va",
    "ve",
    "voi",
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    no_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return no_marks.replace("đ", "d").replace("Đ", "D")


def norm_text(text: str) -> str:
    return normalize_space(strip_accents(text).lower())


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", norm_text(text))
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def api_efficiency(num_segments: int, api_calls: int) -> float:
    if num_segments <= 0:
        return 0.0
    if api_calls <= 2 * num_segments:
        return 1.0
    if api_calls >= 5 * num_segments:
        return 0.0
    return 1.0 - ((api_calls - 2 * num_segments) / (3 * num_segments))


def case_recall(predicted: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(predicted & gold) / len(gold)


def parse_slot(selector: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"([a-z]+)(\d+)", selector)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def parse_corpus_selector(selector: str) -> int | None:
    match = re.fullmatch(r"corpus:(\d+)", selector)
    return int(match.group(1)) if match else None


class BM25Retriever:
    def __init__(self, corpus_by_case: dict[str, dict[str, Any]]) -> None:
        self.corpus_by_case = corpus_by_case
        self.idf = self._build_idf()
        self.chunk_token_counts: dict[str, Counter[str]] = {}
        self.chunk_lengths: dict[str, int] = {}
        total_len = 0
        total_chunks = 0
        for case in corpus_by_case.values():
            for chunk in case.get("chunks", []):
                chunk_id = chunk["chunk_id"]
                tokens = tokenize(chunk.get("text", ""))
                self.chunk_token_counts[chunk_id] = Counter(tokens)
                self.chunk_lengths[chunk_id] = len(tokens)
                total_len += len(tokens)
                total_chunks += 1
        self.avg_len = max(1.0, total_len / max(1, total_chunks))

    def _build_idf(self) -> dict[str, float]:
        doc_freq: Counter[str] = Counter()
        total_docs = 0
        for case in self.corpus_by_case.values():
            for chunk in case.get("chunks", []):
                total_docs += 1
                doc_freq.update(set(tokenize(chunk.get("text", ""))))
        return {
            token: math.log((total_docs + 1) / (freq + 0.5)) + 1.0
            for token, freq in doc_freq.items()
        }

    def score_chunk(self, query: str, chunk_id: str) -> float:
        query_tokens = tokenize(query)
        counts = self.chunk_token_counts.get(chunk_id, Counter())
        if not query_tokens or not counts:
            return 0.0
        chunk_len = self.chunk_lengths.get(chunk_id, 0)
        k1 = 1.4
        b = 0.75
        score = 0.0
        for token in set(query_tokens):
            tf = counts.get(token, 0)
            if tf <= 0:
                continue
            denom = tf + k1 * (1 - b + b * chunk_len / self.avg_len)
            score += self.idf.get(token, 1.0) * (tf * (k1 + 1)) / denom
        return score

    def retrieve(self, case_id: str, query: str, top_k: int) -> set[str]:
        scored: list[tuple[float, str]] = []
        for chunk in self.corpus_by_case[case_id].get("chunks", []):
            chunk_id = chunk["chunk_id"]
            score = self.score_chunk(query, chunk_id)
            if score > 0:
                scored.append((score, chunk_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return {chunk_id for _, chunk_id in scored[:top_k]}


class CaseKeywordStore:
    def __init__(self, public_cases: dict[str, dict[str, Any]], config: dict[str, Any]) -> None:
        self.public_cases = public_cases
        self.config = config
        self.slot_types = config["slot_types"]
        self.keywords_by_case: dict[str, dict[str, str]] = {}
        self.source_by_case: dict[str, str] = {}
        self._load_llm_keyword_outputs()

    def _load_llm_keyword_outputs(self) -> None:
        paths = self.config["paths"]
        batch_path = resolve_path(paths["case_query_keywords_batch"])
        if batch_path.exists():
            for row in load_json(batch_path):
                case_id = row.get("case_id")
                keywords = row.get("keywords")
                if case_id and isinstance(keywords, list):
                    self.keywords_by_case[case_id] = self._keywords_to_slots(keywords)
                    self.source_by_case[case_id] = str(batch_path)

        keyword_dir = resolve_path(paths["case_query_keywords_dir"])
        if keyword_dir.exists():
            for path in sorted(keyword_dir.glob("*.json")):
                row = load_json(path)
                case_id = row.get("case_id") or path.stem
                keywords = row.get("keywords")
                if case_id and case_id not in self.keywords_by_case and isinstance(keywords, list):
                    self.keywords_by_case[case_id] = self._keywords_to_slots(keywords)
                    self.source_by_case[case_id] = str(path)

    def _keywords_to_slots(self, keywords: list[dict[str, Any]]) -> dict[str, str]:
        slots: dict[str, str] = {}
        counts: Counter[str] = Counter()
        seen = set()
        for item in keywords:
            keyword_type = str(item.get("type") or "noun").strip().lower()
            if keyword_type == "date":
                keyword_type = "number"
            if keyword_type not in self.slot_types:
                keyword_type = "noun"
            text = normalize_space(item.get("keyword_text", ""))
            key = (keyword_type, norm_text(text))
            if not text or key in seen:
                continue
            seen.add(key)
            counts[keyword_type] += 1
            if counts[keyword_type] <= int(self.slot_types[keyword_type]):
                slots[f"{keyword_type}{counts[keyword_type]}"] = text
        return slots

    def get_slots(self, case_id: str) -> dict[str, str]:
        if case_id not in self.keywords_by_case:
            if self.config.get("keyword_fallback_strategy") == "heuristic":
                self.keywords_by_case[case_id] = self._heuristic_slots(case_id)
                self.source_by_case[case_id] = "heuristic"
            else:
                self.keywords_by_case[case_id] = {}
                self.source_by_case[case_id] = "missing"
        return self.keywords_by_case[case_id]

    def source(self, case_id: str) -> str:
        self.get_slots(case_id)
        return self.source_by_case.get(case_id, "unknown")

    def _heuristic_slots(self, case_id: str) -> dict[str, str]:
        query = self.public_cases[case_id].get("case_query", "")
        items: list[dict[str, str]] = []

        title_name_re = re.compile(
            r"\b(?:Ông|Bà|Anh|Chị|Cụ)\s+[A-ZĐ][\wÀ-ỹ0-9]*(?:\s+[A-ZĐ][\wÀ-ỹ0-9]*){0,5}",
            flags=re.UNICODE,
        )
        org_re = re.compile(
            r"\b(?:Công ty|Ngân hàng|Quỹ tín dụng|UBND|Ủy ban nhân dân|Bệnh viện)\s+[^,.;:()]{2,70}",
            flags=re.IGNORECASE | re.UNICODE,
        )
        location_re = re.compile(
            r"\b(?:tỉnh|thành phố|tp\.?|huyện|quận|xã|phường|ấp|thôn|đường)\s+[A-ZĐ0-9][^,.;:()]{0,35}",
            flags=re.IGNORECASE | re.UNICODE,
        )
        number_re = re.compile(
            r"\b\d[\d.,/]*(?:m2|m²|đồng|triệu|tỷ|%|kg)?\b",
            flags=re.IGNORECASE | re.UNICODE,
        )

        for regex, keyword_type in [
            (title_name_re, "name"),
            (org_re, "name"),
            (location_re, "location"),
            (number_re, "number"),
        ]:
            for match in regex.finditer(query):
                items.append({"type": keyword_type, "keyword_text": match.group(0)})

        normalized_query = norm_text(query)
        for phrase in [
            "khởi kiện",
            "yêu cầu",
            "đề nghị",
            "buộc",
            "công nhận",
            "hủy",
            "trả",
            "thanh toán",
            "bồi thường",
            "chia",
        ]:
            if norm_text(phrase) in normalized_query:
                items.append({"type": "verb", "keyword_text": phrase})

        tokens = [
            token
            for token in re.findall(r"[A-Za-zÀ-ỹ0-9]+", query)
            if len(norm_text(token)) >= 5 and norm_text(token) not in STOPWORDS
        ]
        for token, _ in Counter(tokens).most_common(int(self.slot_types.get("noun", 10))):
            items.append({"type": "noun", "keyword_text": token})
        return self._keywords_to_slots(items)


class EvolutionQueryOptimizer:
    def __init__(
        self,
        config: dict[str, Any],
        case_type_id: str,
        population_size: int | None = None,
        generations: int | None = None,
        limit_cases: int | None = None,
        no_pruning: bool = False,
    ) -> None:
        self.config = copy.deepcopy(config)
        if population_size is not None:
            self.config["population_size"] = population_size
        if generations is not None:
            self.config["generations"] = generations
        if no_pruning:
            self.config["local_search_pruning"] = False

        self.case_type_id = case_type_id
        self.public_cases = self._load_public_cases()
        self.corpus_by_case = self._load_public_corpus()
        self.case_type_text_by_id, self.case_ids_by_type = self._load_case_groups_from_classifier()
        if case_type_id not in self.case_ids_by_type:
            known = ", ".join(sorted(self.case_ids_by_type))
            raise ValueError(f"Unknown or empty case_type_id {case_type_id!r}. Known: {known}")

        case_ids = [
            case_id
            for case_id in self.case_ids_by_type[case_type_id]
            if case_id in self.public_cases and case_id in self.corpus_by_case
        ]
        if limit_cases is not None:
            case_ids = case_ids[:limit_cases]
        if not case_ids:
            raise ValueError(f"No public cases with chunks for case_type_id={case_type_id}")
        self.case_ids = case_ids

        self.keyword_corpus = self._load_keyword_corpus(case_type_id)
        self.slot_selectors = self._build_slot_selectors()
        self.corpus_selectors = [f"corpus:{keyword_id}" for keyword_id in self.keyword_corpus]
        self.all_selectors = self.slot_selectors + self.corpus_selectors
        self.keyword_store = CaseKeywordStore(self.public_cases, self.config)
        self.retriever = BM25Retriever(self.corpus_by_case)
        self.gold_by_case = self._load_gold()
        self.rng = random.Random(
            int(self.config.get("random_seed", 42)) + sum(ord(ch) for ch in case_type_id)
        )
        self.fitness_cache: dict[str, dict[str, Any]] = {}

    def _load_public_cases(self) -> dict[str, dict[str, Any]]:
        path = resolve_path(self.config["paths"]["public_cases"])
        return {row["case_id"]: row for row in load_json(path)}

    def _load_public_corpus(self) -> dict[str, dict[str, Any]]:
        path = resolve_path(self.config["paths"]["public_corpus"])
        return {row["case_id"]: row for row in load_jsonl(path)}

    def _load_case_groups_from_classifier(self) -> tuple[dict[str, str], dict[str, list[str]]]:
        path = resolve_path(self.config["paths"]["case_type_predictions"])
        rows = load_json(path)
        text_by_id: dict[str, str] = {}
        case_ids_by_type: dict[str, list[str]] = {}
        for row in rows:
            case_id = row.get("case_id")
            case_type_id = row.get("primary_case_type_id")
            case_type_text = row.get("primary_case_type_text", "")
            if not case_id or not case_type_id:
                continue
            text_by_id.setdefault(case_type_id, case_type_text)
            case_ids_by_type.setdefault(case_type_id, []).append(case_id)
        return text_by_id, case_ids_by_type

    def _load_keyword_corpus(self, case_type_id: str) -> dict[int, str]:
        path = resolve_path(self.config["paths"]["keyword_corpus_simple"])
        rows = load_json(path)
        for row in rows:
            if row["case_type_id"] == case_type_id:
                return {
                    int(item["keyword_id"]): item["keyword_text"]
                    for item in row.get("keywords", [])
                }
        raise ValueError(f"case_type_id={case_type_id} not found in keyword corpus simple")

    def _build_slot_selectors(self) -> list[str]:
        selectors: list[str] = []
        for keyword_type, max_index in self.config["slot_types"].items():
            for index in range(1, int(max_index) + 1):
                selectors.append(f"{keyword_type}{index}")
        return selectors

    def _load_gold(self) -> dict[str, set[str]]:
        if self.config.get("gold_strategy") != "all_public_chunks":
            raise ValueError("This optimizer currently supports gold_strategy=all_public_chunks")
        return {
            case_id: {chunk["chunk_id"] for chunk in self.corpus_by_case[case_id].get("chunks", [])}
            for case_id in self.case_ids
        }

    def selector_text(self, selector: str, case_slots: dict[str, str]) -> str:
        corpus_id = parse_corpus_selector(selector)
        if corpus_id is not None:
            return self.keyword_corpus.get(corpus_id, "")
        if parse_slot(selector) is not None:
            return case_slots.get(selector, "")
        return ""

    def instantiate_query(self, query_blueprint: list[str], case_id: str) -> str:
        case_slots = self.keyword_store.get_slots(case_id)
        parts: list[str] = []
        seen = set()
        for selector in query_blueprint:
            text = normalize_space(self.selector_text(selector, case_slots))
            key = norm_text(text)
            if text and key not in seen:
                parts.append(text)
                seen.add(key)
        return normalize_space(" ".join(parts))

    def instantiate_individual(self, individual: list[list[str]], case_id: str) -> list[str]:
        queries: list[str] = []
        seen = set()
        for query_blueprint in individual:
            query = self.instantiate_query(query_blueprint, case_id)
            key = norm_text(query)
            if query and key not in seen:
                queries.append(query)
                seen.add(key)
        return queries

    def random_selector(self) -> str:
        if self.rng.random() < float(self.config.get("case_keyword_source_probability", 0.5)):
            return self.rng.choice(self.slot_selectors)
        return self.rng.choice(self.corpus_selectors)

    def random_query(self) -> list[str]:
        min_len = int(self.config.get("query_keyword_min", 1))
        max_len = int(self.config.get("query_keyword_max", 10))
        length = self.rng.randint(min_len, max_len)
        return self.clean_query([self.random_selector() for _ in range(length)])

    def random_individual(self) -> list[list[str]]:
        query_limit = int(self.config["query_limits"][self.case_type_id])
        min_count = int(self.config.get("initial_query_count_min", 2))
        max_count = min(query_limit, int(self.config.get("initial_query_count_max", 8)))
        count = self.rng.randint(min_count, max(min_count, max_count))
        return self.clean_individual([self.random_query() for _ in range(count)])

    def seed_individuals(self) -> list[list[list[str]]]:
        preferred_corpus_ids = [8, 2, 3, 75, 79, 23, 239, 221, 15, 17, 34]
        available = [f"corpus:{keyword_id}" for keyword_id in preferred_corpus_ids if keyword_id in self.keyword_corpus]
        seed_queries = [
            ["name1", "name2", "number1"],
            ["name1", "noun1", "number1"],
            ["noun1", "noun2", "number1"],
            ["location1", "number1"],
            ["name1", "verb1", "name2"],
        ]
        for selector in available:
            seed_queries.append([selector])
        if available:
            seed_queries.extend(
                [
                    ["name1", available[0]],
                    ["noun1", available[0]],
                    ["name1", "number1", available[0]],
                ]
            )
        seeds: list[list[list[str]]] = [[query] for query in seed_queries]
        seeds.append(seed_queries[:3])
        return [self.clean_individual(seed) for seed in seeds]

    def clean_query(self, query: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen = set()
        for selector in query:
            selector = str(selector).strip()
            if selector in self.slot_selectors:
                valid = selector
            else:
                corpus_id = parse_corpus_selector(selector)
                if corpus_id is None or corpus_id not in self.keyword_corpus:
                    continue
                valid = f"corpus:{corpus_id}"
            if valid not in seen:
                cleaned.append(valid)
                seen.add(valid)
            if len(cleaned) >= int(self.config.get("query_keyword_max", 10)):
                break
        return cleaned

    def clean_individual(self, individual: list[list[str]]) -> list[list[str]]:
        query_limit = int(self.config["query_limits"][self.case_type_id])
        cleaned: list[list[str]] = []
        seen = set()
        for query in individual:
            clean_query = self.clean_query(query)
            if not clean_query:
                continue
            key = tuple(clean_query)
            if key in seen:
                continue
            cleaned.append(clean_query)
            seen.add(key)
            if len(cleaned) >= query_limit:
                break
        if not cleaned:
            cleaned = [self.random_query()]
        return cleaned

    def initial_population(self) -> list[list[list[str]]]:
        population = self.seed_individuals()
        population_size = int(self.config.get("population_size", 40))
        while len(population) < population_size:
            population.append(self.random_individual())
        return population[:population_size]

    def evaluate(self, individual: list[list[str]], include_details: bool = False) -> dict[str, Any]:
        key = canonical(individual)
        if not include_details and key in self.fitness_cache:
            return self.fitness_cache[key]

        top_k = int(self.config.get("top_k_per_query", 1))
        case_scores: list[dict[str, Any]] = []
        for case_id in self.case_ids:
            queries = self.instantiate_individual(individual, case_id)
            if not queries and self.config.get("skip_empty_query_cases", False):
                continue

            predicted: set[str] = set()
            for query in queries:
                predicted.update(self.retriever.retrieve(case_id, query, top_k))

            gold = self.gold_by_case[case_id]
            num_segments = len(self.corpus_by_case[case_id].get("chunks", []))
            recall = case_recall(predicted, gold)
            efficiency = api_efficiency(num_segments, len(queries))
            case_payload: dict[str, Any] = {
                "case_id": case_id,
                "num_segments": num_segments,
                "api_calls": len(queries),
                "case_recall": recall,
                "api_efficiency": efficiency,
                "penalized_case_recall": recall * efficiency,
                "gold_count": len(gold),
                "predicted_count": len(predicted),
                "correct_count": len(predicted & gold),
                "keyword_source": self.keyword_store.source(case_id),
            }
            if include_details:
                case_payload["case_query"] = self.public_cases[case_id].get("case_query", "")
                case_payload["case_slots"] = self.keyword_store.get_slots(case_id)
                case_payload["instantiated_queries"] = queries
                case_payload["predicted_evidence"] = sorted(predicted)
                case_payload["correct_evidence"] = sorted(predicted & gold)
            case_scores.append(case_payload)

        fitness = (
            sum(item["penalized_case_recall"] for item in case_scores) / len(case_scores)
            if case_scores
            else 0.0
        )
        payload: dict[str, Any] = {
            "fitness": fitness,
            "num_cases": len(case_scores),
            "avg_case_recall": (
                sum(item["case_recall"] for item in case_scores) / len(case_scores)
                if case_scores
                else 0.0
            ),
            "avg_api_efficiency": (
                sum(item["api_efficiency"] for item in case_scores) / len(case_scores)
                if case_scores
                else 0.0
            ),
            "avg_api_calls": (
                sum(item["api_calls"] for item in case_scores) / len(case_scores)
                if case_scores
                else 0.0
            ),
        }
        if include_details:
            payload["cases"] = case_scores
        else:
            self.fitness_cache[key] = payload
        return payload

    def tournament_select(self, scored: list[tuple[list[list[str]], dict[str, Any]]]) -> list[list[str]]:
        tournament_size = int(self.config.get("tournament_size", 3))
        candidates = self.rng.sample(scored, k=min(tournament_size, len(scored)))
        candidates.sort(key=lambda item: item[1]["fitness"], reverse=True)
        return copy.deepcopy(candidates[0][0])

    def crossover(self, parent_a: list[list[str]], parent_b: list[list[str]]) -> list[list[str]]:
        if self.rng.random() > float(self.config.get("crossover_rate", 0.8)):
            return copy.deepcopy(parent_a)
        cut_a = self.rng.randint(0, len(parent_a))
        cut_b = self.rng.randint(0, len(parent_b))
        if self.rng.random() < 0.5:
            child = parent_a[:cut_a] + parent_b[cut_b:]
        else:
            child = parent_b[:cut_b] + parent_a[cut_a:]
        return self.clean_individual(child)

    def mutate(self, individual: list[list[str]]) -> list[list[str]]:
        if self.rng.random() > float(self.config.get("mutation_rate", 0.9)):
            return individual
        child = copy.deepcopy(individual)
        op_min = int(self.config.get("mutation_ops_min", 1))
        op_max = int(self.config.get("mutation_ops_max", 3))
        op_count = self.rng.randint(op_min, max(op_min, op_max))
        for _ in range(op_count):
            op = self.rng.choice(
                [
                    "add_query",
                    "remove_query",
                    "replace_query",
                    "add_selector",
                    "remove_selector",
                    "replace_selector",
                ]
            )
            query_limit = int(self.config["query_limits"][self.case_type_id])
            if op == "add_query" and len(child) < query_limit:
                child.append(self.random_query())
            elif op == "remove_query" and len(child) > 1:
                del child[self.rng.randrange(len(child))]
            elif op == "replace_query":
                child[self.rng.randrange(len(child))] = self.random_query()
            elif op == "add_selector":
                query = child[self.rng.randrange(len(child))]
                if len(query) < int(self.config.get("query_keyword_max", 10)):
                    query.append(self.random_selector())
            elif op == "remove_selector":
                query = child[self.rng.randrange(len(child))]
                if len(query) > int(self.config.get("query_keyword_min", 1)):
                    del query[self.rng.randrange(len(query))]
            elif op == "replace_selector":
                query = child[self.rng.randrange(len(child))]
                query[self.rng.randrange(len(query))] = self.random_selector()
            child = self.clean_individual(child)
        return child

    def prune_queries(self, individual: list[list[str]]) -> list[list[str]]:
        if not self.config.get("local_search_pruning", True):
            return individual
        tolerance = float(self.config.get("local_search_tolerance", 1e-12))
        best = self.clean_individual(individual)
        best_score = self.evaluate(best)["fitness"]
        changed = True
        while changed and len(best) > 1:
            changed = False
            for index in range(len(best)):
                candidate = self.clean_individual(
                    [query for query_index, query in enumerate(best) if query_index != index]
                )
                score = self.evaluate(candidate)["fitness"]
                if score + tolerance >= best_score:
                    best = candidate
                    best_score = score
                    changed = True
                    break
        return best

    def optimize(self) -> dict[str, Any]:
        started_at = time.perf_counter()
        population = self.initial_population()
        generations = int(self.config.get("generations", 30))
        elite_size = int(self.config.get("elite_size", 4))
        history: list[dict[str, Any]] = []
        best_individual: list[list[str]] | None = None
        best_eval: dict[str, Any] | None = None

        for generation in range(generations + 1):
            scored = [(individual, self.evaluate(individual)) for individual in population]
            scored.sort(key=lambda item: item[1]["fitness"], reverse=True)
            generation_best, generation_eval = scored[0]
            if best_eval is None or generation_eval["fitness"] > best_eval["fitness"]:
                best_individual = copy.deepcopy(generation_best)
                best_eval = copy.deepcopy(generation_eval)

            avg_fitness = sum(item[1]["fitness"] for item in scored) / len(scored)
            history.append(
                {
                    "generation": generation,
                    "best_fitness": generation_eval["fitness"],
                    "avg_fitness": avg_fitness,
                    "best_query_count": len(generation_best),
                    "best_avg_api_calls": generation_eval["avg_api_calls"],
                    "cache_size": len(self.fitness_cache),
                }
            )
            print(
                f"[{self.case_type_id}] gen={generation}/{generations} "
                f"best={generation_eval['fitness']:.6f} "
                f"avg={avg_fitness:.6f} queries={len(generation_best)}"
            )
            early_stop_threshold = self.config.get("early_stop_fitness_threshold")
            if early_stop_threshold is not None and generation_eval["fitness"] >= float(early_stop_threshold):
                history[-1]["early_stopped"] = True
                history[-1]["early_stop_fitness_threshold"] = float(early_stop_threshold)
                print(
                    f"[{self.case_type_id}] early stop: "
                    f"best={generation_eval['fitness']:.6f} >= threshold={float(early_stop_threshold):.6f}"
                )
                break
            if generation == generations:
                break

            next_population = [copy.deepcopy(item[0]) for item in scored[:elite_size]]
            while len(next_population) < int(self.config.get("population_size", 40)):
                parent_a = self.tournament_select(scored)
                parent_b = self.tournament_select(scored)
                child = self.crossover(parent_a, parent_b)
                child = self.mutate(child)
                next_population.append(child)
            population = next_population

        assert best_individual is not None
        pruned = self.prune_queries(best_individual)
        detailed_eval = self.evaluate(pruned, include_details=True)
        return {
            "case_type_id": self.case_type_id,
            "case_type_text": self.case_type_text_by_id.get(self.case_type_id, ""),
            "case_group_source": str(resolve_path(self.config["paths"]["case_type_predictions"])),
            "case_keyword_sources": {
                case_id: self.keyword_store.source(case_id)
                for case_id in self.case_ids
            },
            "case_ids": self.case_ids,
            "slot_space": self.slot_selectors,
            "corpus_keyword_count": len(self.keyword_corpus),
            "gold_strategy": self.config.get("gold_strategy"),
            "config": self.config,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "best_fitness": detailed_eval["fitness"],
            "best_query_blueprints": pruned,
            "evaluation": detailed_eval,
            "history": history,
        }


def load_config(path: Path) -> dict[str, Any]:
    return load_json(path)


def save_result(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    case_dir = output_dir / result["case_type_id"]
    best_path = case_dir / "best_individual.json"
    history_path = case_dir / "history.json"
    best_payload = {key: value for key, value in result.items() if key != "history"}
    history_payload = {
        "case_type_id": result["case_type_id"],
        "case_type_text": result["case_type_text"],
        "history": result["history"],
    }
    write_json(best_path, best_payload)
    write_json(history_path, history_payload)
    return {"best_individual": str(best_path), "history": str(history_path)}


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.early_stop_fitness_threshold is not None:
        config["early_stop_fitness_threshold"] = args.early_stop_fitness_threshold
    optimizer = EvolutionQueryOptimizer(
        config=config,
        case_type_id=args.case_type_id,
        population_size=args.population_size,
        generations=args.generations,
        limit_cases=args.limit_cases,
        no_pruning=args.no_pruning,
    )
    result = optimizer.optimize()
    output_dir = resolve_path(config.get("output_dir", "outputs/evolution_query_optimizer"))
    saved = save_result(result, output_dir)
    summary = {
        "case_type_id": result["case_type_id"],
        "case_type_text": result["case_type_text"],
        "best_fitness": result["best_fitness"],
        "num_cases": result["evaluation"]["num_cases"],
        "avg_api_calls": result["evaluation"]["avg_api_calls"],
        "query_count": len(result["best_query_blueprints"]),
        "saved": saved,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_all(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output_dir = resolve_path(args.output_dir or config.get("output_dir", "outputs/evolution_query_optimizer"))
    summaries: list[dict[str, Any]] = []
    for case_type_id in config["query_limits"]:
        sub_args = argparse.Namespace(
            config=args.config,
            case_type_id=case_type_id,
            population_size=args.population_size,
            generations=args.generations,
            limit_cases=args.limit_cases,
            output_dir=output_dir,
            no_pruning=args.no_pruning,
            early_stop_fitness_threshold=args.early_stop_fitness_threshold,
        )
        try:
            summaries.append(run_one(sub_args))
        except ValueError as exc:
            summaries.append({"case_type_id": case_type_id, "error": str(exc)})
            print(f"[{case_type_id}] skipped: {exc}")
    write_json(output_dir / "run_all_summary.json", summaries)
    print(json.dumps({"saved_to": str(output_dir / "run_all_summary.json")}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize ALQAC query blueprints after case-level classification and keyword extraction."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--population-size", type=int)
    common.add_argument("--generations", type=int)
    common.add_argument("--limit-cases", type=int)
    common.add_argument("--output-dir", type=Path)
    common.add_argument("--no-pruning", action="store_true")
    common.add_argument("--early-stop-fitness-threshold", type=float)

    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument("--case-type-id", required=True)
    run_parser.set_defaults(func=lambda args: run_one(args))

    run_all_parser = subparsers.add_parser("run-all", parents=[common])
    run_all_parser.set_defaults(func=run_all)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
