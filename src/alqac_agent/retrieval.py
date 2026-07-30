from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .outcome import normalize_ascii

TOKEN_PATTERN = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


@dataclass(slots=True)
class _Document:
    payload: dict[str, object]
    frequencies: Counter[str]
    length: int


class BM25Index:
    """Small dependency-free BM25 index suitable for the 3,352 law articles."""

    def __init__(
        self,
        payloads: Iterable[dict[str, object]],
        *,
        text_key: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[_Document] = []
        document_frequency: Counter[str] = Counter()
        for payload in payloads:
            frequencies = Counter(tokenize(str(payload[text_key])))
            self.documents.append(
                _Document(payload=dict(payload), frequencies=frequencies, length=sum(frequencies.values()))
            )
            document_frequency.update(frequencies.keys())
        self.average_length = (
            sum(document.length for document in self.documents) / len(self.documents)
            if self.documents
            else 1.0
        )
        total = len(self.documents)
        self.idf = {
            token: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        query_terms = Counter(tokenize(query))
        scored: list[tuple[float, dict[str, object]]] = []
        for document in self.documents:
            score = 0.0
            normalization = self.k1 * (
                1 - self.b + self.b * document.length / self.average_length
            )
            for token, query_frequency in query_terms.items():
                term_frequency = document.frequencies.get(token, 0)
                if not term_frequency:
                    continue
                score += (
                    self.idf.get(token, 0.0)
                    * (term_frequency * (self.k1 + 1))
                    / (term_frequency + normalization)
                    * min(query_frequency, 2)
                )
            if score > 0:
                payload = dict(document.payload)
                payload["score"] = round(score, 6)
                scored.append((score, payload))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in scored[:top_k]]


class LawRetriever:
    def __init__(self, articles: list[dict[str, object]]) -> None:
        self.index = BM25Index(articles, text_key="content")

    def search(self, query: str, top_k: int = 18) -> list[dict[str, object]]:
        return self.index.search(query, top_k)


class ExampleRetriever:
    def __init__(self, examples: list[dict[str, object]]) -> None:
        normalized = [
            {
                "case_id": example["case_id"],
                "case_query": example["case_query"],
                "case_type": example.get("case_type", ""),
                "verdict_label": example["verdict_label"],
                "search_text": str(example["case_query"]),
            }
            for example in examples
        ]
        self.index = BM25Index(normalized, text_key="search_text")
        self.examples = normalized

    @staticmethod
    def _word_ngram_counts(text: str) -> Counter[str]:
        words = re.findall(r"[a-z0-9]+", normalize_ascii(text))
        features = words + [
            f"{words[index]} {words[index + 1]}"
            for index in range(len(words) - 1)
        ]
        return Counter(features)

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        exclude_case_id: str | None = None,
    ) -> list[dict[str, object]]:
        # Retrieve one extra candidate because the evaluated public case is often
        # the exact top-1 match and must not reveal its own label.
        search_k = top_k + 1 if exclude_case_id else top_k
        results = self.index.search(query, search_k)
        if exclude_case_id:
            results = [
                item for item in results if str(item["case_id"]) != exclude_case_id
            ]
        results = results[:top_k]
        for result in results:
            result.pop("search_text", None)
        return results

    def predict_prior(
        self,
        query: str,
        top_k: int = 7,
        *,
        exclude_case_id: str | None = None,
    ) -> dict[str, object]:
        search_k = top_k + 1 if exclude_case_id else top_k
        hits = self.index.search(query, search_k)
        if exclude_case_id:
            hits = [x for x in hits if str(x["case_id"]) != exclude_case_id]
        hits = hits[:top_k]
        scores: dict[str, float] = {}
        neighbors: list[dict[str, object]] = []
        for rank, hit in enumerate(hits, start=1):
            label = str(hit["verdict_label"])
            contribution = float(hit["score"]) / rank
            scores[label] = scores.get(label, 0.0) + contribution
            neighbors.append(
                {
                    "case_id": hit["case_id"],
                    "label": label,
                    "score": hit["score"],
                }
            )
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        prediction = ordered[0][0] if ordered else "PARTIAL_A_WIN"
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else 0.0
        return {
            "prediction": prediction,
            "scores": {key: round(value, 6) for key, value in scores.items()},
            "margin": round(margin, 6),
            "neighbors": neighbors,
        }

    def predict_partial_prior(
        self,
        query: str,
        *,
        exclude_case_id: str | None = None,
        top_k: int = 3,
    ) -> dict[str, object]:
        candidates = [
            item
            for item in self.examples
            if item["verdict_label"] in {"PARTIAL_A_WIN", "PARTIAL_B_WIN"}
            and str(item["case_id"]) != exclude_case_id
        ]
        if not candidates:
            return {"prediction": "PARTIAL_A_WIN", "scores": {}, "neighbors": []}

        counts = [
            self._word_ngram_counts(str(item["case_query"]))
            for item in candidates
        ]
        document_frequency: Counter[str] = Counter()
        for frequencies in counts:
            document_frequency.update(frequencies.keys())
        total_documents = len(candidates)

        def vectorize(frequencies: Counter[str]) -> dict[str, float]:
            vector = {
                term: (1 + math.log(frequency))
                * (
                    math.log(
                        (total_documents + 1)
                        / (document_frequency.get(term, 0) + 1)
                    )
                    + 1
                )
                for term, frequency in frequencies.items()
            }
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
            return {term: value / norm for term, value in vector.items()}

        vectors = [vectorize(frequencies) for frequencies in counts]
        target_vector = vectorize(self._word_ngram_counts(query))
        similarities: list[tuple[float, int]] = []
        for index, vector in enumerate(vectors):
            similarity = sum(
                value * vector.get(term, 0.0)
                for term, value in target_vector.items()
            )
            similarities.append((similarity, index))
        similarities.sort(reverse=True)
        class_counts = Counter(str(item["verdict_label"]) for item in candidates)
        scores: dict[str, float] = {}
        neighbors: list[dict[str, object]] = []
        for similarity, index in similarities[:top_k]:
            label = str(candidates[index]["verdict_label"])
            contribution = 1 / math.sqrt(class_counts[label])
            scores[label] = scores.get(label, 0.0) + contribution
            neighbors.append(
                {
                    "case_id": candidates[index]["case_id"],
                    "label": label,
                    "similarity": round(similarity, 8),
                }
            )
        prediction = "PARTIAL_B_WIN" if (
            scores.get("PARTIAL_B_WIN", 0.0)
            > scores.get("PARTIAL_A_WIN", 0.0)
        ) else "PARTIAL_A_WIN"
        return {
            "prediction": prediction,
            "scores": {key: round(value, 8) for key, value in scores.items()},
            "neighbors": neighbors,
        }
