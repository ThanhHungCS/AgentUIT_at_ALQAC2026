from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import httpx

from .data import write_json_atomic
from .schemas import CaseEvidence


class CaseContentClient:
    """Official Top-1 API client with persistent cache and global pacing."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        cache_path: str | Path,
        interval_seconds: float = 5.1,
        timeout_seconds: float = 30,
        max_retries: int = 3,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.cache_path = Path(cache_path)
        self.interval_seconds = interval_seconds
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout_seconds)
        self._lock = threading.Lock()
        self._last_request_at = 0.0
        self._cache: dict[str, dict[str, object]] = self._read_cache()

    @staticmethod
    def _key(case_id: str, query: str) -> str:
        raw = f"{case_id}\0{' '.join(query.split()).casefold()}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _read_cache(self) -> dict[str, dict[str, object]]:
        if not self.cache_path.exists():
            return {}
        with self.cache_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}

    def _pace(self) -> None:
        remaining = self.interval_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def retrieve(self, case_id: str, query: str) -> CaseEvidence | None:
        key = self._key(case_id, query)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return CaseEvidence.model_validate({**cached, "cached": True})
            if not self.api_key:
                raise ValueError("Thiếu ALQAC_API_KEY")
            for attempt in range(self.max_retries + 1):
                self._pace()
                try:
                    response = self._client.post(
                        self.url,
                        headers={"X-API-Key": self.api_key},
                        json={"query": query, "case_id": case_id},
                    )
                except httpx.RequestError:
                    self._last_request_at = time.monotonic()
                    if attempt == self.max_retries:
                        raise
                    time.sleep(self.interval_seconds)
                    continue
                self._last_request_at = time.monotonic()
                if response.status_code not in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    break
                if attempt == self.max_retries:
                    response.raise_for_status()
                retry_after = float(response.headers.get("Retry-After", self.interval_seconds))
                time.sleep(max(retry_after, self.interval_seconds))
            results = response.json().get("results", [])
            if not results:
                return None
            hit = results[0]
            evidence = CaseEvidence(
                chunk_id=str(hit["chunk_id"]),
                score=float(hit.get("score", 0)),
                text=str(hit["text"]),
                query=query,
            )
            self._cache[key] = evidence.model_dump(exclude={"cached"})
            write_json_atomic(self.cache_path, self._cache)
            return evidence

    def close(self) -> None:
        self._client.close()
