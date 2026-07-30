from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from urllib import error, request


TASK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TASK_ROOT
DEFAULT_CONFIG = TASK_ROOT / "config.json"
CASE_RETRIEVE_URL = "https://alqac-api.ngrok.pro/retrieve"

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
    return path if path.is_absolute() else TASK_ROOT / path


def resolve_cli_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    no_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return no_marks.replace("đ", "d").replace("Đ", "D")


def norm_text(text: str) -> str:
    return normalize_space(strip_accents(text).lower())


def load_env(path: Path = TASK_ROOT / ".env") -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lstrip("\ufeff")] = value.strip().strip('"').strip("'")
    return values


def cache_key(case_id: str, query: str) -> str:
    raw = json.dumps({"case_id": case_id, "query": normalize_space(query)}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    write_json(path, cache)


def wait_for_rate_limit(last_call_path: Path, min_seconds: float) -> None:
    last_call_path.parent.mkdir(parents=True, exist_ok=True)
    if not last_call_path.exists():
        return
    try:
        last_call = float(last_call_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    delay = min_seconds - (time.time() - last_call)
    if delay > 0:
        time.sleep(delay)


def mark_api_call(last_call_path: Path) -> None:
    last_call_path.parent.mkdir(parents=True, exist_ok=True)
    last_call_path.write_text(str(time.time()), encoding="utf-8")


def call_case_api(case_id: str, query: str, token: str, insecure: bool, min_seconds_between_calls: float, max_retries: int = 5) -> Any:
    last_call_path = REPO_ROOT / ".cache" / "case_api_last_call.txt"
    body = json.dumps({"case_id": case_id, "query": query}, ensure_ascii=False).encode("utf-8")
    context = ssl._create_unverified_context() if insecure else None
    for attempt in range(max_retries):
        wait_for_rate_limit(last_call_path, min_seconds_between_calls)
        req = request.Request(
            CASE_RETRIEVE_URL,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": token},
        )
        try:
            with request.urlopen(req, timeout=90, context=context) as resp:
                mark_api_call(last_call_path)
                return json.load(resp)
        except error.HTTPError as exc:
            mark_api_call(last_call_path)
            if exc.code in (429, 502, 503, 504) and attempt < max_retries - 1:
                wait = min(10 * (attempt + 1), 60)
                print(f"  [retry {attempt+1}/{max_retries}] HTTP {exc.code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Case API failed with HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            mark_api_call(last_call_path)
            if attempt < max_retries - 1:
                wait = min(10 * (attempt + 1), 60)
                print(f"  [retry {attempt+1}/{max_retries}] {exc}, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Case API failed after {max_retries} retries")


def parse_api_response(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "chunk_id" in data:
        return [data]
    raise ValueError(f"Unexpected API response shape: {type(data).__name__}")


class CaseTypeStore:
    def __init__(self, path: Path) -> None:
        self.rows_by_case_id = {row["case_id"]: row for row in load_json(path)}

    def get(self, case_id: str) -> dict[str, str]:
        row = self.rows_by_case_id.get(case_id)
        if not row:
            raise KeyError(f"Missing case type prediction for {case_id}")
        return {
            "case_type_id": row["primary_case_type_id"],
            "case_type_text": row["primary_case_type_text"],
        }


class KeywordStore:
    def __init__(self, config: dict[str, Any], public_cases: dict[str, dict[str, Any]]) -> None:
        self.config = config
        self.public_cases = public_cases
        self.slot_types = config["slot_types"]
        self.slots_by_case_id: dict[str, dict[str, str]] = {}
        self.source_by_case_id: dict[str, str] = {}
        self._load_batch()
        self._load_dir_missing_only()

    def _load_batch(self) -> None:
        path = resolve_path(self.config["paths"]["case_query_keywords_batch"])
        if not path.exists():
            return
        for row in load_json(path):
            case_id = row.get("case_id")
            keywords = row.get("keywords")
            if case_id and isinstance(keywords, list):
                self.slots_by_case_id[case_id] = self.keywords_to_slots(keywords)
                self.source_by_case_id[case_id] = str(path)

    def _load_dir_missing_only(self) -> None:
        path = resolve_path(self.config["paths"]["case_query_keywords_dir"])
        if not path.exists():
            return
        for file_path in sorted(path.glob("*.json")):
            row = load_json(file_path)
            case_id = row.get("case_id") or file_path.stem
            keywords = row.get("keywords")
            if case_id and case_id not in self.slots_by_case_id and isinstance(keywords, list):
                self.slots_by_case_id[case_id] = self.keywords_to_slots(keywords)
                self.source_by_case_id[case_id] = str(file_path)

    def keywords_to_slots(self, keywords: list[dict[str, Any]]) -> dict[str, str]:
        slots: dict[str, str] = {}
        counts: dict[str, int] = {}
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
            counts[keyword_type] = counts.get(keyword_type, 0) + 1
            if counts[keyword_type] <= int(self.slot_types[keyword_type]):
                slots[f"{keyword_type}{counts[keyword_type]}"] = text
        return slots

    def get(self, case_id: str) -> dict[str, str]:
        if case_id not in self.slots_by_case_id:
            if self.config.get("keyword_fallback_strategy") == "heuristic":
                self.slots_by_case_id[case_id] = self.heuristic_slots(case_id)
                self.source_by_case_id[case_id] = "heuristic"
            else:
                self.slots_by_case_id[case_id] = {}
                self.source_by_case_id[case_id] = "missing"
        return self.slots_by_case_id[case_id]

    def source(self, case_id: str) -> str:
        self.get(case_id)
        return self.source_by_case_id.get(case_id, "unknown")

    def heuristic_slots(self, case_id: str) -> dict[str, str]:
        query = self.public_cases[case_id]["case_query"]
        items: list[dict[str, str]] = []
        title_name_re = re.compile(
            r"\b(?:Ông|Bà|Anh|Chị|Cụ)\s+[A-ZĐ][\wÀ-ỹ0-9]*(?:\s+[A-ZĐ][\wÀ-ỹ0-9]*){0,5}",
            flags=re.UNICODE,
        )
        location_re = re.compile(
            r"\b(?:tỉnh|thành phố|tp\.?|huyện|quận|xã|phường|ấp|thôn|đường)\s+[A-ZĐ0-9][^,.;:()]{0,35}",
            flags=re.IGNORECASE | re.UNICODE,
        )
        number_re = re.compile(r"\b\d[\d.,/]*(?:m2|m²|đồng|triệu|tỷ|%|kg)?\b", flags=re.IGNORECASE | re.UNICODE)
        for regex, keyword_type in [(title_name_re, "name"), (location_re, "location"), (number_re, "number")]:
            for match in regex.finditer(query):
                items.append({"type": keyword_type, "keyword_text": match.group(0)})
        normalized_query = norm_text(query)
        for phrase in ["khởi kiện", "yêu cầu", "đề nghị", "buộc", "công nhận", "hủy", "trả", "thanh toán", "bồi thường", "chia"]:
            if norm_text(phrase) in normalized_query:
                items.append({"type": "verb", "keyword_text": phrase})
        tokens = [
            token
            for token in re.findall(r"[A-Za-zÀ-ỹ0-9]+", query)
            if len(norm_text(token)) >= 5 and norm_text(token) not in STOPWORDS
        ]
        for token, _ in Counter(tokens).most_common(int(self.slot_types.get("noun", 10))):
            items.append({"type": "noun", "keyword_text": token})
        return self.keywords_to_slots(items)


def load_public_cases(path: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in load_json(path)}


def load_keyword_corpus(path: Path) -> dict[str, dict[int, str]]:
    by_type: dict[str, dict[int, str]] = {}
    for row in load_json(path):
        by_type[row["case_type_id"]] = {
            int(item["keyword_id"]): item["keyword_text"]
            for item in row.get("keywords", [])
        }
    return by_type


def load_blueprints(path: Path) -> dict[str, dict[str, Any]]:
    registry = load_json(path)
    return {item["case_type_id"]: item for item in registry.get("blueprints", [])}


def selector_text(selector: str, slots: dict[str, str], corpus_keywords: dict[int, str]) -> str:
    if selector.startswith("corpus:"):
        try:
            return corpus_keywords.get(int(selector.split(":", 1)[1]), "")
        except ValueError:
            return ""
    return slots.get(selector, "")


def instantiate_blueprints(
    query_blueprints: list[list[str]],
    slots: dict[str, str],
    corpus_keywords: dict[int, str],
) -> list[str]:
    queries: list[str] = []
    seen_queries = set()
    for blueprint in query_blueprints:
        parts: list[str] = []
        seen_parts = set()
        for selector in blueprint:
            text = normalize_space(selector_text(selector, slots, corpus_keywords))
            key = norm_text(text)
            if text and key not in seen_parts:
                parts.append(text)
                seen_parts.add(key)
        query = normalize_space(" ".join(parts))
        key = norm_text(query)
        if query and key not in seen_queries:
            queries.append(query)
            seen_queries.add(key)
    return queries


def retrieve_query(case_id: str, query: str, token: str, config: dict[str, Any], cache: dict[str, Any], cache_path: Path, dry_run: bool) -> tuple[list[dict[str, Any]], bool]:
    key = cache_key(case_id, query)
    if key in cache:
        return parse_api_response(cache[key]["response"]), True
    if dry_run:
        return [], False
    response = call_case_api(
        case_id=case_id,
        query=query,
        token=token,
        insecure=bool(config["api"].get("insecure", True)),
        min_seconds_between_calls=float(config["api"].get("min_seconds_between_calls", 5.2)),
    )
    cache[key] = {
        "case_id": case_id,
        "query": query,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": response,
    }
    save_cache(cache_path, cache)
    return parse_api_response(response), False


def prioritize_queries(queries: list[str]) -> list[str]:
    """Sort queries: shorter and more diverse queries first (better BM25 match)."""
    return sorted(queries, key=lambda q: (len(q.split()), len(q)))


def retrieve_cases(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    public_cases = load_public_cases(resolve_path(config["paths"]["public_cases"]))
    case_type_store = CaseTypeStore(resolve_path(config["paths"]["case_type_predictions"]))
    keyword_store = KeywordStore(config, public_cases)
    keyword_corpus_by_type = load_keyword_corpus(resolve_path(config["paths"]["keyword_corpus_simple"]))
    blueprint_path = resolve_cli_path(args.blueprints) if args.blueprints else resolve_path(config["paths"]["blueprint_registry"])
    blueprints_by_type = load_blueprints(blueprint_path)
    cache_path = resolve_path(config["paths"]["api_cache"])
    cache = load_cache(cache_path)
    output_path = resolve_cli_path(args.output) if args.output else resolve_path(config["paths"]["output_jsonl"])

    env = {**load_env(), **os.environ}
    token = args.token or env.get("ALQAC_TOKEN")
    if not token and not args.dry_run:
        raise ValueError("Missing ALQAC_TOKEN in .env or --token")

    case_ids = list(public_cases)
    if args.case_id:
        case_ids = [args.case_id]
    if args.start is not None or args.limit is not None:
        start = args.start or 0
        end = start + args.limit if args.limit is not None else None
        case_ids = case_ids[start:end]

    if args.overwrite and output_path.exists():
        output_path.unlink()

    for index, case_id in enumerate(case_ids, 1):
        case = public_cases[case_id]
        case_type = case_type_store.get(case_id)
        case_type_id = case_type["case_type_id"]
        blueprint_record = blueprints_by_type.get(case_type_id)
        if not blueprint_record:
            payload = {
                "case_id": case_id,
                "case_query": case["case_query"],
                "case_type_id": case_type_id,
                "case_type_text": case_type["case_type_text"],
                "status": "missing_blueprint",
                "chunks": [],
            }
            append_jsonl(output_path, payload)
            print(f"[{index}/{len(case_ids)}] {case_id} -> missing blueprint for {case_type_id}")
            continue

        slots = keyword_store.get(case_id)
        corpus_keywords = keyword_corpus_by_type.get(case_type_id, {})
        all_queries = instantiate_blueprints(
            blueprint_record["best_query_blueprints"],
            slots,
            corpus_keywords,
        )
        max_queries = args.max_queries if args.max_queries is not None else len(all_queries)
        instantiated_queries = prioritize_queries(all_queries)[:max_queries]

        chunks_by_id: dict[str, dict[str, Any]] = {}
        query_logs: list[dict[str, Any]] = []
        api_calls_made = 0
        cache_hits = 0
        consecutive_dupes = 0
        early_stopped = False
        for query in instantiated_queries:
            results, cache_hit = retrieve_query(case_id, query, token or "", config, cache, cache_path, args.dry_run)
            cache_hits += int(cache_hit)
            api_calls_made += int(not cache_hit and not args.dry_run)
            new_chunks = 0
            for item in results:
                chunk_id = item.get("chunk_id")
                if not isinstance(chunk_id, str):
                    continue
                is_new = chunk_id not in chunks_by_id
                chunk = chunks_by_id.setdefault(
                    chunk_id,
                    {
                        "chunk_id": chunk_id,
                        "text": str(item.get("text", "")),
                        "score": item.get("score"),
                        "source_queries": [],
                    },
                )
                chunk["source_queries"].append(query)
                if is_new:
                    new_chunks += 1
            query_logs.append({"query": query, "cache_hit": cache_hit, "num_results": len(results), "new_chunks": new_chunks})

            if not args.dry_run and not cache_hit:
                if new_chunks == 0:
                    consecutive_dupes += 1
                else:
                    consecutive_dupes = 0
                if args.early_stop and consecutive_dupes >= args.early_stop and len(chunks_by_id) >= 1:
                    early_stopped = True
                    break

        payload = {
            "case_id": case_id,
            "case_query": case["case_query"],
            "case_type_id": case_type_id,
            "case_type_text": case_type["case_type_text"],
            "keyword_source": keyword_store.source(case_id),
            "num_instantiated_queries": len(instantiated_queries),
            "num_total_available_queries": len(all_queries),
            "early_stopped": early_stopped,
            "api_calls_made": api_calls_made,
            "cache_hits": cache_hits,
            "chunks": list(chunks_by_id.values()),
            "query_logs": query_logs,
        }
        append_jsonl(output_path, payload)
        print(
            f"[{index}/{len(case_ids)}] {case_id} -> "
            f"type={case_type_id} queries={len(query_logs)}{'(early_stop)' if early_stopped else ''} "
            f"chunks={len(chunks_by_id)} api_calls={api_calls_made} cache_hits={cache_hits}"
        )
    print(json.dumps({"saved_to": str(output_path), "cases": len(case_ids)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve ALQAC public-test evidence with optimized query blueprints.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--blueprints", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--start", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--token")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None, help="Cap queries per case")
    parser.add_argument("--early-stop", type=int, default=None, help="Stop after N consecutive duplicate-only queries")
    args = parser.parse_args()
    retrieve_cases(args)


if __name__ == "__main__":
    main()
