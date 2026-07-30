from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TASK_ROOT.parent
DEFAULT_CONFIG = TASK_ROOT / "config.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


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


def build_registry(config: dict[str, Any], include_excluded: bool = False) -> dict[str, Any]:
    optimizer_dir = resolve_path(config["paths"]["optimizer_output_dir"])
    excluded = set(config.get("exclude_case_type_ids", []))
    blueprints: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for path in sorted(optimizer_dir.glob("*/best_individual.json")):
        data = load_json(path)
        case_type_id = data.get("case_type_id")
        if not case_type_id:
            skipped.append({"path": str(path), "reason": "missing case_type_id"})
            continue
        if case_type_id in excluded and not include_excluded:
            skipped.append({"path": str(path), "case_type_id": case_type_id, "reason": "excluded"})
            continue
        query_blueprints = data.get("best_query_blueprints")
        if not isinstance(query_blueprints, list):
            skipped.append({"path": str(path), "case_type_id": case_type_id, "reason": "missing best_query_blueprints"})
            continue
        blueprints.append(
            {
                "case_type_id": case_type_id,
                "case_type_text": data.get("case_type_text", ""),
                "best_query_blueprints": query_blueprints,
                "metadata": {
                    "best_fitness": data.get("best_fitness"),
                    "query_count": len(query_blueprints),
                    "source": str(path),
                },
            }
        )

    return {
        "metadata": {
            "source_optimizer_output_dir": str(optimizer_dir),
            "excluded_case_type_ids": sorted(excluded) if not include_excluded else [],
            "num_case_types": len(blueprints),
        },
        "blueprints": blueprints,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build best query blueprint registry from optimizer outputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-excluded", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    registry = build_registry(config, include_excluded=args.include_excluded)
    output = resolve_cli_path(args.output) if args.output else resolve_path(config["paths"]["blueprint_registry"])
    write_json(output, registry)
    print(json.dumps({"saved_to": str(output), "num_case_types": registry["metadata"]["num_case_types"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
