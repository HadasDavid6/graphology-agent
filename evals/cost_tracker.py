
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def usage_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """
    Extract usage information from the final agent state.

    Expected state structure:
    {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.001
        }
    }
    """
    usage = state.get("usage") or {}

    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cost_usd": _safe_float(usage.get("cost_usd")),
        "embedding_tokens": int(usage.get("embedding_tokens") or 0),
        "embedding_cost_usd": _safe_float(
            usage.get("embedding_cost_usd")
        ),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_usage_record(
    *,
    case_id: str,
    model: str,
    usage: dict[str, Any],
    latency_seconds: float,
    passed: bool,
) -> dict[str, Any]:
    """Build one record for a single evaluation case."""

    completion_cost = usage.get("cost_usd")
    embedding_cost = usage.get("embedding_cost_usd")

    total_cost = None
    if completion_cost is not None or embedding_cost is not None:
        total_cost = (completion_cost or 0.0) + (embedding_cost or 0.0)

    return {
        "case_id": case_id,
        "model": model,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "embedding_tokens": usage.get("embedding_tokens", 0),
        "completion_cost_usd": completion_cost,
        "embedding_cost_usd": embedding_cost,
        "total_cost_usd": total_cost,
        "latency_seconds": round(latency_seconds, 4),
        "passed": bool(passed),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate aggregate metrics for a model."""

    if not records:
        return {
            "cases": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "embedding_tokens": 0,
            "total_cost_usd": None,
            "average_latency_seconds": 0.0,
        }

    passed = sum(1 for r in records if r["passed"])

    known_costs = [
        r["total_cost_usd"]
        for r in records
        if r.get("total_cost_usd") is not None
    ]

    return {
        "cases": len(records),
        "passed": passed,
        "failed": len(records) - passed,
        "pass_rate": round(passed / len(records), 4),
        "input_tokens": sum(r.get("input_tokens", 0) for r in records),
        "output_tokens": sum(r.get("output_tokens", 0) for r in records),
        "embedding_tokens": sum(
            r.get("embedding_tokens", 0) for r in records
        ),
        "total_cost_usd": (
            round(sum(known_costs), 8) if known_costs else None
        ),
        "average_latency_seconds": round(
            sum(r.get("latency_seconds", 0) for r in records)
            / len(records),
            4,
        ),
    }


def save_records(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    model: str,
) -> None:
    """Save detailed records as JSON and CSV."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = model.replace("/", "_").replace(":", "_")

    json_path = output_dir / f"{safe_model}_results.json"
    csv_path = output_dir / f"{safe_model}_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    if records:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(records[0].keys()),
            )
            writer.writeheader()
            writer.writerows(records)


def save_summary(
    summary: dict[str, Any],
    output_dir: str | Path,
    model: str,
) -> None:
    """Save aggregate model metrics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = model.replace("/", "_").replace(":", "_")
    path = output_dir / f"{safe_model}_summary.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
