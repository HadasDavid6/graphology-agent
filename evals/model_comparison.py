
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_models(
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compare models using summaries produced by cost_tracker.py.

    Input:
    {
        "model-a": {
            "cases": 10,
            "passed": 9,
            "pass_rate": 0.9,
            "total_cost_usd": 0.02,
            "average_latency_seconds": 1.5
        },
        ...
    }
    """

    results = []

    for model, summary in summaries.items():
        cases = int(summary.get("cases") or 0)
        passed = int(summary.get("passed") or 0)
        cost = summary.get("total_cost_usd")
        latency = summary.get("average_latency_seconds")

        cost_per_success = None
        if cost is not None and passed > 0:
            cost_per_success = float(cost) / passed

        results.append({
            "model": model,
            "cases": cases,
            "passed": passed,
            "failed": int(summary.get("failed") or 0),
            "pass_rate": float(
                summary.get("pass_rate")
                if summary.get("pass_rate") is not None
                else (passed / cases if cases else 0)
            ),
            "total_cost_usd": (
                float(cost) if cost is not None else None
            ),
            "cost_per_success_usd": cost_per_success,
            "average_latency_seconds": (
                float(latency) if latency is not None else None
            ),
            "input_tokens": int(summary.get("input_tokens") or 0),
            "output_tokens": int(summary.get("output_tokens") or 0),
            "embedding_tokens": int(
                summary.get("embedding_tokens") or 0
            ),
        })

    # Primary ranking: pass rate.
    # Tie-breakers: cost per successful case, then latency.
    results.sort(
        key=lambda item: (
            -item["pass_rate"],
            (
                item["cost_per_success_usd"]
                if item["cost_per_success_usd"] is not None
                else float("inf")
            ),
            (
                item["average_latency_seconds"]
                if item["average_latency_seconds"] is not None
                else float("inf")
            ),
        )
    )

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results


def print_comparison(results: list[dict[str, Any]]) -> None:
    """Print a readable comparison table."""

    if not results:
        print("No model results to compare.")
        return

    print("\n=== MODEL COMPARISON ===\n")

    headers = [
        "Rank",
        "Model",
        "Pass rate",
        "Passed",
        "Cost ($)",
        "Cost / success",
        "Avg latency",
    ]

    rows = []

    for result in results:
        cost = result["total_cost_usd"]
        cost_per_success = result["cost_per_success_usd"]
        latency = result["average_latency_seconds"]

        rows.append([
            str(result["rank"]),
            result["model"],
            f'{result["pass_rate"]:.1%}',
            f'{result["passed"]}/{result["cases"]}',
            f"{cost:.6f}" if cost is not None else "N/A",
            (
                f"{cost_per_success:.6f}"
                if cost_per_success is not None
                else "N/A"
            ),
            f"{latency:.2f}s" if latency is not None else "N/A",
        ])

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[i])
            for i, value in enumerate(row)
        )

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        print(format_row(row))

    print()


def save_comparison(
    results: list[dict[str, Any]],
    output_dir: str | Path = "evals/results",
) -> Path:
    """Save comparison results to JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model_comparison.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return output_path


def compare_and_save(
    summaries: dict[str, dict[str, Any]],
    output_dir: str | Path = "evals/results",
) -> list[dict[str, Any]]:
    """Convenience function: compare, print and save."""

    results = compare_models(summaries)
    print_comparison(results)
    save_comparison(results, output_dir)

    return results
