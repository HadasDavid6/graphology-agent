"""Helpers for aggregating provider usage across a single agent run."""

from __future__ import annotations

from typing import Any


def add_usage(
    current: dict[str, Any] | None,
    addition: dict[str, int | float | None],
) -> dict[str, Any]:
    usage = dict(current or {})
    usage["input_tokens"] = usage.get("input_tokens", 0) + (addition.get("input_tokens") or 0)
    usage["output_tokens"] = usage.get("output_tokens", 0) + (addition.get("output_tokens") or 0)

    cost = addition.get("cost_usd")
    if cost is None:
        usage["cost_complete"] = False
    else:
        usage["cost_usd"] = usage.get("cost_usd", 0.0) + cost
        usage["cost_complete"] = usage.get("cost_complete", True)
    return usage
