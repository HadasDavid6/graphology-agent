
"""Eval runner.

Runs an agent against its dataset and scores each case.
Returns a metrics summary for every model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import yaml

from core.agents.registry import get_agent
from core.runtime.orchestrator import run_agent
from core.security.secrets import get_secret

_ROOT = Path(__file__).resolve().parent

def _save_results(
    suite: str,
    summaries: list[dict],
) -> None:
    results_dir = _ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    report = {
        "suite": suite,
        "timestamp": timestamp,
        "models": summaries,
        "comparison": [
            {
                "model": s["model"],
                "passed_checks": s["passed_checks"],
                "total_checks": s["total_checks"],
                "pass_rate": s["pass_rate"],
                "total_tokens": s["total_tokens"],
                "total_cost_usd": s["total_cost_usd"],
                "avg_latency_seconds": s["avg_latency_seconds"],
                "success": s["success"],
            }
            for s in summaries
        ],
    }

    # Latest results
    latest = results_dir / "results.json"
    latest.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Historical results
    filename = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    history = results_dir / f"results_{filename}.json"
    history.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResults saved:")
    print(f"  Latest: {latest}")
    print(f"  History: {history}")
    
def _load_dataset(suite: str) -> list[dict]:
    path = _ROOT / "datasets" / f"{suite}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("cases", [])


def _load_checks(suite: str) -> dict:
    mod = importlib.import_module(f"evals.suites.{suite}")
    return mod.CHECKS


def _num(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def run_suite(
    suite: str,
    mock: bool | None,
    model_override: str | None = None,
) -> dict:
    cases = _load_dataset(suite)
    checks = _load_checks(suite)

    total_checks = 0
    passed_checks = 0
    passed_cases = 0
    total_latency = 0.0

    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "embedding_tokens": 0,
        "cost_usd": 0.0,
        "embedding_cost_usd": 0.0,
        "retrieval_seconds": 0.0,
        "generation_seconds": 0.0,
    }

    mode = "mock" if mock else "real"
    model_label = model_override or "agent.yaml"

    print(
        f"=== eval suite: {suite} "
        f"({mode}, model={model_label}) ===\n"
    )

    for case in cases:
        agent = get_agent(
            suite,
            mock=mock,
            model_override=model_override,
        )

        case_started = perf_counter()
        final = run_agent(agent, case["input"])
        case_latency = perf_counter() - case_started
        total_latency += case_latency

        # run_agent is expected to return the final state dictionary.
        if not isinstance(final, dict):
            raise TypeError(
                "run_agent() must return the final state as a dict; "
                f"received {type(final).__name__}"
            )

        usage = final.get("usage") or {}

        for key in totals:
            totals[key] += _num(usage.get(key))

        print(f"[{case['id']}] {case['input'][:50]}")

        case_ok = True

        for key, expected in case.get("checks", {}).items():
            if key not in checks:
                print(f"    ? unknown check '{key}' — skipped")
                continue

            total_checks += 1
            ok, detail = checks[key](final, expected)

            if ok:
                passed_checks += 1
            else:
                case_ok = False

            print(f"    {'PASS' if ok else 'FAIL'}  {key}: {detail}")

        if case_ok:
            passed_cases += 1

        print(
            f"  -> {'OK' if case_ok else 'CASE FAILED'} "
            f"| {case_latency:.2f}s\n"
        )

    failed_checks = total_checks - passed_checks
    pass_rate = (
        passed_checks / total_checks
        if total_checks
        else 0.0
    )

    avg_latency = (
        total_latency / len(cases)
        if cases
        else 0.0
    )

    total_cost = (
        totals["cost_usd"]
        + totals["embedding_cost_usd"]
    )

    total_tokens = (
        totals["input_tokens"]
        + totals["output_tokens"]
        + totals["embedding_tokens"]
    )

    cost_per_success = (
        total_cost / passed_cases
        if passed_cases
        else 0.0
    )

    summary = {
        "suite": suite,
        "model": model_label,
        "mode": mode,

        "total_cases": len(cases),
        "passed_cases": passed_cases,

        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "pass_rate": pass_rate,

        "input_tokens": int(totals["input_tokens"]),
        "output_tokens": int(totals["output_tokens"]),
        "embedding_tokens": int(totals["embedding_tokens"]),
        "total_tokens": int(total_tokens),

        "completion_cost_usd": totals["cost_usd"],
        "embedding_cost_usd": totals["embedding_cost_usd"],
        "total_cost_usd": total_cost,
        "cost_per_success": cost_per_success,

        "total_latency_seconds": total_latency,
        "latency": avg_latency,
        "avg_latency_seconds": avg_latency,

        "retrieval_seconds": totals["retrieval_seconds"],
        "generation_seconds": totals["generation_seconds"],

        "success": passed_checks == total_checks,
    }

    print(f"=== {passed_checks}/{total_checks} checks passed ===")
    print(
        f"Tokens: {total_tokens:.0f} | "
        f"Cost: ${total_cost:.6f} | "
        f"Avg time: {avg_latency:.2f}s"
    )

    return summary


def _parse_models(
    single: str | None,
    many: str | None,
) -> list[str | None]:
    if many:
        models = [m.strip() for m in many.split(",") if m.strip()]
        return models or [single]

    return [single]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run agent evals.")
    p.add_argument("--suite", default="plumber")
    p.add_argument("--mock", action="store_true")
    p.add_argument(
        "--model",
        help="override the agent.yaml completion model for this run",
    )
    p.add_argument(
        "--models",
        help="comma-separated completion models to compare",
    )
    args = p.parse_args(argv)

    mock = True if args.mock else None

    if mock is None and not get_secret("OPENROUTER_API_KEY"):
        print(
            "ℹ️ OPENROUTER_API_KEY not set — running evals in MOCK mode "
            "(content checks will not be meaningful).\n"
        )
        mock = True

    models = _parse_models(args.model, args.models)
    summaries = []

    for i, model in enumerate(models):
        if i:
            print()

        summary = run_suite(
            args.suite,
            mock,
            model_override=model,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        print("\n=== model comparison summary ===")

        for summary in summaries:
            print(
                f"{summary['model']}: "
                f"{summary['passed_checks']}/{summary['total_checks']} checks "
                f"({summary['pass_rate']:.1%}) | "
                f"${summary['total_cost_usd']:.6f} | "
                f"{summary['total_tokens']} tokens | "
                f"{summary['avg_latency_seconds']:.2f}s avg"
            )

    # Keep summaries available to other Python code that calls main's runner.
    # run_suite() itself returns the full summary dictionary.
    all_passed = all(summary["success"] for summary in summaries)

    _save_results(args.suite, summaries)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
