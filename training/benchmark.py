"""Metrics for independently verified outcomes from equivalent benchmark tasks."""

from __future__ import annotations

from statistics import median
from typing import Any


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"task_count": 0, "completion_rate": None, "test_pass_rate": None, "tool_call_validity_rate": None, "format_validity_rate": None, "regression_rate": None, "unsupported_success_claim_rate": None, "median_duration_seconds": None}
    def rate(key: str) -> float | None:
        values = [row[key] for row in results if isinstance(row.get(key), bool)]
        return sum(values) / len(values) if values else None
    invalid_claims = [row for row in results if row.get("model_claimed_success") is True and row.get("independently_verified") is not True]
    durations = [float(row["duration_seconds"]) for row in results if isinstance(row.get("duration_seconds"), (int, float)) and row["duration_seconds"] >= 0]
    return {
        "task_count": len(results),
        "completion_rate": rate("independently_verified"),
        "test_pass_rate": rate("tests_passed"),
        "tool_call_validity_rate": rate("tool_calls_valid"),
        "format_validity_rate": rate("format_valid"),
        "regression_rate": rate("regression_detected"),
        "unsupported_success_claim_rate": len(invalid_claims) / len(results),
        "median_duration_seconds": median(durations) if durations else None,
        "unsupported_success_claim_count": len(invalid_claims),
    }


def compare_results(base: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    base_by_id = {str(row["task_id"]): row for row in base if row.get("task_id") is not None}
    candidate_by_id = {str(row["task_id"]): row for row in candidate if row.get("task_id") is not None}
    common = sorted(base_by_id.keys() & candidate_by_id.keys())
    if len(common) != len(base) or len(common) != len(candidate):
        raise ValueError("Base and candidate results must contain exactly the same unique task IDs")
    return {"task_ids": common, "base": summarize_results([base_by_id[key] for key in common]), "candidate": summarize_results([candidate_by_id[key] for key in common])}
