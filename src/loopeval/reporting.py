from __future__ import annotations

from collections import Counter
from typing import Any

from .models import EvaluationReport


def report_summary(report: EvaluationReport) -> dict[str, Any]:
    verdicts = Counter(result.verdict.value for result in report.results)
    reasons = Counter(reason for result in report.results for reason in result.escalation_reasons)
    failures = Counter(
        check.check_id
        for result in report.results
        for check in result.checks
        if check.status.value == "fail"
    )
    return {
        "run_id": report.run_id,
        "samples": report.sample_count,
        "verdicts": dict(verdicts),
        "escalated": sum(result.escalated for result in report.results),
        "escalation_rate": report.escalation_rate,
        "escalation_reasons": dict(reasons),
        "failed_checks": dict(failures),
        "input_tokens": sum(result.usage.input_tokens for result in report.results),
        "output_tokens": sum(result.usage.output_tokens for result in report.results),
        "cost_usd": report.total_cost_usd,
        "latency_ms": sum(result.latency_ms for result in report.results),
    }


def compare_reports(before: EvaluationReport, after: EvaluationReport) -> dict[str, Any]:
    a = report_summary(before)
    b = report_summary(after)
    return {
        "before": before.run_id,
        "after": after.run_id,
        "sample_count_change": b["samples"] - a["samples"],
        "escalation_rate_before": a["escalation_rate"],
        "escalation_rate_after": b["escalation_rate"],
        "escalation_rate_change": b["escalation_rate"] - a["escalation_rate"],
        "cost_before": a["cost_usd"],
        "cost_after": b["cost_usd"],
        "cost_change": (
            b["cost_usd"] - a["cost_usd"]
            if a["cost_usd"] is not None and b["cost_usd"] is not None
            else None
        ),
        "verdicts_before": a["verdicts"],
        "verdicts_after": b["verdicts"],
    }
