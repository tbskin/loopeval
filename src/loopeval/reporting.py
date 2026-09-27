from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .models import EvaluationReport, OverallVerdict


def report_summary(report: EvaluationReport) -> dict[str, Any]:
    verdicts = Counter(result.verdict.value for result in report.results)
    reasons = Counter(reason for result in report.results for reason in result.escalation_reasons)
    failures = Counter(
        check.check_id
        for result in report.results
        for check in result.checks
        if check.status.value == "fail"
    )
    failed_samples = []
    unresolved_samples = []
    for result in report.results:
        failed_check_ids = [
            check.check_id for check in result.checks if check.status.value == "fail"
        ]
        if result.verdict == OverallVerdict.FAIL:
            failed_samples.append(
                {
                    "sample_id": result.sample_id,
                    "failed_checks": failed_check_ids,
                    "fallback_category": (
                        result.fallback.verdict.category if result.fallback else None
                    ),
                }
            )
        elif result.verdict == OverallVerdict.UNRESOLVED:
            unresolved_samples.append(
                {
                    "sample_id": result.sample_id,
                    "reasons": result.escalation_reasons,
                }
            )
    return {
        "run_id": report.run_id,
        "samples": report.sample_count,
        "verdicts": dict(verdicts),
        "escalated": sum(result.escalated for result in report.results),
        "escalation_rate": report.escalation_rate,
        "escalation_reasons": dict(reasons),
        "failed_checks": dict(failures),
        "failed_samples": failed_samples,
        "unresolved_samples": unresolved_samples,
        "input_tokens": sum(result.usage.input_tokens for result in report.results),
        "output_tokens": sum(result.usage.output_tokens for result in report.results),
        "cost_usd": report.total_cost_usd,
        "latency_ms": sum(result.latency_ms for result in report.results),
    }


def gate_failures(
    report: EvaluationReport,
    *,
    fail_on: set[OverallVerdict] | None = None,
    max_failures: int | None = None,
    max_unresolved_rate: float | None = None,
    max_cost_usd: float | None = None,
) -> list[str]:
    reasons: list[str] = []
    verdict_counts = Counter(result.verdict for result in report.results)
    for verdict in sorted(fail_on or set(), key=str):
        count = verdict_counts[verdict]
        if count:
            reasons.append(f"{count} sample(s) had verdict {verdict.value}")
    failure_count = verdict_counts[OverallVerdict.FAIL]
    if max_failures is not None and failure_count > max_failures:
        reasons.append(f"{failure_count} failures exceeded maximum {max_failures}")
    unresolved_count = verdict_counts[OverallVerdict.UNRESOLVED]
    unresolved_rate = unresolved_count / report.sample_count if report.sample_count else 0.0
    if max_unresolved_rate is not None and unresolved_rate > max_unresolved_rate:
        reasons.append(
            f"unresolved rate {unresolved_rate:.3f} exceeded maximum {max_unresolved_rate:.3f}"
        )
    if max_cost_usd is not None:
        if report.total_cost_usd is None:
            reasons.append("run cost is unknown, so the configured maximum cannot be verified")
        elif report.total_cost_usd > max_cost_usd:
            reasons.append(
                f"run cost ${report.total_cost_usd:.6f} exceeded maximum ${max_cost_usd:.6f}"
            )
    return reasons


def write_junit_report(report: EvaluationReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    failures = sum(result.verdict == OverallVerdict.FAIL for result in report.results)
    errors = sum(result.verdict == OverallVerdict.UNRESOLVED for result in report.results)
    suite = ElementTree.Element(
        "testsuite",
        {
            "name": "LoopEval",
            "tests": str(report.sample_count),
            "failures": str(failures),
            "errors": str(errors),
            "time": f"{sum(result.latency_ms for result in report.results) / 1000:.3f}",
            "timestamp": report.completed_at.isoformat(),
        },
    )
    for result in report.results:
        case = ElementTree.SubElement(
            suite,
            "testcase",
            {
                "classname": "loopeval",
                "name": result.sample_id,
                "time": f"{result.latency_ms / 1000:.3f}",
            },
        )
        if result.verdict == OverallVerdict.FAIL:
            failed_checks = [
                check.check_id for check in result.checks if check.status.value == "fail"
            ]
            failure = ElementTree.SubElement(
                case,
                "failure",
                {"message": "LoopEval verdict: fail", "type": "evaluation_failure"},
            )
            failure.text = ", ".join(failed_checks) or (
                result.fallback.verdict.evidence if result.fallback else "fallback failure"
            )
        elif result.verdict == OverallVerdict.UNRESOLVED:
            error = ElementTree.SubElement(
                case,
                "error",
                {"message": "LoopEval verdict: unresolved", "type": "unresolved_evaluation"},
            )
            error.text = ", ".join(result.escalation_reasons)
        output = ElementTree.SubElement(case, "system-out")
        output.text = result.model_dump_json()
    ElementTree.indent(suite)
    destination.write_bytes(
        ElementTree.tostring(suite, encoding="utf-8", xml_declaration=True)
    )
    return destination


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
