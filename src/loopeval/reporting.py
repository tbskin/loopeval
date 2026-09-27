from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .models import EvaluationReport, OverallVerdict


def _classification_metrics(predictions: list[bool | None], labels: list[bool]) -> dict[str, Any]:
    tp = fp = tn = fn = unresolved = 0
    for predicted, actual in zip(predictions, labels, strict=True):
        if predicted is None:
            unresolved += 1
        elif predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    resolved = tp + fp + tn + fn
    return {
        "examples": len(labels),
        "resolved": resolved,
        "unresolved": unresolved,
        "coverage": resolved / len(labels) if labels else 0.0,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / resolved if resolved else 0.0,
    }


def labeled_run_metrics(report: EvaluationReport) -> dict[str, Any]:
    verdict_labeled = [result for result in report.results if result.expected_verdict is not None]
    correct = sum(result.verdict == result.expected_verdict for result in verdict_labeled)
    confusion: dict[str, Counter[str]] = {}
    for result in verdict_labeled:
        expected = result.expected_verdict.value if result.expected_verdict else ""
        confusion.setdefault(expected, Counter())[result.verdict.value] += 1

    check_labeled = [result for result in report.results if result.expected_labels is not None]
    check_ids = sorted(
        {
            check_id
            for result in check_labeled
            for check_id in [
                *(result.expected_labels or []),
                *(check.check_id for check in result.checks),
            ]
        }
    )
    per_check: dict[str, Any] = {}
    for check_id in check_ids:
        predictions: list[bool | None] = []
        labels: list[bool] = []
        for result in check_labeled:
            check_result = next((check for check in result.checks if check.check_id == check_id), None)
            if check_result is None or check_result.status.value not in {"pass", "fail"}:
                predictions.append(None)
            else:
                predictions.append(check_result.status.value == "fail")
            labels.append(check_id in (result.expected_labels or []))
        per_check[check_id] = _classification_metrics(predictions, labels)
    return {
        "labeled_samples": len(verdict_labeled),
        "correct_verdicts": correct,
        "verdict_accuracy": correct / len(verdict_labeled) if verdict_labeled else None,
        "verdict_confusion": {key: dict(value) for key, value in confusion.items()},
        "check_metrics": per_check,
    }


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
        **labeled_run_metrics(report),
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
        "labeled_samples_before": a["labeled_samples"],
        "labeled_samples_after": b["labeled_samples"],
        "verdict_accuracy_before": a["verdict_accuracy"],
        "verdict_accuracy_after": b["verdict_accuracy"],
        "verdict_accuracy_change": (
            b["verdict_accuracy"] - a["verdict_accuracy"]
            if a["verdict_accuracy"] is not None and b["verdict_accuracy"] is not None
            else None
        ),
        "check_metrics_before": a["check_metrics"],
        "check_metrics_after": b["check_metrics"],
    }
