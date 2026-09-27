from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml
from pydantic import ValidationError

from loopeval.config import LoopEvalConfig, load_config
from loopeval.io import load_samples
from loopeval.models import (
    EvaluationReport,
    OverallVerdict,
    ProviderUsage,
    SampleResult,
)
from loopeval.reporting import compare_reports, gate_failures, report_summary, write_junit_report


def test_load_config_resolves_project_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "checks").mkdir()
    raw = {
        "providers": {"decision": {"type": "mock"}, "fallback": {"type": "disabled"}},
        "checks": ["checks"],
        "storage": {"directory": "artifacts"},
    }
    path = tmp_path / "loopeval.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config.checks == [str((tmp_path / "checks").resolve())]
    assert config.storage.directory == str((tmp_path / "artifacts").resolve())
    assert len(config.fingerprint()) == 16


def test_policy_threshold_validation() -> None:
    with pytest.raises(ValidationError, match="must be below"):
        LoopEvalConfig.model_validate(
            {
                "providers": {"decision": {"type": "mock"}},
                "escalation": {
                    "coverage_pass_threshold": 0.8,
                    "coverage_failure_threshold": 0.2,
                },
            }
        )


def test_provider_configuration_is_complete() -> None:
    with pytest.raises(ValidationError, match="requires model"):
        LoopEvalConfig.model_validate(
            {"providers": {"decision": {"type": "openrouter_decisions", "api_key_env": "KEY"}}}
        )
    with pytest.raises(ValidationError, match="cannot be used as a decision provider"):
        LoopEvalConfig.model_validate(
            {
                "providers": {
                    "decision": {
                        "type": "openai",
                        "model": "gpt-test",
                        "api_key_env": "KEY",
                    }
                }
            }
        )
    with pytest.raises(ValidationError, match="requires base_url"):
        LoopEvalConfig.model_validate(
            {
                "providers": {
                    "decision": {"type": "mock"},
                    "fallback": {
                        "type": "openai_compatible",
                        "model": "custom",
                        "api_key_env": "KEY",
                    },
                }
            }
        )
    local = LoopEvalConfig.model_validate(
        {
            "providers": {
                "decision": {"type": "mock"},
                "fallback": {
                    "type": "openai_compatible",
                    "model": "local-model",
                    "base_url": "http://localhost:11434/v1",
                },
            }
        }
    )
    assert local.providers.fallback.api_key_env is None


def test_load_json_and_jsonl(tmp_path: Path) -> None:
    json_path = tmp_path / "samples.json"
    json_path.write_text(json.dumps({"id": "one", "input": "x"}))
    assert load_samples(json_path)[0].sample_id == "one"
    jsonl_path = tmp_path / "samples.jsonl"
    jsonl_path.write_text('\n{"id":"two","input":"y"}\n')
    assert load_samples(jsonl_path)[0].sample_id == "two"
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    with pytest.raises(ValueError, match=":1: invalid sample"):
        load_samples(bad)
    unsupported = tmp_path / "samples.csv"
    unsupported.write_text("input\nx\n")
    with pytest.raises(ValueError, match=r"must be \.jsonl or \.json"):
        load_samples(unsupported)


def report(run_id: str, escalated: bool, cost: float) -> EvaluationReport:
    now = datetime.now(UTC)
    return EvaluationReport(
        run_id=run_id,
        started_at=now,
        completed_at=now,
        config_hash="abc",
        results=[
            SampleResult(
                sample_id="one",
                verdict=OverallVerdict.FAIL if escalated else OverallVerdict.PASS,
                checks=[],
                escalated=escalated,
                escalation_reasons=["audit"] if escalated else [],
                usage=ProviderUsage(cost_usd=cost),
            )
        ],
    )


def test_report_summary_and_comparison() -> None:
    before = report("before", True, 0.5)
    after = report("after", False, 0.1)
    summary = report_summary(before)
    assert summary["verdicts"] == {"fail": 1}
    assert summary["escalation_rate"] == 1.0
    assert summary["failed_samples"] == [
        {"sample_id": "one", "failed_checks": [], "fallback_category": None}
    ]
    comparison = compare_reports(before, after)
    assert comparison["escalation_rate_change"] == -1.0
    assert comparison["cost_change"] == pytest.approx(-0.4)


def test_evaluation_gates_and_junit_report(tmp_path: Path) -> None:
    value = report("failed", True, 0.5)
    failures = gate_failures(
        value,
        fail_on={OverallVerdict.FAIL},
        max_failures=0,
        max_unresolved_rate=0,
        max_cost_usd=0.1,
    )
    assert len(failures) == 3
    junit = write_junit_report(value, tmp_path / "reports" / "loopeval.xml")
    suite = ElementTree.parse(junit).getroot()
    assert suite.attrib["tests"] == "1"
    assert suite.attrib["failures"] == "1"
    assert suite.find("testcase/failure") is not None

    unknown_cost = report("unknown", False, 0)
    unknown_cost.results[0].usage = ProviderUsage()
    assert "cost is unknown" in gate_failures(unknown_cost, max_cost_usd=1)[0]


def test_empty_report_properties() -> None:
    now = datetime.now(UTC)
    empty = EvaluationReport(
        run_id="empty", results=[], started_at=now, completed_at=now, config_hash="abc"
    )
    assert empty.escalation_rate == 0
    assert empty.total_cost_usd is None
