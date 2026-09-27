from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from loopeval.config import ProviderConfig
from loopeval.models import (
    CheckLifecycle,
    CheckSpec,
    DecisionAnswer,
    DecisionResponse,
    EvalSample,
    OverallVerdict,
    ProviderUsage,
    ResultStatus,
)
from loopeval.providers.base import DecisionProvider, GenerativeProvider, ProviderError
from loopeval.providers.mock import MockDecisionProvider, MockGenerativeProvider
from loopeval.runtime import Evaluator, interpret_answer
from loopeval.storage import LocalStore


def evaluator(config: Any, checks: list[CheckSpec], tmp_path: Path) -> Evaluator:
    return Evaluator(
        config=config,
        checks=checks,
        decision_provider=MockDecisionProvider(ProviderConfig(type="mock")),
        fallback_provider=MockGenerativeProvider(ProviderConfig(type="mock")),
        store=LocalStore(tmp_path / "db"),
    )


@pytest.mark.asyncio
async def test_deterministic_failure_short_circuits(
    config_factory: Any, not_empty_check: CheckSpec, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    runtime = evaluator(config_factory(), [not_empty_check, semantic_check], tmp_path)
    result = await runtime.evaluate_sample(EvalSample(id="empty", input="x", output=""))
    assert result.verdict == OverallVerdict.FAIL
    assert [item.check_id for item in result.checks] == [not_empty_check.id]
    assert not result.escalated
    runtime.store.close()


@pytest.mark.asyncio
async def test_healthy_and_known_failure_paths(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    runtime = evaluator(config_factory(), [semantic_check], tmp_path)
    healthy = await runtime.evaluate_sample(EvalSample(id="ok", input="x", output="fine"))
    failed = await runtime.evaluate_sample(EvalSample(id="bad", input="x", output="[FAIL]"))
    assert healthy.verdict == OverallVerdict.PASS
    assert failed.verdict == OverallVerdict.FAIL
    assert failed.checks[0].status == ResultStatus.FAIL
    runtime.store.close()


@pytest.mark.asyncio
async def test_uncertain_and_novel_paths_create_candidate(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    runtime = evaluator(config_factory(), [semantic_check], tmp_path)
    uncertain = await runtime.evaluate_sample(
        EvalSample(id="uncertain", input="x", output="[UNCERTAIN]")
    )
    novel = await runtime.evaluate_sample(EvalSample(id="novel", input="x", output="[NOVEL]"))
    assert uncertain.escalated and uncertain.verdict == OverallVerdict.PASS
    assert "uncertain_check" in uncertain.escalation_reasons
    assert novel.verdict == OverallVerdict.FAIL
    assert novel.fallback and novel.fallback.verdict.category == "novel_failure"
    assert len(runtime.store.list_candidates()) == 1
    runtime.store.close()


class BrokenDecision(DecisionProvider):
    name = "broken"
    model = "broken-1"

    async def decide(self, state: dict[str, Any], questions: list[Any]) -> DecisionResponse:
        raise ProviderError("boom")


class CountingFallback(GenerativeProvider):
    name = "counting"
    model = "counting-1"

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], ProviderUsage, int]:
        self.calls += 1
        return (
            {
                "category": "acceptable",
                "existing_check_id": None,
                "label": "acceptable",
                "title": "Acceptable",
                "description": "No issue.",
                "severity": "info",
                "evidence": "No material issue found.",
                "confidence": 0.9,
                "candidate_check": None,
            },
            ProviderUsage(input_tokens=10, output_tokens=2, cost_usd=0.01),
            1,
        )


class CountingDecision(MockDecisionProvider):
    def __init__(self, *, cost_usd: float | None = 0.03) -> None:
        super().__init__(ProviderConfig(type="mock"))
        self.calls = 0
        self.cost_usd = cost_usd

    async def decide(self, state: dict[str, Any], questions: list[Any]) -> DecisionResponse:
        self.calls += 1
        response = await super().decide(state, questions)
        return response.model_copy(
            update={
                "usage": ProviderUsage(input_tokens=20, cost_usd=self.cost_usd),
                "latency_ms": 123,
            }
        )


@pytest.mark.asyncio
async def test_provider_error_without_fallback_is_unresolved(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    config = config_factory(
        providers={
            "decision": {"type": "mock"},
            "fallback": {"type": "disabled"},
        },
        escalation={"random_audit_rate": 0, "coverage_check": False},
    )
    runtime = Evaluator(
        config=config,
        checks=[semantic_check],
        decision_provider=BrokenDecision(),
        fallback_provider=None,
        store=LocalStore(tmp_path / "db"),
    )
    result = await runtime.evaluate_sample(EvalSample(input="x", output="y"))
    assert result.verdict == OverallVerdict.UNRESOLVED
    assert result.checks[0].status == ResultStatus.ERROR
    assert "decision_provider_error" in result.escalation_reasons
    assert "fallback_disabled" in result.escalation_reasons
    runtime.store.close()


@pytest.mark.asyncio
async def test_random_fence_does_not_defeat_fallback_cache(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    fallback = CountingFallback()
    decision = CountingDecision()
    runtime = Evaluator(
        config=config_factory(),
        checks=[semantic_check],
        decision_provider=decision,
        fallback_provider=fallback,
        store=LocalStore(tmp_path / "db"),
    )
    sample = EvalSample(id="same", input="x", output="[UNCERTAIN]")
    first = await runtime.evaluate_sample(sample)
    second = await runtime.evaluate_sample(sample)
    assert fallback.calls == 1
    assert decision.calls == 1
    assert first.usage.cost_usd == pytest.approx(0.04)
    assert second.usage.cost_usd == 0.0
    runtime.store.close()


def test_interpret_choice_and_score() -> None:
    response = DecisionResponse(
        answers={}, provider="test", model="test", usage=ProviderUsage(input_tokens=2)
    )
    choice = CheckSpec(
        id="quality.choice",
        name="Choice",
        description="Choice",
        kind="choice",
        instructions="Choose",
        criteria={"ok": "ok", "bad": "bad"},
        failure_labels=["bad"],
    )
    score = CheckSpec(
        id="quality.score",
        name="Score",
        description="Score",
        kind="score",
        instructions="Score",
        criteria=["ok", "bad"],
        failure_score_gte=1,
    )
    bad = DecisionAnswer(
        kind="choice", choice="bad", confidence=0.9, probabilities={"bad": 0.9, "ok": 0.1}
    )
    low_confidence = DecisionAnswer(
        kind="score", score=1, confidence=0.2, probabilities={"0": 0.6, "1": 0.4}
    )
    assert interpret_answer(choice, bad, response).status == ResultStatus.FAIL
    assert interpret_answer(score, low_confidence, response).status == ResultStatus.UNCERTAIN


def test_shadow_checks_are_excluded_and_duplicate_active_ids_rejected(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    shadow = semantic_check.model_copy(update={"lifecycle": CheckLifecycle.SHADOW})
    runtime = evaluator(config_factory(), [shadow], tmp_path)
    assert runtime.checks == []
    runtime.store.close()
    duplicate = semantic_check.model_copy(update={"version": "2.0.0"})
    store = LocalStore(tmp_path / "duplicate-db")
    try:
        with pytest.raises(ValueError, match="one active version"):
            Evaluator(
                config=config_factory(),
                checks=[semantic_check, duplicate],
                decision_provider=MockDecisionProvider(ProviderConfig(type="mock")),
                fallback_provider=None,
                store=store,
            )
    finally:
        store.close()


@pytest.mark.parametrize("invalid_score", [-1, 2])
def test_interpret_rejects_out_of_range_scores(invalid_score: float) -> None:
    check = CheckSpec(
        id="quality.score",
        name="Score",
        description="Score",
        kind="score",
        instructions="Score",
        criteria=["good", "bad"],
        failure_score_gte=1,
    )
    answer = DecisionAnswer(
        kind="score", score=invalid_score, confidence=0.9, probabilities={"0": 1.0}
    )
    result = interpret_answer(check, answer, DecisionResponse(answers={}, provider="x", model="x"))
    assert result.status == ResultStatus.ERROR


@pytest.mark.parametrize("invalid_score", [float("inf"), float("nan")])
def test_decision_answer_rejects_nonfinite_scores(invalid_score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        DecisionAnswer(kind="score", score=invalid_score, confidence=0.9, probabilities={"0": 1.0})


def test_interpret_rejects_unknown_choices_and_wrong_kinds(semantic_check: CheckSpec) -> None:
    check = CheckSpec(
        id="quality.choice",
        name="Choice",
        description="Choice",
        kind="choice",
        instructions="Choose",
        criteria={"good": "good", "bad": "bad"},
        failure_labels=["bad"],
    )
    answer = DecisionAnswer(
        kind="choice", choice="unexpected", confidence=0.9, probabilities={"unexpected": 1.0}
    )
    response = DecisionResponse(answers={}, provider="x", model="x")
    assert interpret_answer(check, answer, response).status == ResultStatus.ERROR
    assert interpret_answer(semantic_check, answer, response).status == ResultStatus.ERROR


@pytest.mark.asyncio
async def test_report_is_persisted(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    runtime = evaluator(config_factory(), [semantic_check], tmp_path)
    report = await runtime.evaluate([EvalSample(id="one", input="x", output="fine")])
    assert report.sample_count == 1
    assert report.results[0].sample_hash is not None
    assert report.dataset_hash is not None
    assert report.checks_hash is not None
    assert runtime.store.get_report(report.run_id) == report
    assert (runtime.store.runs_dir / f"{report.run_id}.jsonl").exists()
    with pytest.raises(ValueError, match="sample ids must be unique"):
        await runtime.evaluate(
            [
                EvalSample(id="duplicate", input="x"),
                EvalSample(id="duplicate", input="y"),
            ]
        )
    runtime.store.close()


@pytest.mark.asyncio
async def test_empty_runs_are_rejected_before_storage(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    runtime = evaluator(config_factory(), [semantic_check], tmp_path)
    runtime.store.start_run = Mock(side_effect=AssertionError("must not start an empty run"))
    with pytest.raises(ValueError, match="at least one sample"):
        await runtime.evaluate([])
    runtime.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", [False, True])
async def test_disabling_escalation_does_not_convert_unknowns_to_passes(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path, broken: bool
) -> None:
    config = config_factory(
        escalation={
            "random_audit_rate": 0,
            "coverage_check": False,
            "on_uncertain": False,
            "on_decision_error": False,
        }
    )
    fallback = CountingFallback()
    runtime = Evaluator(
        config=config,
        checks=[semantic_check],
        decision_provider=BrokenDecision() if broken else CountingDecision(),
        fallback_provider=fallback,
        store=LocalStore(tmp_path / "db"),
    )
    result = await runtime.evaluate_sample(EvalSample(input="x", output="[UNCERTAIN]"))
    assert result.verdict == OverallVerdict.UNRESOLVED
    assert fallback.calls == 0
    runtime.store.close()


@pytest.mark.asyncio
async def test_semantic_checks_report_missing_evidence(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    check = semantic_check.model_copy(update={"requires": ["input", "context"]})
    config = config_factory(
        escalation={
            "random_audit_rate": 0,
            "coverage_check": False,
            "on_no_applicable_checks": False,
        }
    )
    runtime = evaluator(config, [check], tmp_path)
    result = await runtime.evaluate_sample(EvalSample(input="x", output="fine"))
    assert result.verdict == OverallVerdict.UNRESOLVED
    assert len(result.checks) == 1
    assert result.checks[0].status == ResultStatus.SKIPPED
    assert result.checks[0].evidence == {"missing_required_fields": ["context"]}
    assert result.usage.cost_usd == 0.0
    runtime.store.close()


@pytest.mark.asyncio
async def test_oversized_state_is_never_silently_truncated(
    config_factory: Any, semantic_check: CheckSpec, not_empty_check: CheckSpec, tmp_path: Path
) -> None:
    decision = CountingDecision()
    fallback = CountingFallback()
    runtime = Evaluator(
        config=config_factory(budgets={"max_state_chars": 1000}),
        checks=[not_empty_check, semantic_check],
        decision_provider=decision,
        fallback_provider=fallback,
        store=LocalStore(tmp_path / "db"),
    )
    sample = EvalSample(input="x", output="fine", context=["x"] * 1000)
    result = await runtime.evaluate_sample(sample)
    assert result.verdict == OverallVerdict.UNRESOLVED
    assert "state_size_exceeded" in result.escalation_reasons
    assert result.checks[0].status == ResultStatus.PASS
    assert result.checks[1].status == ResultStatus.ERROR
    assert result.usage.cost_usd == 0.0
    assert decision.calls == fallback.calls == 0
    exact = await runtime.evaluate_sample(sample, checks=[not_empty_check], enable_coverage=False)
    assert exact.verdict == OverallVerdict.PASS
    runtime.store.close()


@pytest.mark.asyncio
async def test_unknown_provider_cost_stays_unknown(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    runtime = Evaluator(
        config=config_factory(),
        checks=[semantic_check],
        decision_provider=CountingDecision(cost_usd=None),
        fallback_provider=CountingFallback(),
        store=LocalStore(tmp_path / "db"),
    )
    result = await runtime.evaluate_sample(EvalSample(input="x", output="[UNCERTAIN]"))
    assert result.fallback is not None
    assert result.fallback.usage.cost_usd == 0.01
    assert result.usage.cost_usd is None
    runtime.store.close()


@pytest.mark.asyncio
async def test_fallback_cannot_override_exact_failure(
    config_factory: Any, not_empty_check: CheckSpec, tmp_path: Path
) -> None:
    config = config_factory(
        escalation={
            "short_circuit_on_deterministic_failure": False,
            "coverage_check": False,
            "random_audit_rate": 1,
        }
    )
    runtime = evaluator(config, [not_empty_check], tmp_path)
    result = await runtime.evaluate_sample(EvalSample(input="x", output=""))
    assert result.fallback is not None and result.fallback.verdict.category == "acceptable"
    assert result.verdict == OverallVerdict.FAIL
    runtime.store.close()


@pytest.mark.asyncio
async def test_fallback_limit_is_shared_across_concurrent_samples(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    config = config_factory(
        escalation={"coverage_check": False, "random_audit_rate": 0, "max_fallbacks_per_run": 2},
        storage={"cache": False},
        budgets={"concurrency": 8},
    )
    fallback = CountingFallback()
    runtime = Evaluator(
        config=config,
        checks=[semantic_check],
        decision_provider=CountingDecision(cost_usd=0),
        fallback_provider=fallback,
        store=LocalStore(tmp_path / "db"),
    )
    report = await runtime.evaluate(
        [EvalSample(id=str(index), input="x", output="[UNCERTAIN]") for index in range(8)]
    )
    assert fallback.calls == 2
    assert sum(result.verdict == OverallVerdict.PASS for result in report.results) == 2
    assert sum("fallback_budget_exhausted" in r.escalation_reasons for r in report.results) == 6
    assert report.total_cost_usd == pytest.approx(0.02)
    runtime.store.close()


@pytest.mark.asyncio
async def test_fallback_evidence_limit_does_not_consume_a_call_or_invent_cost(
    config_factory,
    semantic_check,
    tmp_path,
) -> None:
    decision = CountingDecision(cost_usd=0.03)
    fallback = CountingFallback()
    runtime = Evaluator(
        config=config_factory(budgets={"max_state_chars": 1000}),
        checks=[semantic_check],
        decision_provider=decision,
        fallback_provider=fallback,
        store=LocalStore(tmp_path / "db"),
    )
    try:
        result = await runtime.evaluate_sample(
            EvalSample(input="x", output="[UNCERTAIN]" + "x" * 700)
        )
        assert result.verdict == OverallVerdict.UNRESOLVED
        assert "fallback_state_size_exceeded" in result.escalation_reasons
        assert result.usage.cost_usd == 0.03
        assert decision.calls == 1
        assert fallback.calls == runtime._fallback_count == 0
    finally:
        runtime.store.close()


class InvalidFallback(CountingFallback):
    async def generate_structured(self, **kwargs: Any) -> tuple[dict[str, Any], ProviderUsage, int]:
        raw, usage, latency = await super().generate_structured(**kwargs)
        return (
            {**raw, "category": "existing_failure", "existing_check_id": "unknown"},
            usage,
            latency,
        )


@pytest.mark.asyncio
async def test_invalid_billed_fallback_keeps_usage_and_counts_toward_budget(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    fallback = InvalidFallback()
    runtime = Evaluator(
        config=config_factory(
            escalation={"coverage_check": False, "random_audit_rate": 0},
            budgets={"run_cost_usd": 0.01, "concurrency": 1},
        ),
        checks=[semantic_check],
        decision_provider=CountingDecision(cost_usd=0),
        fallback_provider=fallback,
        store=LocalStore(tmp_path / "db"),
    )
    report = await runtime.evaluate(
        [EvalSample(id=str(index), input="x", output="[UNCERTAIN]") for index in range(2)]
    )
    assert all(result.verdict == OverallVerdict.UNRESOLVED for result in report.results)
    assert report.results[0].usage.cost_usd == 0.01
    assert "fallback_error:ValueError" in report.results[0].escalation_reasons
    assert "fallback_budget_exhausted" in report.results[1].escalation_reasons
    assert report.total_cost_usd == 0.01
    assert fallback.calls == 1
    runtime.store.close()
