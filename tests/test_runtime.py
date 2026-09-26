from __future__ import annotations

from pathlib import Path
from typing import Any

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
from loopeval.runtime import Evaluator, _truncate, interpret_answer
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
    runtime = Evaluator(
        config=config_factory(),
        checks=[semantic_check],
        decision_provider=MockDecisionProvider(ProviderConfig(type="mock")),
        fallback_provider=fallback,
        store=LocalStore(tmp_path / "db"),
    )
    sample = EvalSample(id="same", input="x", output="[UNCERTAIN]")
    first = await runtime.evaluate_sample(sample)
    second = await runtime.evaluate_sample(sample)
    assert fallback.calls == 1
    assert first.usage.cost_usd == pytest.approx(0.01)
    assert second.usage.cost_usd is None
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
        kind="score", score=2, confidence=0.2, probabilities={"2": 0.6, "1": 0.4}
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


def test_truncate_bounds_strings_and_structures() -> None:
    assert _truncate("abcdefgh", 4).startswith("abcd")
    assert _truncate({"a": "x"}, 100) == {"a": "x"}
    assert _truncate(["x"], 100) == ["x"]
    assert _truncate("x", 0) == "[truncated]"


@pytest.mark.asyncio
async def test_report_is_persisted(
    config_factory: Any, semantic_check: CheckSpec, tmp_path: Path
) -> None:
    runtime = evaluator(config_factory(), [semantic_check], tmp_path)
    report = await runtime.evaluate([EvalSample(id="one", input="x", output="fine")])
    assert report.sample_count == 1
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
