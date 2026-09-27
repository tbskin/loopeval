from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
from datetime import UTC, datetime
from typing import Any

from .checks import RULES, applies, missing_required_fields, run_deterministic
from .config import LoopEvalConfig
from .learning import FALLBACK_SYSTEM, fallback_prompt, fallback_schema, ingest_candidate
from .models import (
    CheckKind,
    CheckLifecycle,
    CheckResult,
    CheckSpec,
    DecisionAnswer,
    DecisionResponse,
    EvalSample,
    EvaluationReport,
    FallbackResult,
    FallbackVerdict,
    OverallVerdict,
    ProviderUsage,
    QuestionKind,
    ResultStatus,
    SampleResult,
    TypedQuestion,
    checks_fingerprint,
    dataset_fingerprint,
)
from .providers import DecisionProvider, GenerativeProvider
from .storage import LocalStore

COVERAGE_ID = "__loopeval_coverage__"


def _cache_key(kind: str, provider: str, payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{kind}:{provider}:{canonical}".encode()).hexdigest()


class _InvalidFallbackResponse(ValueError):
    def __init__(self, error: Exception, usage: ProviderUsage) -> None:
        super().__init__(str(error))
        self.error_type = type(error).__name__
        self.usage = usage


class _FallbackEvidenceTooLarge(ValueError):
    pass


def _combine_usage(*usages: ProviderUsage) -> ProviderUsage:
    return ProviderUsage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        cost_usd=(
            sum(usage.cost_usd for usage in usages if usage.cost_usd is not None)
            if all(usage.cost_usd is not None for usage in usages)
            else None
        ),
    )


def _coverage_question(checks: list[CheckSpec]) -> TypedQuestion:
    descriptions = "; ".join(f"{check.id}: {check.description}" for check in checks)
    return TypedQuestion(
        id=COVERAGE_ID,
        kind=QuestionKind.NOUL,
        instructions=(
            "Does this evaluation sample contain a material quality, safety, correctness, or "
            "task-completion failure that is NOT captured by any active check listed here? "
            f"Active checks: {descriptions or '[none]'}."
        ),
        criteria={
            "true": "A material reusable failure exists and none of the active checks describe it.",
            "false": "The sample is acceptable, or every material failure is covered by an active check.",
        },
    )


def interpret_answer(
    check: CheckSpec, answer: DecisionAnswer, response: DecisionResponse
) -> CheckResult:
    def make_result(
        status: ResultStatus,
        *,
        score: float | None = None,
        label: str | None = None,
        error: str | None = None,
    ) -> CheckResult:
        return CheckResult(
            check_id=check.id,
            check_version=check.version,
            severity=check.severity,
            provider=response.provider,
            model=response.model,
            latency_ms=response.latency_ms,
            usage=response.usage,
            probabilities=answer.probabilities,
            confidence=answer.confidence,
            status=status,
            score=score,
            label=label,
            error=error,
        )

    if answer.kind.value != check.kind.value:
        return make_result(ResultStatus.ERROR, error="answer kind does not match the check")

    if check.kind == CheckKind.NOUL:
        if answer.noul is None:
            return make_result(ResultStatus.ERROR, error="missing noul value")
        if answer.noul >= check.failure_threshold:
            status = ResultStatus.FAIL
        elif answer.noul <= check.pass_threshold:
            status = ResultStatus.PASS
        else:
            status = ResultStatus.UNCERTAIN
        return make_result(
            status, score=answer.noul, label="true" if answer.noul >= 0.5 else "false"
        )

    if check.kind == CheckKind.CHOICE:
        if answer.choice is None:
            return make_result(ResultStatus.ERROR, error="missing choice value")
        if not isinstance(check.criteria, dict) or answer.choice not in check.criteria:
            return make_result(ResultStatus.ERROR, error="choice is not present in check criteria")
        if answer.choice in check.uncertain_labels or (
            answer.confidence is not None and answer.confidence < check.min_confidence
        ):
            status = ResultStatus.UNCERTAIN
        elif answer.choice in check.failure_labels:
            status = ResultStatus.FAIL
        else:
            status = ResultStatus.PASS
        return make_result(status, label=answer.choice)

    if check.kind == CheckKind.SCORE:
        if answer.score is None:
            return make_result(ResultStatus.ERROR, error="missing score value")
        if (
            not math.isfinite(answer.score)
            or not isinstance(check.criteria, list)
            or not 0 <= answer.score <= len(check.criteria) - 1
        ):
            return make_result(
                ResultStatus.ERROR, error="score is outside the check criteria range"
            )
        if answer.confidence is not None and answer.confidence < check.min_confidence:
            status = ResultStatus.UNCERTAIN
        elif answer.score >= float(check.failure_score_gte or 0):
            status = ResultStatus.FAIL
        else:
            status = ResultStatus.PASS
        return make_result(status, score=answer.score)
    raise ValueError(f"unsupported semantic kind: {check.kind}")


class Evaluator:
    def __init__(
        self,
        *,
        config: LoopEvalConfig,
        checks: list[CheckSpec],
        decision_provider: DecisionProvider,
        fallback_provider: GenerativeProvider | None,
        store: LocalStore,
    ) -> None:
        self.config = config
        self.checks = [
            check for check in checks if check.enabled and check.lifecycle == CheckLifecycle.ACTIVE
        ]
        active_ids = [check.id for check in self.checks]
        if len(active_ids) != len(set(active_ids)):
            raise ValueError("only one active version of a check id may be loaded")
        self.decision_provider = decision_provider
        self.fallback_provider = fallback_provider
        self.store = store
        self._fallback_count = 0
        self._spent = 0.0
        self._budget_lock = asyncio.Lock()

    async def _decide(
        self, state: dict[str, Any], questions: list[TypedQuestion], timeout: float
    ) -> DecisionResponse:
        payload = {
            "state": state,
            "questions": [question.model_dump(mode="json") for question in questions],
        }
        identity = f"{self.decision_provider.identity}:{self.config.fingerprint()}"
        key = _cache_key("decision", identity, payload)
        if self.config.storage.cache:
            cached = self.store.cache_get(key)
            if cached:
                response = DecisionResponse.model_validate(cached)
                return response.model_copy(
                    update={"usage": ProviderUsage(cost_usd=0.0), "latency_ms": 0}
                )
        response = await asyncio.wait_for(
            self.decision_provider.decide(state, questions), timeout=max(timeout, 0.001)
        )
        if self.config.storage.cache:
            self.store.cache_put(key, "decision", response.model_dump(mode="json"))
        return response

    async def _fallback(
        self,
        *,
        state: dict[str, Any],
        checks: list[CheckSpec],
        results: list[CheckResult],
        reasons: list[str],
        timeout: float,
    ) -> FallbackResult | None:
        if self.fallback_provider is None:
            return None
        result_payload = [
            result.model_dump(mode="json", exclude={"usage", "latency_ms"}) for result in results
        ]
        schema = fallback_schema()
        stable_payload = {
            "system": FALLBACK_SYSTEM,
            "state": state,
            "checks": [check.model_dump(mode="json") for check in checks],
            "results": result_payload,
            "reasons": reasons,
            "schema": schema,
        }
        identity = f"{self.fallback_provider.identity}:{self.config.fingerprint()}"
        key = _cache_key("fallback", identity, stable_payload)
        if self.config.storage.cache:
            cached = self.store.cache_get(key)
            if cached:
                verdict = FallbackVerdict.model_validate(cached)
                return FallbackResult(
                    verdict=verdict,
                    provider=self.fallback_provider.name,
                    model=self.fallback_provider.model,
                    usage=ProviderUsage(cost_usd=0.0),
                )

        try:
            prompt = fallback_prompt(
                state=state,
                checks=checks,
                results=result_payload,
                reasons=reasons,
                max_chars=self.config.budgets.max_state_chars,
            )
        except ValueError as exc:
            raise _FallbackEvidenceTooLarge from exc

        async with self._budget_lock:
            limit = self.config.escalation.max_fallbacks_per_run
            if limit is not None and self._fallback_count >= limit:
                return None
            cost_limit = self.config.budgets.run_cost_usd
            if cost_limit is not None and self._spent >= cost_limit:
                return None
            self._fallback_count += 1

        raw, usage, latency = await asyncio.wait_for(
            self.fallback_provider.generate_structured(
                system=FALLBACK_SYSTEM,
                user=prompt,
                schema=schema,
                schema_name="loopeval_fallback_verdict",
            ),
            timeout=max(timeout, 0.001),
        )
        if usage.cost_usd is not None:
            async with self._budget_lock:
                self._spent += usage.cost_usd
        try:
            verdict = FallbackVerdict.model_validate(raw)
            known_ids = {check.id for check in checks}
            if (
                verdict.category == "existing_failure"
                and verdict.existing_check_id not in known_ids
            ):
                raise ValueError(
                    f"fallback referenced unknown existing check {verdict.existing_check_id!r}"
                )
            if verdict.category == "novel_failure" and verdict.candidate_check:
                spec = verdict.candidate_check.to_spec()
                if spec.kind == CheckKind.DETERMINISTIC and spec.rule not in RULES:
                    raise ValueError("fallback proposed an unregistered deterministic rule")
                if verdict.candidate_check.id in known_ids:
                    raise ValueError("fallback proposed a novel check with an existing active id")
        except (TypeError, ValueError) as exc:
            raise _InvalidFallbackResponse(exc, usage) from exc
        if self.config.storage.cache:
            self.store.cache_put(key, "fallback", verdict.model_dump(mode="json"))
        return FallbackResult(
            verdict=verdict,
            provider=self.fallback_provider.name,
            model=self.fallback_provider.model,
            usage=usage,
            latency_ms=latency,
        )

    @staticmethod
    def _audit(sample_id: str, rate: float) -> bool:
        if rate <= 0:
            return False
        if rate >= 1:
            return True
        seed = int(hashlib.sha256(sample_id.encode()).hexdigest()[:16], 16)
        return random.Random(seed).random() < rate

    async def evaluate_sample(
        self,
        sample: EvalSample,
        *,
        checks: list[CheckSpec] | None = None,
        enable_fallback: bool = True,
        enable_coverage: bool | None = None,
    ) -> SampleResult:
        started = time.perf_counter()
        deadline = started + self.config.budgets.per_sample_seconds
        selected = checks if checks is not None else self.checks
        deterministic = [check for check in selected if check.kind == CheckKind.DETERMINISTIC]
        semantic = [check for check in selected if check.is_semantic and applies(check, sample)]
        check_results = [run_deterministic(check, sample) for check in deterministic]
        reasons: list[str] = []
        decision_usage = ProviderUsage(cost_usd=0.0)

        if self.config.escalation.short_circuit_on_deterministic_failure and any(
            result.status == ResultStatus.FAIL for result in check_results
        ):
            sample_result = SampleResult(
                sample_id=sample.sample_id,
                sample_hash=sample.content_hash,
                verdict=OverallVerdict.FAIL,
                checks=check_results,
                expected_verdict=sample.expected_verdict,
                expected_labels=sample.labels,
                usage=ProviderUsage(cost_usd=0.0),
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
            return sample_result

        for check in selected:
            if check.is_semantic and not applies(check, sample):
                check_results.append(
                    CheckResult(
                        check_id=check.id,
                        check_version=check.version,
                        status=ResultStatus.SKIPPED,
                        severity=check.severity,
                        evidence={
                            "missing_required_fields": missing_required_fields(check, sample)
                        },
                    )
                )

        state = sample.state()
        state_size_exceeded = (
            len(json.dumps(state, ensure_ascii=False)) > self.config.budgets.max_state_chars
        )

        coverage_enabled = (
            self.config.escalation.coverage_check if enable_coverage is None else enable_coverage
        )
        questions = [check.question() for check in semantic]
        if coverage_enabled:
            applied_ids = {
                result.check_id for result in check_results if result.status != ResultStatus.SKIPPED
            } | {check.id for check in semantic}
            applicable = [check for check in selected if check.id in applied_ids]
            questions.append(_coverage_question(applicable))

        decision_response: DecisionResponse | None = None
        decision_failed = False
        if questions and state_size_exceeded:
            reasons.append("state_size_exceeded")
            for check in semantic:
                check_results.append(
                    CheckResult(
                        check_id=check.id,
                        check_version=check.version,
                        status=ResultStatus.ERROR,
                        severity=check.severity,
                        error="sample state exceeds budgets.max_state_chars",
                    )
                )
        elif questions:
            try:
                decision_response = await self._decide(
                    state, questions, deadline - time.perf_counter()
                )
                decision_usage = decision_response.usage
                if decision_usage.cost_usd is not None:
                    async with self._budget_lock:
                        self._spent += decision_usage.cost_usd
            except Exception as exc:
                decision_failed = True
                decision_usage = ProviderUsage()
                if self.config.escalation.on_decision_error:
                    reasons.append("decision_provider_error")
                for check in semantic:
                    check_results.append(
                        CheckResult(
                            check_id=check.id,
                            check_version=check.version,
                            status=ResultStatus.ERROR,
                            severity=check.severity,
                            error=str(exc),
                        )
                    )
        if decision_response:
            for check in semantic:
                answer = decision_response.answers.get(check.id)
                if answer is None:
                    check_results.append(
                        CheckResult(
                            check_id=check.id,
                            check_version=check.version,
                            status=ResultStatus.ERROR,
                            severity=check.severity,
                            error="decision response omitted this check",
                        )
                    )
                    if self.config.escalation.on_decision_error:
                        reasons.append("decision_provider_error")
                else:
                    check_result = interpret_answer(check, answer, decision_response)
                    # Provider usage belongs to the batched call, not every
                    # individual check. Keeping it on results would multiply cost.
                    check_result.usage = ProviderUsage()
                    check_results.append(check_result)
                    if (
                        check_result.status == ResultStatus.ERROR
                        and self.config.escalation.on_decision_error
                    ):
                        reasons.append("decision_provider_error")
            if coverage_enabled:
                coverage = decision_response.answers.get(COVERAGE_ID)
                if coverage is None or coverage.kind != QuestionKind.NOUL or coverage.noul is None:
                    reasons.append("coverage_error")
                elif coverage.noul >= self.config.escalation.coverage_failure_threshold:
                    reasons.append("uncovered_issue")
                elif coverage.noul > self.config.escalation.coverage_pass_threshold:
                    reasons.append("coverage_uncertain")

        applied_results = [r for r in check_results if r.status != ResultStatus.SKIPPED]
        if not applied_results and self.config.escalation.on_no_applicable_checks:
            reasons.append("no_applicable_checks")
        if self.config.escalation.on_uncertain and any(
            result.status in {ResultStatus.UNCERTAIN, ResultStatus.ERROR}
            for result in check_results
        ):
            reasons.append("uncertain_check")
        if self._audit(sample.sample_id, self.config.escalation.random_audit_rate):
            reasons.append("random_audit")
        reasons = list(dict.fromkeys(reasons))

        fallback: FallbackResult | None = None
        fallback_usage = ProviderUsage(cost_usd=0.0)
        if reasons and enable_fallback and state_size_exceeded:
            if "state_size_exceeded" not in reasons:
                reasons.append("state_size_exceeded")
        elif reasons and enable_fallback:
            try:
                fallback = await self._fallback(
                    state=state,
                    checks=selected,
                    results=check_results,
                    reasons=reasons,
                    timeout=deadline - time.perf_counter(),
                )
                if fallback is None:
                    reasons.append(
                        "fallback_disabled"
                        if self.fallback_provider is None
                        else "fallback_budget_exhausted"
                    )
            except _FallbackEvidenceTooLarge:
                reasons.append("fallback_state_size_exceeded")
            except _InvalidFallbackResponse as exc:
                fallback_usage = exc.usage
                reasons.append(f"fallback_error:{exc.error_type}")
            except Exception as exc:
                fallback_usage = ProviderUsage()
                reasons.append(f"fallback_error:{type(exc).__name__}")

        known_failure = any(result.status == ResultStatus.FAIL for result in check_results)
        if known_failure or (
            fallback and fallback.verdict.category in {"existing_failure", "novel_failure"}
        ):
            verdict = OverallVerdict.FAIL
        elif fallback and fallback.verdict.category == "acceptable":
            verdict = OverallVerdict.PASS
        elif (
            reasons
            or decision_failed
            or not applied_results
            or any(
                result.status in {ResultStatus.UNCERTAIN, ResultStatus.ERROR}
                for result in check_results
            )
        ):
            verdict = (
                OverallVerdict.FAIL
                if self.config.escalation.fail_closed_without_fallback
                else OverallVerdict.UNRESOLVED
            )
        else:
            verdict = OverallVerdict.PASS

        fallback_usage = fallback.usage if fallback else fallback_usage
        usage = _combine_usage(decision_usage, fallback_usage)
        sample_result = SampleResult(
            sample_id=sample.sample_id,
            sample_hash=sample.content_hash,
            verdict=verdict,
            checks=check_results,
            expected_verdict=sample.expected_verdict,
            expected_labels=sample.labels,
            escalated=bool(reasons),
            escalation_reasons=reasons,
            fallback=fallback,
            usage=usage,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        ingest_candidate(self.store, sample_result)
        return sample_result

    async def evaluate(self, samples: list[EvalSample]) -> EvaluationReport:
        if not samples:
            raise ValueError("at least one sample is required")
        sample_ids = [sample.sample_id for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample ids must be unique within a run")
        self._fallback_count = 0
        self._spent = 0.0
        run_id, started_at = self.store.start_run(self.config.fingerprint())
        semaphore = asyncio.Semaphore(self.config.budgets.concurrency)

        async def one(sample: EvalSample) -> SampleResult:
            async with semaphore:
                result = await self.evaluate_sample(sample)
                self.store.save_result(run_id, result)
                return result

        try:
            results = await asyncio.gather(*(one(sample) for sample in samples))
            report = EvaluationReport(
                run_id=run_id,
                results=results,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                config_hash=self.config.fingerprint(),
                dataset_hash=dataset_fingerprint(samples),
                checks_hash=checks_fingerprint(self.checks),
            )
            self.store.finish_run(report)
            return report
        except Exception as exc:
            self.store.fail_run(run_id, str(exc))
            raise
