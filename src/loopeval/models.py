from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QuestionKind(StrEnum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class CheckKind(StrEnum):
    DETERMINISTIC = "deterministic"
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class CheckLifecycle(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    SHADOW = "shadow"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class ResultStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"
    ERROR = "error"


class OverallVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNRESOLVED = "unresolved"


class EvalSample(BaseModel):
    """Canonical input accepted by every LoopEval check.

    The common fields cover response, RAG, and agent evaluations. Extra
    application fields belong in ``data`` so the stable schema does not need to
    change whenever a new domain is evaluated.
    """

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    input: Any
    output: Any = None
    context: Any = None
    expected: Any = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    labels: list[str] | None = None
    expected_verdict: OverallVerdict | None = None

    @property
    def sample_id(self) -> str:
        if self.id:
            return self.id
        payload = json.dumps(
            self.model_dump(exclude={"id"}, mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def state(self) -> dict[str, Any]:
        state = self.model_dump(
            exclude_none=True,
            exclude={"id", "labels", "expected_verdict"},
            mode="json",
        )
        state["sample_id"] = self.sample_id
        return state


class TypedQuestion(FrozenModel):
    id: str
    kind: QuestionKind
    instructions: str
    criteria: dict[str, Any] | list[Any] | None = None

    def provider_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.kind.value,
            "instructions": self.instructions,
        }
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return payload


class CheckSpec(BaseModel):
    """A versioned deterministic rule or typed semantic question.

    Semantic checks are failure-oriented: a Noul asks whether a defect is
    present, Choice declares which labels represent failure, and Score declares
    the score at or above which the sample fails. This makes aggregation
    predictable across check kinds.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str = "1.0.0"
    name: str
    description: str
    kind: CheckKind
    lifecycle: CheckLifecycle = CheckLifecycle.ACTIVE
    source: Literal["builtin", "user", "learned"] = "user"
    severity: Literal["info", "warn", "error", "critical"] = "error"
    enabled: bool = True
    requires: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    # Deterministic checks.
    rule: str | None = None
    field: str = "output"
    params: dict[str, Any] = Field(default_factory=dict)

    # Semantic checks.
    instructions: str | None = None
    criteria: dict[str, Any] | list[Any] | None = None
    pass_threshold: float = Field(default=0.2, ge=0, le=1)
    failure_threshold: float = Field(default=0.8, ge=0, le=1)
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    failure_labels: list[str] = Field(default_factory=list)
    uncertain_labels: list[str] = Field(
        default_factory=lambda: ["other", "uncertain", "insufficient_evidence"]
    )
    failure_score_gte: float | None = None
    examples: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_shape(self) -> CheckSpec:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", self.id):
            raise ValueError(
                "check id must contain only lowercase letters, digits, '.', '_' or '-'"
            )
        if self.pass_threshold >= self.failure_threshold:
            raise ValueError("pass_threshold must be lower than failure_threshold")
        if self.kind == CheckKind.DETERMINISTIC and not self.rule:
            raise ValueError("deterministic checks require rule")
        if self.kind != CheckKind.DETERMINISTIC and not self.instructions:
            raise ValueError("semantic checks require instructions")
        if self.kind == CheckKind.CHOICE:
            if not isinstance(self.criteria, dict) or not self.criteria:
                raise ValueError("choice checks require criteria as a non-empty object")
            if not self.failure_labels:
                raise ValueError("choice checks require at least one failure_label")
            unknown = set(self.failure_labels) - set(self.criteria)
            if unknown:
                raise ValueError(f"failure_labels are not present in criteria: {sorted(unknown)}")
        if self.kind == CheckKind.SCORE:
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score checks require between 2 and 10 ordered criteria")
            if self.failure_score_gte is None:
                raise ValueError("score checks require failure_score_gte")
            if not 0 <= self.failure_score_gte <= len(self.criteria) - 1:
                raise ValueError("failure_score_gte must be within the score criteria range")
        return self

    @property
    def is_semantic(self) -> bool:
        return self.kind != CheckKind.DETERMINISTIC

    def question(self) -> TypedQuestion:
        if not self.is_semantic or self.instructions is None:
            raise ValueError(f"{self.id} is not a semantic check")
        return TypedQuestion(
            id=self.id,
            kind=QuestionKind(self.kind.value),
            instructions=self.instructions,
            criteria=self.criteria,
        )


class ProviderUsage(FrozenModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class DecisionAnswer(FrozenModel):
    kind: QuestionKind
    noul: float | None = Field(default=None, ge=0, le=1)
    choice: str | None = None
    score: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_shape(self) -> DecisionAnswer:
        if any(probability < 0 or probability > 1 for probability in self.probabilities.values()):
            raise ValueError("probabilities must be between 0 and 1")
        if self.kind == QuestionKind.NOUL and self.noul is None:
            raise ValueError("noul answers require noul")
        if self.kind == QuestionKind.CHOICE:
            if self.choice is None or self.confidence is None or not self.probabilities:
                raise ValueError("choice answers require choice, confidence, and probabilities")
            if self.choice not in self.probabilities:
                raise ValueError("choice must be present in probabilities")
        if self.kind == QuestionKind.SCORE and (
            self.score is None or self.confidence is None or not self.probabilities
        ):
            raise ValueError("score answers require score, confidence, and probabilities")
        return self


class DecisionResponse(FrozenModel):
    answers: dict[str, DecisionAnswer]
    provider: str
    model: str
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    latency_ms: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    check_version: str
    status: ResultStatus
    severity: str
    score: float | None = None
    label: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    latency_ms: int = 0
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    error: str | None = None


class CandidateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    kind: CheckKind
    instructions: str | None = None
    criteria: dict[str, Any] | list[Any] | None = None
    rule: str | None = None
    field: str = "output"
    params: dict[str, Any] = Field(default_factory=dict)
    requires: list[str] = Field(default_factory=list)
    failure_labels: list[str] = Field(default_factory=list)
    uncertain_labels: list[str] = Field(default_factory=list)
    pass_threshold: float = 0.2
    failure_threshold: float = 0.8
    min_confidence: float = 0.6
    failure_score_gte: float | None = None
    severity: Literal["info", "warn", "error", "critical"] = "error"
    examples: list[dict[str, Any]] = Field(default_factory=list)

    def to_spec(self, *, lifecycle: CheckLifecycle = CheckLifecycle.PROPOSED) -> CheckSpec:
        return CheckSpec(
            **self.model_dump(),
            version="1.0.0",
            lifecycle=lifecycle,
            source="learned",
        )


class BootstrapCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: CandidateCheck
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class BootstrapProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    candidates: list[BootstrapCandidate] = Field(min_length=1, max_length=20)


class FallbackVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["existing_failure", "novel_failure", "acceptable", "insufficient_evidence"]
    existing_check_id: str | None = None
    label: str | None = None
    title: str | None = None
    description: str | None = None
    severity: Literal["info", "warn", "error", "critical"] = "error"
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    candidate_check: CandidateCheck | None = None

    @model_validator(mode="after")
    def validate_category(self) -> FallbackVerdict:
        if self.category == "existing_failure" and not self.existing_check_id:
            raise ValueError("existing_failure requires existing_check_id")
        if self.category == "novel_failure" and self.candidate_check is None:
            raise ValueError("novel_failure requires candidate_check")
        if self.category != "novel_failure" and self.candidate_check is not None:
            raise ValueError("only novel_failure may include candidate_check")
        if self.category != "existing_failure" and self.existing_check_id is not None:
            raise ValueError("only existing_failure may include existing_check_id")
        return self


class FallbackResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: FallbackVerdict
    provider: str
    model: str
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    latency_ms: int = 0


class SampleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    verdict: OverallVerdict
    checks: list[CheckResult]
    expected_verdict: OverallVerdict | None = None
    expected_labels: list[str] | None = None
    escalated: bool = False
    escalation_reasons: list[str] = Field(default_factory=list)
    fallback: FallbackResult | None = None
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    latency_ms: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    results: list[SampleResult]
    started_at: datetime
    completed_at: datetime
    config_hash: str

    @property
    def sample_count(self) -> int:
        return len(self.results)

    @property
    def escalation_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.escalated for r in self.results) / len(self.results)

    @property
    def total_cost_usd(self) -> float | None:
        costs = [r.usage.cost_usd for r in self.results if r.usage.cost_usd is not None]
        return sum(costs) if costs else None
