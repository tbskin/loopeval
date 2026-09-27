from __future__ import annotations

import time
from typing import Any

from ..config import ProviderConfig
from ..models import (
    DecisionAnswer,
    DecisionResponse,
    ProviderUsage,
    QuestionKind,
    TypedQuestion,
)
from .base import DecisionProvider, GenerativeProvider


class MockDecisionProvider(DecisionProvider):
    def __init__(self, config: ProviderConfig) -> None:
        self.name = "mock"
        self.model = config.model or "mock-decision"
        self.responses = config.mock_responses

    async def decide(
        self, state: dict[str, Any], questions: list[TypedQuestion]
    ) -> DecisionResponse:
        started = time.perf_counter()
        marker = str(state.get("output", ""))
        learned_novel_check = any(question.id == "learned.novel_pattern" for question in questions)
        answers: dict[str, DecisionAnswer] = {}
        for question in questions:
            configured = self.responses.get(question.id)
            if isinstance(configured, dict):
                answers[question.id] = DecisionAnswer.model_validate(
                    {"kind": question.kind, **configured}
                )
            elif question.kind == QuestionKind.NOUL:
                value = 0.95 if "[FAIL]" in marker else 0.7 if "[UNCERTAIN]" in marker else 0.05
                if (question.id == "learned.novel_pattern" and "[NOVEL]" in marker) or (
                    question.id == "__loopeval_coverage__"
                    and "[NOVEL]" in marker
                    and not learned_novel_check
                ):
                    value = 0.95
                answers[question.id] = DecisionAnswer(kind=question.kind, noul=value)
            elif question.kind == QuestionKind.CHOICE:
                choices = question.criteria if isinstance(question.criteria, dict) else {"pass": ""}
                label = next(iter(choices.keys()))
                answers[question.id] = DecisionAnswer(
                    kind=question.kind,
                    choice=str(label),
                    confidence=0.95,
                    probabilities={str(label): 0.95},
                )
            else:
                answers[question.id] = DecisionAnswer(
                    kind=question.kind,
                    score=0.0,
                    confidence=0.95,
                    probabilities={"0": 0.95},
                )
        return DecisionResponse(
            answers=answers,
            provider=self.name,
            model=self.model,
            usage=ProviderUsage(cost_usd=0.0),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )


class MockGenerativeProvider(GenerativeProvider):
    def __init__(self, config: ProviderConfig) -> None:
        self.name = "mock"
        self.model = config.model or "mock-generative"
        self.responses = config.mock_responses

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], ProviderUsage, int]:
        if schema_name == "loopeval_doctor":
            return {"ok": True}, ProviderUsage(cost_usd=0.0), 0
        if schema_name == "loopeval_bootstrap":
            return (
                {
                    "summary": "A focused starter check for the supplied application requirements.",
                    "candidates": [
                        {
                            "check": {
                                "id": "quality.requirement_violation",
                                "name": "Requirement violation",
                                "description": "The output violates a supplied product requirement.",
                                "kind": "noul",
                                "instructions": "Does `output` violate a material supplied requirement?",
                                "criteria": {
                                    "true": "A material supplied requirement is violated.",
                                    "false": "The output follows the supplied requirements.",
                                },
                                "requires": ["output"],
                                "severity": "error",
                            },
                            "rationale": "The scenarios require a semantic product-policy judgment.",
                            "confidence": 0.8,
                        }
                    ],
                },
                ProviderUsage(cost_usd=0.0),
                0,
            )
        if self.responses.get("verdict"):
            return dict(self.responses["verdict"]), ProviderUsage(cost_usd=0.0), 0
        if "[NOVEL]" in user:
            return (
                {
                    "category": "novel_failure",
                    "existing_check_id": None,
                    "label": "novel-pattern",
                    "title": "Novel pattern",
                    "description": "The sample contains the offline demo's novel marker.",
                    "severity": "error",
                    "evidence": "The output contains [NOVEL].",
                    "confidence": 0.95,
                    "candidate_check": {
                        "id": "learned.novel_pattern",
                        "name": "Novel pattern",
                        "description": "Detect the demonstrated novel failure pattern.",
                        "kind": "noul",
                        "instructions": "Does the output exhibit the reviewed novel failure pattern?",
                        "criteria": {
                            "true": "The reviewed failure pattern is present.",
                            "false": "The reviewed failure pattern is absent.",
                        },
                        "severity": "error",
                    },
                },
                ProviderUsage(cost_usd=0.0),
                0,
            )
        return (
            {
                "category": "acceptable",
                "existing_check_id": None,
                "label": "acceptable",
                "title": "Acceptable",
                "description": "No material issue found.",
                "severity": "info",
                "evidence": "No known or novel failure was found.",
                "confidence": 0.9,
                "candidate_check": None,
            },
            ProviderUsage(cost_usd=0.0),
            0,
        )
