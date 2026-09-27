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
from .base import DecisionProvider, MissingCredentialError, ProviderError
from .http import post_json
from .usage import parse_usage


class HTTPDecisionProvider(DecisionProvider):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.model = config.model
        self.name = config.type
        self.api_key = config.api_key
        if not self.api_key:
            raise MissingCredentialError(
                f"{config.type} requires environment variable {config.api_key_env!r}"
            )
        if config.type == "typesafe":
            self.url = config.base_url or "https://api.typesafe.ai/v1/systemone"
            self.title_header = None
        elif config.type == "openrouter_decisions":
            self.url = config.base_url or "https://openrouter.ai/api/alpha/decisions"
            self.title_header = "LoopEval"
        else:
            raise ValueError(f"unsupported decision provider type: {config.type}")

    def _usage(self, raw: dict[str, Any]) -> ProviderUsage:
        return parse_usage(raw, self.config)

    @staticmethod
    def _answer(kind: QuestionKind, raw: dict[str, Any]) -> DecisionAnswer:
        probabilities = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
        return DecisionAnswer(
            kind=kind,
            noul=float(raw["noul"]) if raw.get("noul") is not None else None,
            choice=str(raw["choice"]) if raw.get("choice") is not None else None,
            score=float(raw["score"]) if raw.get("score") is not None else None,
            probabilities=probabilities,
            confidence=float(raw["confidence"]) if raw.get("confidence") is not None else None,
        )

    async def decide(
        self, state: dict[str, Any], questions: list[TypedQuestion]
    ) -> DecisionResponse:
        if not questions:
            raise ProviderError("decision request requires at least one question")
        if len({question.id for question in questions}) != len(questions):
            raise ProviderError("decision request contains duplicate question ids")
        payload = {
            "model": self.model,
            "state": state,
            "questions": {question.id: question.provider_payload() for question in questions},
        }
        headers = {
            "Content-Type": "application/json",
            **self.config.headers,
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.title_header:
            headers["X-Title"] = self.title_header

        started = time.perf_counter()
        raw = await post_json(self.config, self.url, headers, payload)
        raw_answers = raw.get("answers", {})
        if not isinstance(raw_answers, dict):
            raise ProviderError("decision response contains invalid answers")
        answers: dict[str, DecisionAnswer] = {}
        for question in questions:
            item = raw_answers.get(question.id)
            if not isinstance(item, dict):
                raise ProviderError(f"response omitted answer for {question.id}")
            if item.get("type", question.kind.value) != question.kind.value:
                raise ProviderError(f"response returned the wrong answer type for {question.id}")
            try:
                answers[question.id] = self._answer(question.kind, item)
            except (ValueError, TypeError, AttributeError):
                raise ProviderError(f"response returned an invalid answer for {question.id}") from None
        return DecisionResponse(
            answers=answers,
            provider=self.name,
            model=str(raw.get("model") or self.model),
            usage=self._usage(raw),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
