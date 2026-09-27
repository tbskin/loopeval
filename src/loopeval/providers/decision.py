from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from ..config import ProviderConfig
from ..models import (
    DecisionAnswer,
    DecisionResponse,
    ProviderUsage,
    QuestionKind,
    TypedQuestion,
)
from .base import DecisionProvider, MissingCredentialError, ProviderError
from .http import RETRYABLE_STATUS_CODES, retry_delay


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
        usage = raw.get("usage") or {}
        input_tokens = int(
            usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("tokens") or 0
        )
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        cost = usage.get("cost") or usage.get("cost_usd")
        if cost is None and self.config.input_cost_per_million is not None:
            cost = input_tokens * self.config.input_cost_per_million / 1_000_000
            if self.config.output_cost_per_million is not None:
                cost += output_tokens * self.config.output_cost_per_million / 1_000_000
        return ProviderUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

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
        payload = {
            "model": self.model,
            "state": state,
            "questions": {question.id: question.provider_payload() for question in questions},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.config.headers,
        }
        if self.title_header:
            headers["X-Title"] = self.title_header

        started = time.perf_counter()
        last_error: Exception | None = None
        raw: dict[str, Any] | None = None
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            for attempt in range(self.config.max_retries + 1):
                response: httpx.Response | None = None
                try:
                    response = await client.post(self.url, headers=headers, json=payload)
                    if response.status_code == 200:
                        raw = response.json()
                        break
                    if response.status_code not in RETRYABLE_STATUS_CODES:
                        raise ProviderError(
                            f"{self.name} returned {response.status_code}: {response.text[:500]}"
                        )
                    last_error = ProviderError(
                        f"{self.name} returned retryable status {response.status_code}"
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                if attempt < self.config.max_retries:
                    await asyncio.sleep(retry_delay(response, attempt))
        if raw is None:
            raise ProviderError(f"{self.name} decision request failed: {last_error}")

        raw_answers = raw.get("answers") or {}
        answers: dict[str, DecisionAnswer] = {}
        for question in questions:
            item = raw_answers.get(question.id)
            if not isinstance(item, dict):
                raise ProviderError(f"response omitted answer for {question.id}")
            answers[question.id] = self._answer(question.kind, item)
        return DecisionResponse(
            answers=answers,
            provider=self.name,
            model=str(raw.get("model") or self.model),
            usage=self._usage(raw),
            latency_ms=round((time.perf_counter() - started) * 1000),
            raw=raw,
        )
