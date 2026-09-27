from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from ..config import ProviderConfig
from ..models import ProviderUsage
from .base import GenerativeProvider, MissingCredentialError, ProviderError
from .generative import parse_json_object
from .http import RETRYABLE_STATUS_CODES, retry_delay


class OpenAIResponsesProvider(GenerativeProvider):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.model = config.model
        self.name = config.type
        self.api_key = config.api_key
        if not self.api_key:
            raise MissingCredentialError(
                f"openai_responses requires environment variable {config.api_key_env!r}"
            )
        base = config.base_url or "https://api.openai.com/v1"
        self.url = f"{base.rstrip('/')}/responses"

    def _usage(self, raw: dict[str, Any]) -> ProviderUsage:
        usage = raw.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cost: float | None = None
        if self.config.input_cost_per_million is not None:
            cost = input_tokens * self.config.input_cost_per_million / 1_000_000
            if self.config.output_cost_per_million is not None:
                cost += output_tokens * self.config.output_cost_per_million / 1_000_000
        return ProviderUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

    @staticmethod
    def _output_text(raw: dict[str, Any]) -> str:
        output_text = raw.get("output_text")
        if isinstance(output_text, str):
            return output_text
        for item in raw.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for block in item.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "refusal":
                    raise ProviderError("openai response contained a refusal")
                text = block.get("text")
                if block.get("type") == "output_text" and isinstance(text, str):
                    return text
        raise ProviderError("openai response did not contain output text")

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], ProviderUsage, int]:
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.structured_output:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            }
        else:
            body["instructions"] += "\nReturn only JSON matching this schema:\n" + json.dumps(
                schema
            )
        headers = {
            "Content-Type": "application/json",
            **self.config.headers,
            "Authorization": f"Bearer {self.api_key}",
        }
        started = time.perf_counter()
        last_error: Exception | None = None
        raw: dict[str, Any] | None = None
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            for attempt in range(self.config.max_retries + 1):
                response: httpx.Response | None = None
                try:
                    response = await client.post(self.url, headers=headers, json=body)
                    if response.status_code == 200:
                        raw = response.json()
                        break
                    if response.status_code not in RETRYABLE_STATUS_CODES:
                        raise ProviderError(
                            f"openai responses returned {response.status_code}: "
                            f"{response.text[:500]}"
                        )
                    last_error = ProviderError(
                        f"openai responses returned retryable status {response.status_code}"
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                if attempt < self.config.max_retries:
                    await asyncio.sleep(retry_delay(response, attempt))
        if raw is None:
            raise ProviderError(f"openai responses request failed: {last_error}")
        if raw.get("status") == "incomplete":
            reason = (raw.get("incomplete_details") or {}).get("reason") or "unknown reason"
            raise ProviderError(f"openai response was incomplete: {reason}")
        latency_ms = round((time.perf_counter() - started) * 1000)
        return parse_json_object(self._output_text(raw)), self._usage(raw), latency_ms
