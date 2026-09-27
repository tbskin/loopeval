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


class AnthropicProvider(GenerativeProvider):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.model = config.model
        self.name = config.type
        api_key = config.api_key
        if not api_key:
            raise MissingCredentialError(
                f"anthropic requires environment variable {config.api_key_env!r}"
            )
        self.api_key: str = api_key
        base = config.base_url or "https://api.anthropic.com/v1"
        self.url = f"{base.rstrip('/')}/messages"

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

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], ProviderUsage, int]:
        del schema_name  # Anthropic's JSON output format does not take a schema name.
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.config.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.config.structured_output:
            body["output_config"] = {
                "format": {"type": "json_schema", "schema": schema}
            }
        else:
            body["system"] += "\nReturn only JSON matching this schema:\n" + json.dumps(schema)

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **self.config.headers,
            "x-api-key": self.api_key,
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
                            f"anthropic returned {response.status_code}: {response.text[:500]}"
                        )
                    last_error = ProviderError(
                        f"anthropic returned retryable status {response.status_code}"
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                if attempt < self.config.max_retries:
                    await asyncio.sleep(retry_delay(response, attempt))
        if raw is None:
            raise ProviderError(f"anthropic generative request failed: {last_error}")
        if raw.get("stop_reason") in {"refusal", "max_tokens"}:
            raise ProviderError(f"anthropic response stopped with {raw['stop_reason']}")
        content = raw.get("content") or []
        text = next(
            (
                block.get("text")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ),
            None,
        )
        if not isinstance(text, str):
            raise ProviderError("anthropic response did not contain text content")
        latency_ms = round((time.perf_counter() - started) * 1000)
        return parse_json_object(text), self._usage(raw), latency_ms
