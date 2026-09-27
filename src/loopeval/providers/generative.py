from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from ..config import ProviderConfig
from ..models import ProviderUsage
from .base import GenerativeProvider, MissingCredentialError, ProviderError
from .http import RETRYABLE_STATUS_CODES, retry_delay


def parse_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ProviderError("generative provider returned non-text content")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        parsed = None
        for offset, character in enumerate(content):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(content[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
        if parsed is None:
            raise ProviderError("generative provider returned no JSON object") from None
    if not isinstance(parsed, dict):
        raise ProviderError("structured response must be a JSON object")
    return parsed


class OpenAICompatibleProvider(GenerativeProvider):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.model = config.model
        self.name = config.type
        self.api_key = config.api_key
        if not self.api_key and config.type != "openai_compatible":
            raise MissingCredentialError(
                f"{config.type} requires environment variable {config.api_key_env!r}"
            )
        bases = {
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": "https://api.openai.com/v1",
        }
        base = config.base_url or bases.get(config.type)
        if not base:
            raise ValueError("openai_compatible provider requires base_url")
        self.url = f"{base.rstrip('/')}/chat/completions"

    def _usage(self, raw: dict[str, Any]) -> ProviderUsage:
        usage = raw.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cost = usage.get("cost") or usage.get("cost_usd")
        if cost is None and self.config.input_cost_per_million is not None:
            cost = input_tokens * self.config.input_cost_per_million / 1_000_000
            if self.config.output_cost_per_million is not None:
                cost += output_tokens * self.config.output_cost_per_million / 1_000_000
        return ProviderUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

    @staticmethod
    def _parse(content: Any) -> dict[str, Any]:
        return parse_json_object(content)

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
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        if self.config.structured_output:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        else:
            body["messages"][0]["content"] += (
                "\nReturn only JSON matching this schema:\n" + json.dumps(schema)
            )
        headers = {"Content-Type": "application/json", **self.config.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.config.type == "openrouter":
            headers["X-Title"] = "LoopEval"

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
                            f"{self.name} returned {response.status_code}: {response.text[:500]}"
                        )
                    last_error = ProviderError(f"retryable status {response.status_code}")
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                if attempt < self.config.max_retries:
                    await asyncio.sleep(retry_delay(response, attempt))
        if raw is None:
            raise ProviderError(f"{self.name} generative request failed: {last_error}")
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("generative response did not contain message content") from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        return self._parse(content), self._usage(raw), latency_ms
