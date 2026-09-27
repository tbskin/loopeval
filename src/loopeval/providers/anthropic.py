from __future__ import annotations

import json
import time
from typing import Any

from ..config import ProviderConfig
from ..models import ProviderUsage
from .base import GenerativeProvider, MissingCredentialError, ProviderError
from .generative import parse_json_object
from .http import post_json
from .schema import decode_response, provider_schema
from .usage import parse_usage


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
        usage = parse_usage(raw, self.config)
        provider_usage = raw.get("usage") or {}
        cached = provider_usage.get("cache_read_input_tokens", 0)
        created = provider_usage.get("cache_creation_input_tokens", 0)
        if cached or created:
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in (cached, created)):
                raise ProviderError("anthropic returned invalid cache token usage")
            # Anthropic excludes these tokens from input_tokens and prices them separately.
            return ProviderUsage(
                input_tokens=usage.input_tokens + cached + created,
                output_tokens=usage.output_tokens,
                cost_usd=None,
            )
        return usage

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
                "format": {"type": "json_schema", "schema": provider_schema(schema)}
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
        raw = await post_json(self.config, self.url, headers, body)
        if raw.get("stop_reason") in {"refusal", "max_tokens"}:
            raise ProviderError(f"anthropic response stopped with {raw['stop_reason']}")
        if raw.get("stop_reason") not in {None, "end_turn", "stop_sequence"}:
            raise ProviderError("anthropic response did not finish normally")
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
        parsed = parse_json_object(text)
        if self.config.structured_output:
            parsed = decode_response(parsed, schema)
        return parsed, self._usage(raw), latency_ms
