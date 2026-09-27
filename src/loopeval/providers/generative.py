from __future__ import annotations

import json
import time
from typing import Any

from ..config import ProviderConfig
from ..models import ProviderUsage
from .base import GenerativeProvider, MissingCredentialError, ProviderError
from .http import post_json
from .schema import decode_response, provider_schema
from .usage import parse_usage


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
        if not self.api_key and (config.type != "openai_compatible" or config.api_key_env):
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
        return parse_usage(raw, self.config)

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
        }
        token_field = "max_completion_tokens" if self.config.type == "openai" else "max_tokens"
        body[token_field] = self.config.max_output_tokens
        if self.config.type == "openai":
            body["store"] = False
        if self.config.structured_output:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name, "strict": True, "schema": provider_schema(schema)
                },
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
        raw = await post_json(self.config, self.url, headers, body)
        try:
            choice = raw["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("generative response did not contain message content") from exc
        if choice.get("finish_reason") not in {None, "stop"}:
            raise ProviderError("generative response did not finish normally")
        if message.get("refusal"):
            raise ProviderError("generative response contained a refusal")
        latency_ms = round((time.perf_counter() - started) * 1000)
        parsed = self._parse(content)
        if self.config.structured_output:
            parsed = decode_response(parsed, schema)
        return parsed, self._usage(raw), latency_ms
