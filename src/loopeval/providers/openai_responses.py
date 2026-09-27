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
        return parse_usage(raw, self.config)

    @staticmethod
    def _output_text(raw: dict[str, Any]) -> str:
        texts: list[str] = []
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
                    texts.append(text)
        if texts:
            return "".join(texts)
        output_text = raw.get("output_text")
        if isinstance(output_text, str):
            return output_text
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
            "store": False,
        }
        if self.config.structured_output:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": provider_schema(schema),
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
        raw = await post_json(self.config, self.url, headers, body)
        if raw.get("status") not in {None, "completed"} or raw.get("error"):
            raise ProviderError("openai response was incomplete or failed")
        latency_ms = round((time.perf_counter() - started) * 1000)
        parsed = parse_json_object(self._output_text(raw))
        if self.config.structured_output:
            parsed = decode_response(parsed, schema)
        return parsed, self._usage(raw), latency_ms
