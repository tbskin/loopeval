from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from ..config import ProviderConfig
from .base import ProviderError

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """Return a bounded retry delay, honoring Retry-After when supplied."""
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        try:
            seconds = float(str(retry_after))
            if math.isfinite(seconds):
                return min(max(seconds, 0.0), 30.0)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                seconds = (retry_at - datetime.now(UTC)).total_seconds()
                return min(max(seconds, 0.0), 30.0)
            except (TypeError, ValueError, OverflowError):
                pass
    return min(0.25 * (2.0**attempt), 2.0)


async def post_json(
    config: ProviderConfig, url: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    """Make a bounded request without exposing provider bodies or transport secrets."""
    last_error = "request failed"
    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        for attempt in range(config.max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    try:
                        raw = response.json()
                    except ValueError:
                        raise ProviderError(f"{config.type} returned invalid JSON") from None
                    if not isinstance(raw, dict):
                        raise ProviderError(f"{config.type} returned a non-object response")
                    return raw
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    hint = {
                        400: "check the model, request schema, and provider settings",
                        401: "check the configured API key",
                        403: "check account permissions and model access",
                        404: "check the endpoint and model name",
                    }.get(response.status_code, "check the provider account and configuration")
                    raise ProviderError(f"{config.type} returned {response.status_code}; {hint}")
                last_error = f"retryable status {response.status_code}"
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Exception strings and response bodies can contain credentials or sample data.
                last_error = type(exc).__name__
            if attempt < config.max_retries:
                await asyncio.sleep(retry_delay(response, attempt))
    raise ProviderError(f"{config.type} request failed: {last_error}")
