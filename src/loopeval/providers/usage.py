from __future__ import annotations

import math
from typing import Any

from ..config import ProviderConfig
from ..models import ProviderUsage
from .base import ProviderError


def parse_usage(raw: dict[str, Any], config: ProviderConfig) -> ProviderUsage:
    """Preserve reported zero costs and leave incomplete price estimates unknown."""
    usage = raw.get("usage")
    if usage is None:
        return ProviderUsage(cost_usd=None)
    if not isinstance(usage, dict):
        raise ProviderError("provider returned invalid usage")

    def count(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ProviderError("provider returned invalid token usage")
                return value
        return None

    input_tokens = count("input_tokens", "prompt_tokens", "tokens")
    output_tokens = count("output_tokens", "completion_tokens")
    cost = usage.get("cost")
    if cost is None:
        cost = usage.get("cost_usd")
    if cost is not None:
        if (
            isinstance(cost, bool)
            or not isinstance(cost, int | float)
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise ProviderError("provider returned invalid cost")
    else:
        prices = [config.input_cost_per_million, config.output_cost_per_million]
        counts = [input_tokens, output_tokens]
        if all(
            price == 0 or (tokens is not None and (tokens == 0 or price is not None))
            for tokens, price in zip(counts, prices, strict=True)
        ):
            cost = sum(
                (tokens or 0) * (price or 0) / 1_000_000
                for tokens, price in zip(counts, prices, strict=True)
            )
    return ProviderUsage(
        input_tokens=input_tokens or 0, output_tokens=output_tokens or 0, cost_usd=cost
    )
