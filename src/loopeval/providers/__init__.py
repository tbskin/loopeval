from __future__ import annotations

from ..config import ProviderConfig
from .base import DecisionProvider, GenerativeProvider, ProviderError
from .decision import HTTPDecisionProvider
from .generative import OpenAICompatibleProvider
from .mock import MockDecisionProvider, MockGenerativeProvider


def build_decision_provider(config: ProviderConfig) -> DecisionProvider:
    if config.type == "mock":
        return MockDecisionProvider(config)
    if config.type in {"typesafe", "openrouter_decisions"}:
        return HTTPDecisionProvider(config)
    raise ValueError(f"{config.type!r} is not a decision provider")


def build_generative_provider(config: ProviderConfig) -> GenerativeProvider | None:
    if config.type == "disabled":
        return None
    if config.type == "mock":
        return MockGenerativeProvider(config)
    if config.type in {"openrouter", "openai", "openai_compatible"}:
        return OpenAICompatibleProvider(config)
    raise ValueError(f"{config.type!r} is not a generative provider")


__all__ = [
    "DecisionProvider",
    "GenerativeProvider",
    "ProviderError",
    "build_decision_provider",
    "build_generative_provider",
]
