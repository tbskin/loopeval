from __future__ import annotations

from ..config import ProviderConfig
from .anthropic import AnthropicProvider
from .base import DecisionProvider, GenerativeProvider, ProviderError
from .decision import HTTPDecisionProvider
from .generative import OpenAICompatibleProvider
from .mock import MockDecisionProvider, MockGenerativeProvider
from .openai_responses import OpenAIResponsesProvider
from .plugins import load_decision_provider_plugin, load_generative_provider_plugin


def build_decision_provider(config: ProviderConfig) -> DecisionProvider:
    if config.type == "mock":
        return MockDecisionProvider(config)
    if config.type in {"typesafe", "openrouter_decisions"}:
        return HTTPDecisionProvider(config)
    if config.type == "plugin":
        return load_decision_provider_plugin(config)
    raise ValueError(f"{config.type!r} is not a decision provider")


def build_generative_provider(config: ProviderConfig) -> GenerativeProvider | None:
    if config.type == "disabled":
        return None
    if config.type == "mock":
        return MockGenerativeProvider(config)
    if config.type in {"openrouter", "openai", "openai_compatible"}:
        return OpenAICompatibleProvider(config)
    if config.type == "openai_responses":
        return OpenAIResponsesProvider(config)
    if config.type == "anthropic":
        return AnthropicProvider(config)
    if config.type == "plugin":
        return load_generative_provider_plugin(config)
    raise ValueError(f"{config.type!r} is not a generative provider")


__all__ = [
    "DecisionProvider",
    "GenerativeProvider",
    "ProviderError",
    "build_decision_provider",
    "build_generative_provider",
]
