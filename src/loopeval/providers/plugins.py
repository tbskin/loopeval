from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, cast

from ..config import ProviderConfig
from .base import DecisionProvider, GenerativeProvider, ProviderError


def _load_provider_plugin(
    config: ProviderConfig,
    *,
    group: str,
    expected_type: type[Any],
) -> Any:
    name = config.plugin
    if not name:
        raise ProviderError("plugin provider is missing its plugin name")
    matches = list(entry_points(group=group, name=name))
    if not matches:
        raise ProviderError(
            f"provider plugin {name!r} is not installed in entry-point group {group!r}"
        )
    if len(matches) > 1:
        raise ProviderError(
            f"provider plugin {name!r} is registered more than once in {group!r}"
        )
    factory = matches[0].load()
    if not callable(factory):
        raise ProviderError(f"provider plugin {name!r} does not expose a callable factory")
    provider = factory(config)
    if not isinstance(provider, expected_type):
        raise ProviderError(
            f"provider plugin {name!r} returned {type(provider).__name__}, "
            f"expected {expected_type.__name__}"
        )
    return provider


def load_decision_provider_plugin(config: ProviderConfig) -> DecisionProvider:
    return cast(
        DecisionProvider,
        _load_provider_plugin(
            config,
            group="loopeval.decision_providers",
            expected_type=DecisionProvider,
        ),
    )


def load_generative_provider_plugin(config: ProviderConfig) -> GenerativeProvider:
    return cast(
        GenerativeProvider,
        _load_provider_plugin(
            config,
            group="loopeval.generative_providers",
            expected_type=GenerativeProvider,
        ),
    )
