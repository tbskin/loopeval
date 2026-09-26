from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import DecisionResponse, ProviderUsage, TypedQuestion


class ProviderError(RuntimeError):
    pass


class MissingCredentialError(ProviderError):
    pass


class DecisionProvider(ABC):
    name: str
    model: str

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.model}"

    @abstractmethod
    async def decide(
        self, state: dict[str, Any], questions: list[TypedQuestion]
    ) -> DecisionResponse:
        raise NotImplementedError


class GenerativeProvider(ABC):
    name: str
    model: str

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.model}"

    @abstractmethod
    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], ProviderUsage, int]:
        raise NotImplementedError
