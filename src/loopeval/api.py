from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from .checks import load_checks
from .config import LoopEvalConfig, load_config
from .models import CheckSpec, EvalSample, EvaluationReport
from .providers import (
    DecisionProvider,
    GenerativeProvider,
    build_decision_provider,
    build_generative_provider,
)
from .runtime import Evaluator
from .storage import LocalStore


class _ConfiguredFallback:
    pass


CONFIGURED_FALLBACK = _ConfiguredFallback()


class LoopEval:
    """High-level standalone API.

    Use ``arun`` inside async applications and ``run`` from scripts. Provider
    credentials remain in environment variables referenced by the config.
    """

    def __init__(
        self,
        config: LoopEvalConfig,
        *,
        checks: Sequence[CheckSpec] | None = None,
        decision_provider: DecisionProvider | None = None,
        fallback_provider: GenerativeProvider | _ConfiguredFallback | None = CONFIGURED_FALLBACK,
    ) -> None:
        self.config = config
        self.store = LocalStore(config.storage.directory, config.storage.database)
        try:
            self.evaluator = Evaluator(
                config=config,
                checks=list(checks) if checks is not None else load_checks(config.checks),
                decision_provider=decision_provider
                or build_decision_provider(config.providers.decision),
                fallback_provider=(
                    build_generative_provider(config.providers.fallback)
                    if isinstance(fallback_provider, _ConfiguredFallback)
                    else fallback_provider
                ),
                store=self.store,
            )
        except Exception:
            self.store.close()
            raise

    @classmethod
    def from_config(
        cls,
        path: str | Path = "loopeval.yaml",
        *,
        checks: Sequence[CheckSpec] | None = None,
        decision_provider: DecisionProvider | None = None,
        fallback_provider: GenerativeProvider | _ConfiguredFallback | None = CONFIGURED_FALLBACK,
    ) -> LoopEval:
        return cls(
            load_config(path),
            checks=checks,
            decision_provider=decision_provider,
            fallback_provider=fallback_provider,
        )

    async def arun(self, samples: Sequence[EvalSample | dict]) -> EvaluationReport:
        validated = [
            sample if isinstance(sample, EvalSample) else EvalSample.model_validate(sample)
            for sample in samples
        ]
        return await self.evaluator.evaluate(validated)

    def run(self, samples: Sequence[EvalSample | dict]) -> EvaluationReport:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(samples))
        raise RuntimeError("LoopEval.run() cannot be called in an active event loop; use arun()")

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> LoopEval:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
