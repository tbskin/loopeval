from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from loopeval.bootstrap import propose_initial_checks
from loopeval.config import ProviderConfig
from loopeval.models import EvalSample, ProviderUsage
from loopeval.providers.base import GenerativeProvider
from loopeval.providers.mock import MockGenerativeProvider
from loopeval.storage import LocalStore


@pytest.mark.asyncio
async def test_bootstrap_proposes_reviewable_candidates(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")
    provider = MockGenerativeProvider(ProviderConfig(type="mock", model="bootstrap"))
    proposal, keys = await propose_initial_checks(
        provider=provider,
        requirements="Answers must follow the refund policy.",
        samples=[EvalSample(id="one", input="Refund?", output="Sixty days")],
        active_checks=[],
        store=store,
    )
    assert proposal.candidates[0].check.id == "quality.requirement_violation"
    assert len(keys) == 1
    stored = store.get_candidate(keys[0])
    assert stored and stored["status"] == "proposed"
    assert stored["check_id"] == "quality.requirement_violation"
    assert "semantic" in store.list_candidate_evidence(keys[0])[0]["evidence"]
    store.close()


class InvalidBootstrapProvider(GenerativeProvider):
    name = "invalid"
    model = "invalid"

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], ProviderUsage, int]:
        return (
            {
                "summary": "Invalid deterministic proposal",
                "candidates": [
                    {
                        "check": {
                            "id": "output.magic",
                            "name": "Magic",
                            "description": "Uses an unknown rule.",
                            "kind": "deterministic",
                            "rule": "run_generated_code",
                        },
                        "rationale": "Invalid on purpose.",
                        "confidence": 1,
                    }
                ],
            },
            ProviderUsage(),
            0,
        )


@pytest.mark.asyncio
async def test_bootstrap_rejects_unknown_deterministic_rules(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")
    with pytest.raises(ValueError, match="unknown deterministic rule"):
        await propose_initial_checks(
            provider=InvalidBootstrapProvider(),
            requirements="Output must be magic.",
            samples=[EvalSample(input="x", output="y")],
            active_checks=[],
            store=store,
        )
    assert store.list_candidates() == []
    store.close()


@pytest.mark.asyncio
async def test_bootstrap_requires_requirements_and_samples(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")
    provider = MockGenerativeProvider(ProviderConfig(type="mock", model="bootstrap"))
    with pytest.raises(ValueError, match="requirements"):
        await propose_initial_checks(
            provider=provider,
            requirements="",
            samples=[EvalSample(input="x")],
            active_checks=[],
            store=store,
        )
    with pytest.raises(ValueError, match="scenario"):
        await propose_initial_checks(
            provider=provider,
            requirements="Be correct",
            samples=[],
            active_checks=[],
            store=store,
        )
    store.close()
