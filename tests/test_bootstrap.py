from __future__ import annotations

import copy
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
    assert "one" in stored["source_sample_ids"]
    assert stored["source_sample_hashes"] == [
        EvalSample(id="one", input="Refund?", output="Sixty days").content_hash
    ]
    store.close()


class RecordingBootstrapProvider(GenerativeProvider):
    name = "recording"
    model = "recording"

    def __init__(self) -> None:
        self.user: str | None = None
        self.raw: dict[str, Any] = {
            "summary": "Initial policy checks",
            "candidates": [
                {
                    "check": {
                        "id": "policy.violation",
                        "name": "Policy violation",
                        "description": "The answer contradicts the policy.",
                        "kind": "noul",
                        "instructions": "Does the answer contradict the supplied policy?",
                    },
                    "rationale": "Answers should comply with the supplied policy.",
                    "confidence": 0.9,
                }
            ],
        }

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], ProviderUsage, int]:
        self.user = user
        return self.raw, ProviderUsage(), 0


@pytest.mark.asyncio
async def test_bootstrap_excludes_labels_and_tracks_changed_scenario_content(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")
    provider = RecordingBootstrapProvider()
    sample = EvalSample(
        id="scenario", input="Refund?", output="Thirty days",
        labels=["private-human-label"], expected_verdict="pass",
    )
    _, keys = await propose_initial_checks(
        provider=provider, requirements="Follow policy", samples=[sample],
        active_checks=[], store=store,
    )
    assert provider.user is not None
    assert "private-human-label" not in provider.user
    assert '"labels"' not in provider.user
    assert '"expected_verdict"' not in provider.user
    changed = sample.model_copy(update={"output": "Sixty days"})
    await propose_initial_checks(
        provider=provider, requirements="Follow policy", samples=[changed],
        active_checks=[], store=store,
    )
    stored = store.get_candidate(keys[0])
    assert stored is not None
    assert stored["evidence_count"] == 2
    assert set(stored["source_sample_hashes"]) == {sample.content_hash, changed.content_hash}
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["duplicate", "unknown_rule"])
async def test_bootstrap_rejects_entire_invalid_batch(tmp_path: Path, invalid_kind: str) -> None:
    store = LocalStore(tmp_path / "state")
    provider = RecordingBootstrapProvider()
    second = copy.deepcopy(provider.raw["candidates"][0])
    if invalid_kind == "unknown_rule":
        second["check"] = {
            "id": "output.magic", "name": "Magic", "description": "Invalid rule",
            "kind": "deterministic", "rule": "run_generated_code",
        }
    provider.raw["candidates"].append(second)
    with pytest.raises(ValueError, match=r"duplicate check|unknown deterministic rule"):
        await propose_initial_checks(
            provider=provider, requirements="Follow policy", samples=[EvalSample(input="x")],
            active_checks=[], store=store,
        )
    assert store.list_candidates() == []
    store.close()


@pytest.mark.asyncio
async def test_bootstrap_rejects_oversized_input_before_call(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")
    provider = RecordingBootstrapProvider()
    with pytest.raises(ValueError, match="exceed max_state_chars"):
        await propose_initial_checks(
            provider=provider, requirements="Follow policy", samples=[EvalSample(input="x" * 100)],
            active_checks=[], store=store, max_chars=50,
        )
    assert provider.user is None
    assert store.list_candidates() == []
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
