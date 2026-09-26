from __future__ import annotations

from pathlib import Path

import pytest

from loopeval.config import LoopEvalConfig
from loopeval.models import CheckSpec


@pytest.fixture
def config_factory(tmp_path: Path):
    def make(**overrides: object) -> LoopEvalConfig:
        data: dict[str, object] = {
            "version": 1,
            "project": "test-project",
            "checks": [],
            "providers": {
                "decision": {"type": "mock", "model": "mock-decision"},
                "fallback": {"type": "mock", "model": "mock-generative"},
            },
            "escalation": {
                "random_audit_rate": 0,
                "coverage_check": True,
            },
            "storage": {"directory": str(tmp_path / "state"), "cache": True},
        }
        for key, value in overrides.items():
            data[key] = value
        return LoopEvalConfig.model_validate(data)

    return make


@pytest.fixture
def not_empty_check() -> CheckSpec:
    return CheckSpec(
        id="output.not_empty",
        name="Output present",
        description="Output must not be empty.",
        kind="deterministic",
        rule="not_empty",
        field="output",
    )


@pytest.fixture
def semantic_check() -> CheckSpec:
    return CheckSpec(
        id="quality.bad",
        name="Bad output",
        description="Output has the known bad pattern.",
        kind="noul",
        instructions="Does `output` contain the known bad pattern?",
        criteria={"true": "It does.", "false": "It does not."},
    )
