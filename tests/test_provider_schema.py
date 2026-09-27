from __future__ import annotations

import json
from typing import Any

import pytest

from loopeval.models import BootstrapProposal, CandidateCheck, FallbackVerdict
from loopeval.providers.base import ProviderError
from loopeval.providers.schema import decode_response, provider_schema


def assert_closed_schema(node: dict[str, Any]) -> None:
    assert "default" not in node
    if node.get("type") == "object":
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])
    for key in ("$defs", "properties"):
        for child in node.get(key, {}).values():
            assert_closed_schema(child)
    for branch in node.get("anyOf", []):
        assert_closed_schema(branch)
    if "items" in node:
        assert_closed_schema(node["items"])


@pytest.mark.parametrize("model", [BootstrapProposal, FallbackVerdict])
def test_actual_learning_schemas_use_closed_provider_objects(model: Any) -> None:
    original = model.model_json_schema()
    before = json.dumps(original, sort_keys=True)
    wire = provider_schema(original)
    assert_closed_schema(wire)
    candidate = wire["$defs"]["CandidateCheck"]
    for name in ("criteria", "params"):
        assert candidate["properties"][name]["type"] == "string"
    assert candidate["properties"]["examples"]["items"]["type"] == "string"
    assert json.dumps(original, sort_keys=True) == before


@pytest.mark.parametrize("criteria", [None, {"true": "bad", "false": "good"}, ["ok", "bad"]])
def test_candidate_json_fields_roundtrip_without_changing_public_schema(criteria: Any) -> None:
    check = CandidateCheck(
        id="quality.bad",
        name="Bad",
        description="A failure",
        kind="noul",
        instructions="Is there a failure?",
        criteria=criteria,
        params={"expected": {"nested": [1, True, None]}},
        examples=[{"input": "question", "output": "answer", "labels": []}],
    )
    wire_check = check.model_dump(mode="json")
    wire_check["criteria"] = json.dumps(check.criteria)
    wire_check["params"] = json.dumps(check.params)
    wire_check["examples"] = [json.dumps(example) for example in check.examples]
    fallback = FallbackVerdict(
        category="novel_failure",
        evidence="Observed a failure",
        confidence=0.9,
        candidate_check=check,
    )
    wire = fallback.model_dump(mode="json") | {"candidate_check": wire_check}
    restored = decode_response(wire, FallbackVerdict.model_json_schema())
    assert FallbackVerdict.model_validate(restored) == fallback
    bootstrap = {
        "summary": "New check",
        "candidates": [
            {"check": wire_check, "rationale": "Reusable", "confidence": 0.9},
        ],
    }
    proposal = BootstrapProposal.model_validate(
        decode_response(bootstrap, BootstrapProposal.model_json_schema())
    )
    assert proposal.candidates[0].check == check


@pytest.mark.parametrize("value", ["not JSON", "[]", '{"x": NaN}', '{"x": Infinity}', {}])
def test_encoded_objects_reject_malformed_nonfinite_and_wrong_type_values(value: Any) -> None:
    schema = {
        "type": "object",
        "properties": {"params": {"type": "object"}},
        "additionalProperties": False,
    }
    with pytest.raises(ProviderError):
        decode_response({"params": value}, schema)


def test_null_candidate_is_not_mistaken_for_an_encoded_object() -> None:
    verdict = FallbackVerdict(category="acceptable", evidence="No defect", confidence=0.8)
    assert (
        decode_response(verdict.model_dump(), FallbackVerdict.model_json_schema())
        == verdict.model_dump()
    )
