from __future__ import annotations

import pytest
from pydantic import ValidationError

from loopeval.models import (
    CheckLifecycle,
    CheckSpec,
    DecisionAnswer,
    EvalSample,
    QuestionKind,
)


def test_sample_id_is_stable_and_explicit_id_wins() -> None:
    first = EvalSample(input="hello", output="world")
    second = EvalSample(input="hello", output="world")
    assert first.sample_id == second.sample_id
    assert EvalSample(id="mine", input="hello").sample_id == "mine"
    assert first.state()["sample_id"] == first.sample_id


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"id": "INVALID", "kind": "deterministic", "rule": "not_empty"}, "check id"),
        ({"id": "x", "kind": "deterministic"}, "require rule"),
        ({"id": "x", "kind": "noul"}, "require instructions"),
        (
            {"id": "x", "kind": "choice", "instructions": "choose", "criteria": {"x": "X"}},
            "failure_label",
        ),
        (
            {"id": "x", "kind": "score", "instructions": "score", "criteria": ["a", "b"]},
            "failure_score_gte",
        ),
    ],
)
def test_check_shape_validation(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        CheckSpec(name="test", description="test", **payload)


def test_check_question_and_lifecycle() -> None:
    check = CheckSpec(
        id="quality.bad",
        name="Bad",
        description="Bad output",
        kind="noul",
        instructions="Is it bad?",
        lifecycle=CheckLifecycle.ACTIVE,
    )
    question = check.question()
    assert question.kind == QuestionKind.NOUL
    assert question.provider_payload() == {"type": "noul", "instructions": "Is it bad?"}
    with pytest.raises(ValueError, match="not a semantic"):
        CheckSpec(
            id="output.present",
            name="Present",
            description="Present",
            kind="deterministic",
            rule="not_empty",
        ).question()


def test_decision_answer_rejects_malformed_provider_output() -> None:
    with pytest.raises(ValidationError, match="noul answers require"):
        DecisionAnswer(kind="noul")
    with pytest.raises(ValidationError, match="choice answers require"):
        DecisionAnswer(kind="choice", choice="bad")
    with pytest.raises(ValidationError, match="present in probabilities"):
        DecisionAnswer(kind="choice", choice="bad", confidence=0.9, probabilities={"good": 1.0})
    with pytest.raises(ValidationError, match="between 0 and 1"):
        DecisionAnswer(kind="noul", noul=0.2, probabilities={"bad": 1.2})


def test_valid_choice_and_score_answers() -> None:
    choice = DecisionAnswer(
        kind="choice", choice="bad", confidence=0.9, probabilities={"bad": 0.9, "ok": 0.1}
    )
    score = DecisionAnswer(
        kind="score", score=1.5, confidence=0.8, probabilities={"1": 0.5, "2": 0.5}
    )
    assert choice.choice == "bad"
    assert score.score == 1.5
