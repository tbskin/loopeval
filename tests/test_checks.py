from __future__ import annotations

from pathlib import Path

import pytest

from loopeval.checks import (
    CheckLoadError,
    applies,
    get_field,
    load_checks,
    register_rule,
    run_deterministic,
)
from loopeval.models import CheckSpec, EvalSample, ResultStatus


def spec(rule: str, **kwargs: object) -> CheckSpec:
    return CheckSpec(
        id=f"test.{rule}",
        name=rule,
        description=rule,
        kind="deterministic",
        rule=rule,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("check", "sample", "expected"),
    [
        (spec("not_empty"), EvalSample(input="x", output=""), ResultStatus.FAIL),
        (
            spec("exact_match", params={"case_sensitive": False}),
            EvalSample(input="x", output=" HELLO ", expected="hello"),
            ResultStatus.PASS,
        ),
        (spec("json_valid"), EvalSample(input="x", output='{"ok": true}'), ResultStatus.PASS),
        (
            spec("regex", params={"pattern": "secret", "should_match": False}),
            EvalSample(input="x", output="safe"),
            ResultStatus.PASS,
        ),
        (
            spec("max_length", params={"maximum": 3}),
            EvalSample(input="x", output="four"),
            ResultStatus.FAIL,
        ),
        (
            spec("required_fields", field="data", params={"required": ["answer"]}),
            EvalSample(input="x", data={"answer": 42}),
            ResultStatus.PASS,
        ),
    ],
)
def test_builtin_rules(check: CheckSpec, sample: EvalSample, expected: ResultStatus) -> None:
    assert run_deterministic(check, sample).status == expected


def test_nested_field_requirements_and_skip() -> None:
    sample = EvalSample(input="x", trace=[{"tool": "search"}], data={"nested": {"x": 1}})
    assert get_field(sample, "trace.0.tool") == "search"
    assert get_field(sample, "data.nested.x") == 1
    assert get_field(sample, "trace.5.tool") is None
    check = spec("not_empty", requires=["context"])
    assert not applies(check, sample)
    assert run_deterministic(check, sample).status == ResultStatus.SKIPPED


@pytest.mark.parametrize(
    "check",
    [
        spec("exact_match"),
        spec("json_valid"),
        spec("regex", params={"pattern": "secret", "should_match": False}),
        spec("max_length", params={"maximum": 3}),
        spec("required_fields", params={"required": ["answer"]}),
    ],
)
def test_rules_skip_missing_evidence(check: CheckSpec) -> None:
    result = run_deterministic(check, EvalSample(input="x"))
    assert result.status == ResultStatus.SKIPPED
    assert "output" in result.evidence["missing_required_fields"]


def test_exact_match_requires_reference_but_allows_empty_values() -> None:
    missing = run_deterministic(spec("exact_match"), EvalSample(input="x", output="answer"))
    assert missing.status == ResultStatus.SKIPPED
    assert missing.evidence == {"missing_required_fields": ["expected"]}
    empty = run_deterministic(
        spec("exact_match"), EvalSample(input="x", output="", expected="")
    )
    assert empty.status == ResultStatus.PASS


def test_rule_requirements_report_only_missing_fields() -> None:
    check = spec("not_empty", requires=["input", "context"])
    result = run_deterministic(check, EvalSample(input="x", output="answer"))
    assert result.evidence == {"missing_required_fields": ["context"]}


@pytest.mark.parametrize("output", ["NaN", "Infinity", '{"value": NaN}', {"value": float("nan")}])
def test_json_valid_rejects_non_json_constants(output: object) -> None:
    result = run_deterministic(spec("json_valid"), EvalSample(input="x", output=output))
    assert result.status == ResultStatus.FAIL


def test_regex_preserves_falsey_values() -> None:
    result = run_deterministic(
        spec("regex", params={"pattern": "^0$"}), EvalSample(input="x", output=0)
    )
    assert result.status == ResultStatus.PASS


def test_max_length_rejects_values_without_length() -> None:
    result = run_deterministic(
        spec("max_length", params={"maximum": 10}), EvalSample(input="x", output=123)
    )
    assert result.status == ResultStatus.ERROR


def test_unknown_and_broken_rule_become_errors() -> None:
    assert run_deterministic(spec("missing"), EvalSample(input="x")).status == ResultStatus.ERROR
    assert (
        run_deterministic(spec("regex", params={}), EvalSample(input="x", output="x")).status
        == ResultStatus.ERROR
    )


def test_custom_rule_registration() -> None:
    register_rule("always_pass_for_test", lambda check, sample: (True, {"custom": True}))
    result = run_deterministic(spec("always_pass_for_test"), EvalSample(input="x"))
    assert result.status == ResultStatus.PASS
    assert result.evidence == {"custom": True}
    with pytest.raises(ValueError, match="snake_case"):
        register_rule("Bad Name", lambda check, sample: (True, {}))


def test_load_checks_directory_and_errors(tmp_path: Path) -> None:
    checks_dir = tmp_path / "checks"
    checks_dir.mkdir()
    (checks_dir / "one.yaml").write_text(
        "id: output.present\nname: Present\ndescription: Present\n"
        "kind: deterministic\nrule: not_empty\n"
    )
    (checks_dir / "two.yml").write_text(
        "checks:\n  - id: quality.bad\n    name: Bad\n    description: Bad\n"
        "    kind: noul\n    instructions: Is it bad?\n"
    )
    assert {check.id for check in load_checks([checks_dir])} == {
        "output.present",
        "quality.bad",
    }
    with pytest.raises(CheckLoadError, match="does not exist"):
        load_checks([tmp_path / "missing"])
    (checks_dir / "bad.yaml").write_text("checks: nope\n")
    with pytest.raises(CheckLoadError, match="must be a list"):
        load_checks([checks_dir / "bad.yaml"])


def test_load_checks_rejects_duplicate_version(tmp_path: Path) -> None:
    content = (
        "id: output.present\nversion: 1.0.0\nname: Present\ndescription: Present\n"
        "kind: deterministic\nrule: not_empty\n"
    )
    (tmp_path / "a.yaml").write_text(content)
    (tmp_path / "b.yaml").write_text(content)
    with pytest.raises(CheckLoadError, match="duplicate check"):
        load_checks([tmp_path])
