from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml

from .models import CheckKind, CheckResult, CheckSpec, EvalSample, ResultStatus


class CheckLoadError(ValueError):
    pass


def _documents(path: Path) -> Iterable[dict[str, Any]]:
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise CheckLoadError(f"could not load {path}: {exc}") from exc
    if loaded is None:
        return []
    if isinstance(loaded, dict) and "checks" in loaded:
        checks = loaded["checks"]
        if not isinstance(checks, list):
            raise CheckLoadError(f"{path}: 'checks' must be a list")
        return checks
    if isinstance(loaded, dict):
        return [loaded]
    if isinstance(loaded, list):
        return loaded
    raise CheckLoadError(f"{path}: expected a check object or a list of checks")


def load_checks(paths: Iterable[str | Path]) -> list[CheckSpec]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.yaml")))
            files.extend(sorted(path.rglob("*.yml")))
        elif path.is_file():
            files.append(path)
        else:
            raise CheckLoadError(f"check path does not exist: {path}")

    checks: list[CheckSpec] = []
    seen: set[tuple[str, str]] = set()
    for path in files:
        for item in _documents(path):
            try:
                check = CheckSpec.model_validate(item)
            except Exception as exc:
                raise CheckLoadError(f"invalid check in {path}: {exc}") from exc
            key = (check.id, check.version)
            if key in seen:
                raise CheckLoadError(f"duplicate check {check.id}@{check.version}")
            seen.add(key)
            if check.enabled:
                checks.append(check)
    return checks


def get_field(sample: EvalSample, path: str) -> Any:
    value: Any = sample.model_dump(mode="python")
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def missing_required_fields(check: CheckSpec, sample: EvalSample) -> list[str]:
    return [
        field
        for field in check.requires
        if get_field(sample, field) in (None, "", [], {})
    ]


def applies(check: CheckSpec, sample: EvalSample) -> bool:
    return not missing_required_fields(check, sample)


Rule = Callable[[CheckSpec, EvalSample], tuple[bool, dict[str, Any]]]


def _not_empty(check: CheckSpec, sample: EvalSample) -> tuple[bool, dict[str, Any]]:
    value = get_field(sample, check.field)
    passed = value is not None and value != "" and value != [] and value != {}
    return passed, {"field": check.field, "empty": not passed}


def _exact_match(check: CheckSpec, sample: EvalSample) -> tuple[bool, dict[str, Any]]:
    actual = get_field(sample, check.field)
    expected_field = str(check.params.get("expected_field", "expected"))
    expected = get_field(sample, expected_field)
    normalize = bool(check.params.get("normalize_whitespace", True))
    case_sensitive = bool(check.params.get("case_sensitive", True))

    def clean(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if normalize:
            value = " ".join(value.split())
        return value if case_sensitive else value.casefold()

    passed = clean(actual) == clean(expected)
    return passed, {"field": check.field, "expected_field": expected_field}


def _json_valid(check: CheckSpec, sample: EvalSample) -> tuple[bool, dict[str, Any]]:
    value = get_field(sample, check.field)

    def reject_constant(constant: str) -> None:
        raise ValueError(f"{constant} is not a JSON value")

    try:
        if isinstance(value, (dict, list)):
            json.dumps(value, allow_nan=False)
            return True, {"field": check.field, "already_structured": True}
        json.loads(value, parse_constant=reject_constant)
        return True, {"field": check.field}
    except (TypeError, ValueError) as exc:
        return False, {"field": check.field, "parse_error": str(exc)}


def _regex(check: CheckSpec, sample: EvalSample) -> tuple[bool, dict[str, Any]]:
    value = str(get_field(sample, check.field))
    pattern = check.params.get("pattern")
    if not pattern:
        raise ValueError("regex rule requires params.pattern")
    flags = re.IGNORECASE if check.params.get("ignore_case") else 0
    matched = re.search(str(pattern), value, flags) is not None
    should_match = bool(check.params.get("should_match", True))
    return matched == should_match, {
        "field": check.field,
        "pattern": pattern,
        "matched": matched,
        "should_match": should_match,
    }


def _max_length(check: CheckSpec, sample: EvalSample) -> tuple[bool, dict[str, Any]]:
    value = get_field(sample, check.field)
    if not hasattr(value, "__len__"):
        raise ValueError(f"max_length requires a sized value in {check.field}")
    length = len(value)
    maximum = int(check.params.get("maximum", 0))
    if maximum <= 0:
        raise ValueError("max_length rule requires params.maximum > 0")
    return length <= maximum, {"field": check.field, "length": length, "maximum": maximum}


def _required_fields(check: CheckSpec, sample: EvalSample) -> tuple[bool, dict[str, Any]]:
    value = get_field(sample, check.field)
    required = list(check.params.get("required", []))
    if not isinstance(value, dict):
        return False, {"field": check.field, "missing": required, "reason": "not an object"}
    missing = [key for key in required if key not in value]
    return not missing, {"field": check.field, "missing": missing}


RULES: dict[str, Rule] = {
    "not_empty": _not_empty,
    "exact_match": _exact_match,
    "json_valid": _json_valid,
    "regex": _regex,
    "max_length": _max_length,
    "required_fields": _required_fields,
}


def register_rule(name: str, rule: Rule) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("rule name must be snake_case")
    RULES[name] = rule


def run_deterministic(check: CheckSpec, sample: EvalSample) -> CheckResult:
    if check.kind != CheckKind.DETERMINISTIC:
        raise ValueError(f"{check.id} is not deterministic")
    missing = missing_required_fields(check, sample)
    # Presence checks intentionally fail on absent data. Other built-in rules
    # need a value to judge, and exact matching also needs a reference value.
    if check.rule in {"exact_match", "json_valid", "regex", "max_length", "required_fields"}:
        fields = [check.field]
        if check.rule == "exact_match":
            fields.append(str(check.params.get("expected_field", "expected")))
        missing.extend(field for field in fields if get_field(sample, field) is None)
    if missing:
        return CheckResult(
            check_id=check.id,
            check_version=check.version,
            status=ResultStatus.SKIPPED,
            severity=check.severity,
            evidence={"missing_required_fields": list(dict.fromkeys(missing))},
        )
    rule = RULES.get(check.rule or "")
    if rule is None:
        return CheckResult(
            check_id=check.id,
            check_version=check.version,
            status=ResultStatus.ERROR,
            severity=check.severity,
            error=f"unknown deterministic rule: {check.rule}",
        )
    try:
        passed, evidence = rule(check, sample)
    except Exception as exc:
        return CheckResult(
            check_id=check.id,
            check_version=check.version,
            status=ResultStatus.ERROR,
            severity=check.severity,
            error=str(exc),
        )
    return CheckResult(
        check_id=check.id,
        check_version=check.version,
        status=ResultStatus.PASS if passed else ResultStatus.FAIL,
        severity=check.severity,
        score=1.0 if passed else 0.0,
        evidence=evidence,
    )
