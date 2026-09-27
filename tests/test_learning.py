from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from loopeval.config import PromotionPolicy
from loopeval.learning import (
    calibrate_noul_thresholds,
    candidate_key,
    fallback_schema,
    fenced_sample,
    meets_promotion_policy,
    promote_candidate,
    strict_provider_schema,
    validation_metrics,
)
from loopeval.models import BootstrapProposal, CandidateCheck
from loopeval.storage import LocalStore


def candidate() -> CandidateCheck:
    return CandidateCheck(
        id="learned.regression",
        name="Regression",
        description="Detect a regression.",
        kind="noul",
        instructions="Does the output contain the regression?",
    )


def test_validation_metrics_and_policy() -> None:
    metrics = validation_metrics([True, True, False, None], [True, False, False, True])
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["unresolved"] == 1
    assert metrics["precision"] == 0.5
    assert meets_promotion_policy(
        metrics,
        PromotionPolicy(
            minimum_examples=4,
            minimum_precision=0.5,
            minimum_recall=0.5,
            minimum_coverage=0.75,
        ),
    )
    with pytest.raises(ValueError, match="counts differ"):
        validation_metrics([True], [])


def test_noul_threshold_calibration() -> None:
    recommendation = calibrate_noul_thresholds(
        [0.95, 0.85, 0.2, 0.1, None],
        [True, True, False, False, True],
        minimum_precision=0.9,
        minimum_coverage=0.8,
    )
    assert recommendation["meets_requirements"] is True
    assert recommendation["metrics"]["precision"] == 1
    assert recommendation["metrics"]["coverage"] >= 0.8
    assert recommendation["pass_threshold"] < recommendation["failure_threshold"]
    with pytest.raises(ValueError, match="counts differ"):
        calibrate_noul_thresholds([0.5], [])


def test_fenced_sample_is_bounded_and_randomly_delimited() -> None:
    first_delimiter, first = fenced_sample({"output": "x" * 100}, [], 30)
    second_delimiter, second = fenced_sample({"output": "x" * 100}, [], 30)
    assert len(first_delimiter) == 24
    assert first_delimiter != second_delimiter
    assert first_delimiter in first and second_delimiter in second


def test_candidate_key_is_stable() -> None:
    assert candidate_key(candidate()) == candidate_key(candidate())


def test_fallback_schema_is_strict_provider_compatible() -> None:
    schema = fallback_schema()
    assert set(schema["required"]) == set(schema["properties"])
    candidate_schema = schema["$defs"]["CandidateCheck"]
    assert set(candidate_schema["required"]) == set(candidate_schema["properties"])
    assert candidate_schema["additionalProperties"] is False
    bootstrap = strict_provider_schema(BootstrapProposal)
    assert set(bootstrap["required"]) == set(bootstrap["properties"])


def test_promotion_requires_review_and_validation(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")
    item = candidate()
    key = candidate_key(item)
    store.upsert_candidate(
        candidate_id=key,
        check_id=item.id,
        title=item.name,
        description=item.description,
        check_json=item.model_dump(mode="json"),
        sample_id="sample-1",
    )
    policy = PromotionPolicy(minimum_examples=2, minimum_precision=0.9, minimum_recall=0.5)
    with pytest.raises(ValueError, match="approved"):
        promote_candidate(
            store=store, candidate_id=key, destination=tmp_path / "checks", policy=policy
        )
    store.review_candidate(key, "approve", "looks good")
    with pytest.raises(ValueError, match="held-out"):
        promote_candidate(
            store=store, candidate_id=key, destination=tmp_path / "checks", policy=policy
        )
    store.save_validation(
        key,
        {"examples": 2, "precision": 1.0, "recall": 1.0, "coverage": 1.0},
        passed=True,
    )
    target = promote_candidate(
        store=store, candidate_id=key, destination=tmp_path / "checks", policy=policy
    )
    document = yaml.safe_load(target.read_text())
    assert document["lifecycle"] == "active"
    assert store.get_candidate(key)["status"] == "active"
    with pytest.raises(ValueError, match="active candidates"):
        store.revise_candidate(key, item.model_dump(mode="json"), None)
    store.close()


def test_force_promotion_is_available_for_experiments(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")
    item = candidate()
    key = candidate_key(item)
    store.upsert_candidate(
        candidate_id=key,
        check_id=item.id,
        title=item.name,
        description=item.description,
        check_json=item.model_dump(mode="json"),
        sample_id="sample-1",
    )
    target = promote_candidate(
        store=store,
        candidate_id=key,
        destination=tmp_path / "forced",
        policy=PromotionPolicy(),
        force=True,
    )
    assert target.exists()
    store.close()


def test_promotion_refuses_collision_and_overwrite(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")
    item = candidate()
    key = candidate_key(item)
    store.upsert_candidate(
        candidate_id=key,
        check_id=item.id,
        title=item.name,
        description=item.description,
        check_json=item.model_dump(mode="json"),
        sample_id="sample-1",
    )
    with pytest.raises(ValueError, match="already uses id"):
        promote_candidate(
            store=store,
            candidate_id=key,
            destination=tmp_path / "checks",
            policy=PromotionPolicy(),
            existing_check_ids={item.id},
            force=True,
        )
    destination = tmp_path / "checks"
    destination.mkdir()
    (destination / f"{item.id}.yaml").write_text("existing\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        promote_candidate(
            store=store,
            candidate_id=key,
            destination=destination,
            policy=PromotionPolicy(),
            force=True,
        )
    store.close()
