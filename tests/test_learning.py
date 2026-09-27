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
    validate_holdout_samples,
    validation_metrics,
)
from loopeval.models import BootstrapProposal, CandidateCheck, EvalSample
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
    assert metrics["recall"] == 0.5
    assert metrics["positive_examples"] == 2
    assert metrics["negative_examples"] == 2
    assert metrics["unresolved_positive"] == 1
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
    first_delimiter, first = fenced_sample({"output": "x" * 100}, [], 300)
    second_delimiter, second = fenced_sample({"output": "x" * 100}, [], 300)
    assert len(first_delimiter) == 24
    assert first_delimiter != second_delimiter
    assert first_delimiter in first and second_delimiter in second
    with pytest.raises(ValueError, match="exceeds max_state_chars"):
        fenced_sample({"output": "x" * 100}, [], 30)


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
        validation_metrics([True, False], [True, False]),
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
    with pytest.raises(ValueError, match="already active"):
        promote_candidate(
            store=store,
            candidate_id=key,
            destination=tmp_path / "duplicate",
            policy=policy,
            force=True,
        )
    store.close()


def test_promotion_cannot_hide_missed_failures_in_abstentions() -> None:
    predictions = [True] + [None] * 9 + [False] * 90
    labels = [True] * 10 + [False] * 90
    metrics = validation_metrics(predictions, labels)
    assert metrics["precision"] == 1.0
    assert metrics["coverage"] == 0.91
    assert metrics["recall"] == 0.1
    assert not meets_promotion_policy(metrics, PromotionPolicy())


@pytest.mark.parametrize("label", [True, False])
def test_promotion_requires_both_positive_and_negative_examples(label: bool) -> None:
    metrics = validation_metrics([label] * 20, [label] * 20)
    policy = PromotionPolicy(minimum_precision=0, minimum_recall=0)
    assert not meets_promotion_policy(metrics, policy)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_calibration_rejects_invalid_scores(score: float) -> None:
    with pytest.raises(ValueError, match="finite values between 0 and 1"):
        calibrate_noul_thresholds([score], [True])


def test_validation_rejects_missing_labels_duplicates_and_discovery_samples() -> None:
    positive = EvalSample(id="positive", input="one", labels=["learned.regression"])
    negative = EvalSample(id="negative", input="two", labels=[])
    validate_holdout_samples([positive, negative])
    with pytest.raises(ValueError, match="at least one"):
        validate_holdout_samples([])
    with pytest.raises(ValueError, match="labels field"):
        validate_holdout_samples([EvalSample(input="unlabeled")])
    with pytest.raises(ValueError, match="unique sample ids"):
        validate_holdout_samples([positive, positive])
    with pytest.raises(ValueError, match="distinct sample content"):
        validate_holdout_samples([positive, positive.model_copy(update={"id": "copy"})])
    with pytest.raises(ValueError, match="reuse discovery or bootstrap samples"):
        validate_holdout_samples([positive], source_sample_ids=[positive.sample_id])
    with pytest.raises(ValueError, match="reuse discovery or bootstrap sample content"):
        validate_holdout_samples(
            [positive.model_copy(update={"id": "renamed"})],
            source_sample_hashes=[positive.content_hash],
        )


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
