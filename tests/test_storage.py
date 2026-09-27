from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from loopeval.models import EvaluationReport, OverallVerdict, SampleResult
from loopeval.storage import LocalStore


def test_store_run_cache_and_failure(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")
    run_id, started = store.start_run("hash")
    result = SampleResult(sample_id="sample", verdict=OverallVerdict.PASS, checks=[])
    store.save_result(run_id, result)
    report = EvaluationReport(
        run_id=run_id,
        results=[result],
        started_at=started,
        completed_at=datetime.now(UTC),
        config_hash="hash",
    )
    artifact = store.finish_run(report)
    assert artifact.exists()
    assert store.get_report(run_id) == report
    assert store.list_runs()[0]["status"] == "completed"
    store.cache_put("key", "decision", {"answer": 1})
    assert store.cache_get("key") == {"answer": 1}

    failed_id, _ = store.start_run("hash")
    assert store.get_report(failed_id) is None
    store.fail_run(failed_id, "boom")
    assert store.list_runs()[0]["status"] == "failed"
    assert store.get_report(failed_id) is None
    store.close()


def test_candidate_dedup_review_and_errors(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")
    item = {
        "id": "learned.x",
        "name": "X",
        "description": "X",
        "kind": "noul",
        "instructions": "Is X present?",
    }
    for sample_id in ("one", "two"):
        store.upsert_candidate(
            candidate_id="cand_x",
            check_id="learned.x",
            title="X",
            description="X",
            check_json=item,
            sample_id=sample_id,
            evidence=f"evidence for {sample_id}",
            confidence=0.9,
        )
    changed_item = {**item, "name": "Mutated", "instructions": "A different question?"}
    store.upsert_candidate(
        candidate_id="cand_x",
        check_id="learned.x",
        title="Mutated",
        description="Changed after observation",
        check_json=changed_item,
        sample_id="two",
        evidence="different wording for the same candidate id",
        confidence=0.8,
    )
    row = store.get_candidate("cand_x")
    assert row and row["evidence_count"] == 2
    assert row["sample_ids"] == ["one", "two"]
    assert row["source_sample_ids"] == ["one", "two"]
    assert row["title"] == "X"
    assert row["check"] == item
    assert len(store.list_candidate_evidence("cand_x")) == 2
    assert len(store.list_candidates("proposed")) == 1
    with pytest.raises(ValueError, match="approve or reject"):
        store.review_candidate("cand_x", "maybe", None)
    with pytest.raises(KeyError, match="not found"):
        store.review_candidate("missing", "approve", None)
    store.review_candidate("cand_x", "reject", "not reusable")
    assert store.get_candidate("cand_x")["status"] == "rejected"
    revised = {**item, "name": "Human revised X"}
    store.revise_candidate("cand_x", revised, "Clarified the boundary")
    revised_row = store.get_candidate("cand_x")
    assert revised_row["status"] == "proposed"
    assert revised_row["title"] == "Human revised X"
    assert revised_row["validation"] is None
    with pytest.raises(ValueError, match="check id"):
        store.revise_candidate("cand_x", {**revised, "id": "learned.other"}, None)
    with pytest.raises(ValueError, match="check kind"):
        store.revise_candidate("cand_x", {**revised, "kind": "choice"}, None)
    with pytest.raises(KeyError, match="not found"):
        store.save_validation("missing", {}, False)
    with pytest.raises(KeyError, match="not found"):
        store.mark_promoted("missing")
    store.close()


def test_candidate_sources_and_evidence_survive_concurrent_observations(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "state")

    def observe(index: int) -> None:
        store.upsert_candidate(
            candidate_id="cand_x", check_id="learned.x", title="X", description="X",
            check_json={"id": "learned.x", "kind": "noul"}, sample_id=f"batch-{index}",
            source_sample_ids=[f"sample-{index}"], source_sample_hashes=[f"hash-{index}"],
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(observe, range(20)))
    observe(0)
    row = store.get_candidate("cand_x")
    assert row is not None
    assert row["evidence_count"] == 20
    assert len(store.list_candidate_evidence("cand_x")) == 20
    assert {f"sample-{index}" for index in range(20)} <= set(row["source_sample_ids"])
    assert set(row["source_sample_hashes"]) == {f"hash-{index}" for index in range(20)}
    store.close()
    reopened = LocalStore(tmp_path / "state")
    assert reopened.get_candidate("cand_x")["source_sample_hashes"] == row["source_sample_hashes"]
    reopened.close()


def test_validation_cannot_approve_candidates_and_review_cannot_deactivate_them(
    tmp_path: Path,
) -> None:
    store = LocalStore(tmp_path / "state")
    item = {
        "id": "learned.x", "name": "X", "description": "X", "kind": "noul",
        "instructions": "Is X present?",
    }
    store.upsert_candidate(
        candidate_id="cand_x", check_id="learned.x", title="X", description="X",
        check_json=item, sample_id="one",
    )
    with pytest.raises(ValueError, match="approved before validation"):
        store.save_validation("cand_x", {}, True)
    store.review_candidate("cand_x", "approve", None)
    store.save_validation("cand_x", {"examples": 20}, True)
    store.revise_candidate("cand_x", {**item, "name": "Revised"}, None)
    assert store.get_candidate("cand_x")["validation"] is None
    with pytest.raises(ValueError, match="approved before validation"):
        store.save_validation("cand_x", {}, True)
    store.mark_promoted("cand_x")
    with pytest.raises(ValueError, match="active candidates cannot be reviewed"):
        store.review_candidate("cand_x", "reject", "This would not remove the active YAML")
    with pytest.raises(ValueError, match="approved before validation"):
        store.save_validation("cand_x", {}, True)
    assert store.get_candidate("cand_x")["status"] == "active"
    store.close()
