from __future__ import annotations

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
    store.fail_run(failed_id, "boom")
    assert store.list_runs()[0]["status"] == "failed"
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
    for sample_id in ("one", "two", "two"):
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
    row = store.get_candidate("cand_x")
    assert row and row["evidence_count"] == 2
    assert row["sample_ids"] == ["one", "two"]
    assert len(store.list_candidate_evidence("cand_x")) == 2
    assert len(store.list_candidates("proposed")) == 1
    with pytest.raises(ValueError, match="approve or reject"):
        store.review_candidate("cand_x", "maybe", None)
    with pytest.raises(KeyError, match="not found"):
        store.review_candidate("missing", "approve", None)
    store.review_candidate("cand_x", "reject", "not reusable")
    assert store.get_candidate("cand_x")["status"] == "rejected"
    with pytest.raises(KeyError, match="not found"):
        store.save_validation("missing", {}, False)
    with pytest.raises(KeyError, match="not found"):
        store.mark_promoted("missing")
    store.close()
