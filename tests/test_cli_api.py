from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from loopeval import LoopEval
from loopeval.cli import app
from loopeval.config import ProviderConfig
from loopeval.providers.mock import MockDecisionProvider

runner = CliRunner()


def test_offline_cli_end_to_end(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    initialized = runner.invoke(app, ["init", str(project), "--offline"])
    assert initialized.exit_code == 0, initialized.output
    assert (project / "loopeval.yaml").exists()

    doctor = runner.invoke(app, ["doctor", "-c", str(project / "loopeval.yaml")])
    assert doctor.exit_code == 0, doctor.output
    assert "Config: OK" in doctor.output

    live_doctor = runner.invoke(app, ["doctor", "--live", "-c", str(project / "loopeval.yaml")])
    assert live_doctor.exit_code == 0, live_doctor.output
    assert "Decision endpoint: verified" in live_doctor.output
    assert "Fallback endpoint: verified" in live_doctor.output

    run = runner.invoke(
        app,
        ["run", str(project / "samples.jsonl"), "-c", str(project / "loopeval.yaml")],
    )
    assert run.exit_code == 0, run.output
    summary = json.loads(run.output)
    assert summary["samples"] == 2
    assert summary["verdicts"] == {"pass": 1, "fail": 1}
    before_run_id = summary["run_id"]

    learned = runner.invoke(app, ["learn", before_run_id, "-c", str(project / "loopeval.yaml")])
    assert learned.exit_code == 0, learned.output
    assert "Learned 1 candidate" in learned.output

    candidates = runner.invoke(app, ["candidates", "-c", str(project / "loopeval.yaml")])
    assert candidates.exit_code == 0
    assert "learned.novel_pattern" in candidates.output

    with LoopEval.from_config(project / "loopeval.yaml") as loop:
        candidate_id = loop.store.list_candidates()[0]["id"]
    shown = runner.invoke(app, ["candidate", candidate_id, "-c", str(project / "loopeval.yaml")])
    assert shown.exit_code == 0, shown.output
    assert "The output contains [NOVEL]." in shown.output

    reviewed = runner.invoke(
        app,
        [
            "review",
            candidate_id,
            "--decision",
            "approve",
            "--notes",
            "Reusable offline pattern.",
            "-c",
            str(project / "loopeval.yaml"),
        ],
    )
    assert reviewed.exit_code == 0, reviewed.output

    holdout = project / "holdout.jsonl"
    holdout.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"holdout-{index}",
                    "input": f"holdout scenario {index}",
                    "output": (f"[NOVEL] pattern {index}" if index < 10 else f"healthy {index}"),
                    "labels": ["learned.novel_pattern"] if index < 10 else [],
                }
            )
            + "\n"
            for index in range(20)
        )
    )
    validated = runner.invoke(
        app,
        [
            "validate",
            candidate_id,
            str(holdout),
            "-c",
            str(project / "loopeval.yaml"),
        ],
    )
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.output)["passed_policy"] is True

    # Validation evidence is tied to the decision configuration that produced it.
    config_file = project / "loopeval.yaml"
    saved_config = config_file.read_text()
    changed_config = yaml.safe_load(saved_config)
    changed_config["providers"]["decision"]["model"] = "another-version"
    config_file.write_text(yaml.safe_dump(changed_config))
    stale = runner.invoke(app, ["promote", candidate_id, "-c", str(config_file)])
    assert stale.exit_code == 2, stale.output
    assert "before promotion" in stale.output
    config_file.write_text(saved_config)

    outside = runner.invoke(
        app,
        [
            "promote",
            candidate_id,
            "-c",
            str(config_file),
            "--destination",
            str(tmp_path / "not-loaded"),
        ],
    )
    assert outside.exit_code == 2, outside.output
    assert "promotion destination" in outside.output

    promoted = runner.invoke(
        app,
        [
            "promote",
            candidate_id,
            "-c",
            str(project / "loopeval.yaml"),
        ],
    )
    assert promoted.exit_code == 0, promoted.output
    assert (project / "checks" / "learned" / "learned.novel_pattern.yaml").exists()

    after = runner.invoke(
        app,
        ["run", str(project / "samples.jsonl"), "-c", str(project / "loopeval.yaml")],
    )
    assert after.exit_code == 0, after.output
    after_summary = json.loads(after.output)
    assert after_summary["escalated"] == 0
    assert after_summary["verdicts"] == {"pass": 1, "fail": 1}

    reported = runner.invoke(app, ["report", before_run_id, "-c", str(project / "loopeval.yaml")])
    assert reported.exit_code == 0, reported.output
    assert json.loads(reported.output)["run_id"] == before_run_id

    compared = runner.invoke(
        app,
        [
            "compare",
            before_run_id,
            after_summary["run_id"],
            "-c",
            str(project / "loopeval.yaml"),
        ],
    )
    assert compared.exit_code == 0, compared.output
    assert json.loads(compared.output)["escalation_rate_change"] == -0.5


def test_offline_bootstrap_creates_inactive_candidate(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    assert runner.invoke(app, ["init", str(project), "--offline"]).exit_code == 0
    requirements = project / "requirements.md"
    requirements.write_text("Answers must comply with the supplied product policy.\n")
    result = runner.invoke(
        app,
        [
            "bootstrap",
            "--scenarios",
            str(project / "samples.jsonl"),
            "--requirements",
            str(requirements),
            "-c",
            str(project / "loopeval.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "quality.requirement_violation" in result.output
    assert "not active" in result.output
    listed = runner.invoke(app, ["candidates", "-c", str(project / "loopeval.yaml")])
    assert "quality.requirement_violation" in listed.output

    with LoopEval.from_config(project / "loopeval.yaml") as loop:
        candidate_id = loop.store.list_candidates()[0]["id"]
    exported = project / "review" / "candidate.yaml"
    shown = runner.invoke(
        app,
        [
            "candidate",
            candidate_id,
            "--export",
            str(exported),
            "-c",
            str(project / "loopeval.yaml"),
        ],
    )
    assert shown.exit_code == 0, shown.output
    edited = yaml.safe_load(exported.read_text())
    edited["name"] = "Revised requirement violation"
    exported.write_text(yaml.safe_dump(edited, sort_keys=False))
    revised = runner.invoke(
        app,
        [
            "revise",
            candidate_id,
            str(exported),
            "--notes",
            "Clarified during review.",
            "-c",
            str(project / "loopeval.yaml"),
        ],
    )
    assert revised.exit_code == 0, revised.output
    assert "review and validation are required again" in revised.output


def test_run_ci_gate_junit_and_detailed_report(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    assert runner.invoke(app, ["init", str(project), "--offline"]).exit_code == 0
    junit = project / "reports" / "loopeval.xml"
    output = project / "results" / "run.json"
    run = runner.invoke(
        app,
        [
            "run",
            str(project / "samples.jsonl"),
            "-c",
            str(project / "loopeval.yaml"),
            "--fail-on",
            "fail,unresolved",
            "--junit",
            str(junit),
            "--output",
            str(output),
        ],
    )
    assert run.exit_code == 3, run.output
    assert "Evaluation gate failed" in run.output
    assert junit.exists()
    assert output.exists()
    summary = json.loads(run.stdout)
    assert summary["failed_samples"][0]["sample_id"] == "offline-novel-example"

    details = runner.invoke(
        app,
        [
            "report",
            summary["run_id"],
            "-c",
            str(project / "loopeval.yaml"),
            "--details",
        ],
    )
    assert details.exit_code == 0, details.output
    assert len(json.loads(details.output)["results"]) == 2

    invalid = runner.invoke(
        app,
        [
            "run",
            str(project / "samples.jsonl"),
            "-c",
            str(project / "loopeval.yaml"),
            "--fail-on",
            "pass",
        ],
    )
    assert invalid.exit_code != 0
    assert "accepts only" in invalid.output


def test_calibrate_noul_check_from_labeled_samples(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    assert runner.invoke(app, ["init", str(project), "--offline"]).exit_code == 0
    dataset = project / "calibration.jsonl"
    rows = [
        {
            "id": f"positive-{index}",
            "input": f"positive scenario {index}",
            "output": f"[FAIL] {index}",
            "labels": ["quality.irrelevant"],
        }
        for index in range(5)
    ] + [
        {
            "id": f"negative-{index}",
            "input": f"negative scenario {index}",
            "output": f"healthy {index}",
            "labels": [],
        }
        for index in range(5)
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = runner.invoke(
        app,
        [
            "calibrate",
            "quality.irrelevant",
            str(dataset),
            "-c",
            str(project / "loopeval.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    recommendation = json.loads(result.output)
    assert recommendation["meets_requirements"] is True
    assert recommendation["metrics"]["precision"] == 1


def test_init_refuses_overwrite_and_version(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    assert runner.invoke(app, ["init", str(project), "--offline"]).exit_code == 0
    second = runner.invoke(app, ["init", str(project), "--offline"])
    assert second.exit_code != 0
    assert "refusing to overwrite" in second.output
    version = runner.invoke(app, ["version"])
    assert version.exit_code == 0
    assert version.output.strip() == "0.1.0"


def test_init_selects_direct_and_openrouter_provider_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    direct = tmp_path / "direct"
    initialized = runner.invoke(app, ["init", str(direct)])
    assert initialized.exit_code == 0, initialized.output
    direct_config = yaml.safe_load((direct / "loopeval.yaml").read_text())
    assert direct_config["providers"]["decision"]["type"] == "typesafe"
    assert direct_config["providers"]["decision"]["api_key_env"] == "TYPESAFE_API_KEY"
    assert direct_config["providers"]["fallback"]["type"] == "openai"
    assert direct_config["providers"]["fallback"]["api_key_env"] == "OPENAI_API_KEY"

    doctor = runner.invoke(app, ["doctor", "-c", str(direct / "loopeval.yaml")])
    assert doctor.exit_code == 2
    assert "Missing required credentials" in doctor.output

    routed = tmp_path / "routed"
    initialized = runner.invoke(
        app,
        ["init", str(routed), "--decision", "openrouter", "--fallback", "none"],
    )
    assert initialized.exit_code == 0, initialized.output
    routed_config = yaml.safe_load((routed / "loopeval.yaml").read_text())
    assert routed_config["providers"]["decision"]["type"] == "openrouter_decisions"
    assert routed_config["providers"]["fallback"]["type"] == "disabled"


def test_init_openai_compatible_and_invalid_provider(tmp_path: Path) -> None:
    compatible = tmp_path / "compatible"
    initialized = runner.invoke(
        app,
        [
            "init",
            str(compatible),
            "--fallback",
            "openai-compatible",
            "--fallback-model",
            "local-model",
            "--fallback-base-url",
            "http://localhost:11434/v1",
        ],
    )
    assert initialized.exit_code == 0, initialized.output
    config = yaml.safe_load((compatible / "loopeval.yaml").read_text())
    assert config["providers"]["fallback"]["type"] == "openai_compatible"
    assert config["providers"]["fallback"]["api_key_env"] is None

    invalid = runner.invoke(
        app,
        ["init", str(tmp_path / "bad"), "--decision", "unknown"],
        color=False,
    )
    assert invalid.exit_code != 0
    plain_output = re.sub(r"\x1b\[[0-9;]*m", "", invalid.output)
    assert "--decision must be one of" in plain_output


def test_init_selects_anthropic_and_openai_responses_fallbacks(tmp_path: Path) -> None:
    anthropic = tmp_path / "anthropic"
    result = runner.invoke(app, ["init", str(anthropic), "--fallback", "anthropic"])
    assert result.exit_code == 0, result.output
    config = yaml.safe_load((anthropic / "loopeval.yaml").read_text())
    assert config["providers"]["fallback"] == {
        "type": "anthropic",
        "model": "claude-haiku-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "timeout_seconds": 45,
    }

    responses = tmp_path / "responses"
    result = runner.invoke(app, ["init", str(responses), "--fallback", "openai-responses"])
    assert result.exit_code == 0, result.output
    config = yaml.safe_load((responses / "loopeval.yaml").read_text())
    assert config["providers"]["fallback"]["type"] == "openai_responses"


def test_python_api_and_active_loop_error(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    runner.invoke(app, ["init", str(project), "--offline"])
    with LoopEval.from_config(project / "loopeval.yaml") as loop:
        report = loop.run([{"id": "one", "input": "x", "output": "ok"}])
    assert report.sample_count == 1


def test_python_api_accepts_custom_provider_injection(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    runner.invoke(app, ["init", str(project), "--offline"])
    custom = MockDecisionProvider(ProviderConfig(type="mock", model="injected"))
    with LoopEval.from_config(
        project / "loopeval.yaml", decision_provider=custom, fallback_provider=None
    ) as loop:
        assert loop.evaluator.decision_provider is custom
        assert loop.evaluator.fallback_provider is None


def test_init_creates_requirements_and_preserves_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("private-data/")
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".gitignore").read_text() == "private-data/\n.loopeval/\n"
    assert (tmp_path / "requirements.md").exists()
    assert "[NOVEL]" not in (tmp_path / "samples.jsonl").read_text()
    assert "60 days" in (tmp_path / "samples.jsonl").read_text()
    config = yaml.safe_load((tmp_path / "loopeval.yaml").read_text())
    assert "input_cost_per_million" not in config["providers"]["decision"]


def test_cli_input_errors_are_actionable_without_tracebacks(tmp_path: Path) -> None:
    config = tmp_path / "loopeval.yaml"
    config.write_text("providers: [broken")
    invalid = runner.invoke(app, ["doctor", "-c", str(config)])
    assert invalid.exit_code == 2
    assert "invalid YAML" in invalid.output
    assert "Traceback" not in invalid.output
    config.write_text("unknown: true\n")
    invalid = runner.invoke(app, ["doctor", "-c", str(config)])
    assert invalid.exit_code == 2
    assert "Extra inputs" in invalid.output
    missing = runner.invoke(app, ["doctor", "-c", str(tmp_path / "missing.yaml")])
    assert missing.exit_code == 2


def test_no_cache_bypasses_existing_cache_and_runs_lists_ids(tmp_path: Path, monkeypatch) -> None:
    runner.invoke(app, ["init", str(tmp_path), "--offline"])
    config = tmp_path / "loopeval.yaml"
    args = ["run", str(tmp_path / "samples.jsonl"), "-c", str(config)]
    calls = []
    original = MockDecisionProvider.decide

    async def counted(self, state, questions):
        calls.append(state)
        return await original(self, state, questions)

    monkeypatch.setattr(MockDecisionProvider, "decide", counted)
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    count = len(calls)
    assert count > 0
    assert runner.invoke(app, args).exit_code == 0
    assert len(calls) == count
    uncached = runner.invoke(app, [*args, "--no-cache"])
    assert uncached.exit_code == 0, uncached.output
    assert len(calls) == 2 * count
    listed = runner.invoke(app, ["runs", "-c", str(config), "--limit", "1"])
    assert listed.exit_code == 0, listed.output
    rows = json.loads(listed.output)
    assert len(rows) == 1
    assert json.loads(uncached.output)["run_id"] == rows[0]["id"]


def test_validation_rejects_source_content_and_labels_before_requests(
    tmp_path: Path, monkeypatch
) -> None:
    runner.invoke(app, ["init", str(tmp_path), "--offline"])
    config = tmp_path / "loopeval.yaml"
    boot = runner.invoke(
        app,
        [
            "bootstrap",
            "-c",
            str(config),
            "--scenarios",
            str(tmp_path / "samples.jsonl"),
            "--requirements",
            str(tmp_path / "requirements.md"),
        ],
    )
    assert boot.exit_code == 0, boot.output
    with LoopEval.from_config(config) as loop:
        candidate_id = loop.store.list_candidates()[0]["id"]
    assert (
        runner.invoke(
            app, ["review", candidate_id, "--decision", "approve", "-c", str(config)]
        ).exit_code
        == 0
    )

    async def forbidden(*args, **kwargs):
        raise AssertionError("Validation must reject this dataset before a provider request")

    monkeypatch.setattr(MockDecisionProvider, "decide", forbidden)
    source = json.loads((tmp_path / "samples.jsonl").read_text().splitlines()[0])
    source["id"] = "renamed-source"
    source["labels"] = []
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text(json.dumps(source) + "\n")
    result = runner.invoke(app, ["validate", candidate_id, str(holdout), "-c", str(config)])
    assert result.exit_code == 2, result.output
    assert "cannot reuse" in result.output
    holdout.write_text('{"input":"new","output":"new"}\n')
    result = runner.invoke(app, ["validate", candidate_id, str(holdout), "-c", str(config)])
    assert result.exit_code == 2, result.output
    assert "labels field" in result.output


def test_calibration_does_not_require_the_fallback_key(tmp_path: Path, monkeypatch) -> None:
    runner.invoke(app, ["init", str(tmp_path), "--offline"])
    config = tmp_path / "loopeval.yaml"
    settings = yaml.safe_load(config.read_text())
    settings["providers"]["fallback"] = {
        "type": "openai",
        "model": "test",
        "api_key_env": "MISSING_FALLBACK_TEST_KEY",
    }
    config.write_text(yaml.safe_dump(settings))
    monkeypatch.delenv("MISSING_FALLBACK_TEST_KEY", raising=False)
    dataset = tmp_path / "labeled.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"input": "bad", "output": "[FAIL]", "labels": ["quality.irrelevant"]},
                {"input": "good", "output": "healthy", "labels": []},
            ]
        )
    )
    result = runner.invoke(
        app, ["calibrate", "quality.irrelevant", str(dataset), "-c", str(config)]
    )
    assert result.exit_code == 0, result.output
