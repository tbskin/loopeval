from __future__ import annotations

import json
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
                    "input": "x",
                    "output": "[NOVEL] pattern" if index < 10 else "healthy",
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

    promoted = runner.invoke(
        app,
        [
            "promote",
            candidate_id,
            "--destination",
            str(project / "checks" / "learned"),
            "-c",
            str(project / "loopeval.yaml"),
        ],
    )
    assert promoted.exit_code == 0, promoted.output

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


def test_init_refuses_overwrite_and_version(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    assert runner.invoke(app, ["init", str(project), "--offline"]).exit_code == 0
    second = runner.invoke(app, ["init", str(project), "--offline"])
    assert second.exit_code != 0
    assert "refusing to overwrite" in second.output
    version = runner.invoke(app, ["version"])
    assert version.exit_code == 0
    assert version.output.strip() == "0.1.0"


def test_init_selects_direct_and_openrouter_provider_paths(
    tmp_path: Path, monkeypatch
) -> None:
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

    invalid = runner.invoke(app, ["init", str(tmp_path / "bad"), "--decision", "unknown"])
    assert invalid.exit_code != 0
    assert "--decision must be one of" in invalid.output


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
