from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from . import __version__
from .api import LoopEval
from .checks import load_checks
from .config import LoopEvalConfig, load_config
from .io import load_samples
from .learning import (
    ingest_candidate,
    meets_promotion_policy,
    promote_candidate,
    validation_metrics,
)
from .models import CandidateCheck, ResultStatus
from .reporting import compare_reports, report_summary
from .storage import LocalStore

app = typer.Typer(
    name="loopeval",
    no_args_is_help=True,
    help="Standalone, self-improving evaluations with deterministic checks, Jev, and BYOK LLMs.",
)


CONFIG_TEMPLATE = {
    "version": 1,
    "project": "my-loopeval-project",
    "checks": ["checks"],
    "providers": {
        "decision": {
            "type": "openrouter_decisions",
            "model": "typesafe/jev-1.13",
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout_seconds": 20,
            "input_cost_per_million": 0.042,
            "output_cost_per_million": 0,
        },
        "fallback": {
            "type": "openrouter",
            "model": "openai/gpt-4.1-mini",
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout_seconds": 45,
        },
    },
    "escalation": {
        "on_uncertain": True,
        "on_decision_error": True,
        "on_no_applicable_checks": True,
        "short_circuit_on_deterministic_failure": True,
        "coverage_check": True,
        "coverage_pass_threshold": 0.2,
        "coverage_failure_threshold": 0.65,
        "random_audit_rate": 0.02,
        "fail_closed_without_fallback": False,
    },
    "budgets": {
        "per_sample_seconds": 60,
        "concurrency": 8,
        "max_state_chars": 30000,
    },
    "promotion": {
        "minimum_examples": 20,
        "minimum_precision": 0.9,
        "minimum_recall": 0.5,
        "minimum_coverage": 0.8,
        "require_human_approval": True,
    },
    "storage": {"directory": ".loopeval", "database": "loopeval.db", "cache": True},
}


OFFLINE_PROVIDERS = {
    "decision": {"type": "mock", "model": "mock-decision"},
    "fallback": {"type": "mock", "model": "mock-generative"},
}


CHECKS_TEMPLATE = {
    "checks": [
        {
            "id": "output.not_empty",
            "version": "1.0.0",
            "name": "Output is present",
            "description": "The system returned a non-empty output.",
            "kind": "deterministic",
            "source": "builtin",
            "severity": "critical",
            "rule": "not_empty",
            "field": "output",
        },
        {
            "id": "quality.irrelevant",
            "version": "1.0.0",
            "name": "Irrelevant response",
            "description": "The output fails to address the user's request.",
            "kind": "noul",
            "source": "builtin",
            "severity": "error",
            "instructions": "Does `output` fail to directly address the request in `input`?",
            "criteria": {
                "true": "The output ignores the request, answers a different question, or is non-responsive.",
                "false": "The output directly and materially addresses the request.",
            },
            "pass_threshold": 0.2,
            "failure_threshold": 0.8,
        },
        {
            "id": "grounding.unsupported_claim",
            "version": "1.0.0",
            "name": "Unsupported claim",
            "description": "A concrete claim in the output is unsupported by the supplied context.",
            "kind": "noul",
            "source": "builtin",
            "severity": "critical",
            "requires": ["context"],
            "instructions": "Does `output` make a material factual claim that is contradicted by, or absent from, `context`?",
            "criteria": {
                "true": "At least one material factual claim cannot be traced to the supplied context or conflicts with it.",
                "false": "Every material factual claim is supported by the supplied context.",
            },
            "pass_threshold": 0.15,
            "failure_threshold": 0.8,
        },
    ]
}


SAMPLES_TEMPLATE = [
    {
        "id": "healthy-example",
        "input": "What is the refund window?",
        "output": "The refund window is 30 days.",
        "context": ["Customers may request a refund within 30 days of purchase."],
    },
    {
        "id": "offline-novel-example",
        "input": "Demonstrate the learning loop.",
        "output": "[NOVEL] This marker makes the offline mock propose a candidate check.",
    },
]


def _store(config: LoopEvalConfig) -> LocalStore:
    return LocalStore(config.storage.directory, config.storage.database)


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Project directory to initialize")] = Path("."),
    offline: Annotated[
        bool, typer.Option("--offline", help="Use deterministic mock providers; no API key needed.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite scaffold files.")] = False,
) -> None:
    """Create a LoopEval project with checks and a runnable sample dataset."""
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "loopeval.yaml"
    check_path = path / "checks" / "core.yaml"
    sample_path = path / "samples.jsonl"
    existing = [target for target in (config_path, check_path, sample_path) if target.exists()]
    if existing and not force:
        raise typer.BadParameter(
            "refusing to overwrite existing files: " + ", ".join(str(item) for item in existing)
        )
    config = json.loads(json.dumps(CONFIG_TEMPLATE))
    if offline:
        config["providers"] = OFFLINE_PROVIDERS
        config["escalation"]["random_audit_rate"] = 0
    check_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    check_path.write_text(yaml.safe_dump(CHECKS_TEMPLATE, sort_keys=False))
    sample_path.write_text("".join(json.dumps(sample) + "\n" for sample in SAMPLES_TEMPLATE))
    typer.echo(f"Initialized LoopEval in {path.resolve()}")
    typer.echo(f"Next: cd {path} && loopeval doctor && loopeval run samples.jsonl")


@app.command()
def doctor(
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
) -> None:
    """Validate configuration, checks, directories, and credential references."""
    import os

    config = load_config(config_path)
    checks = load_checks(config.checks)
    typer.echo(f"Config: OK ({config.project}, schema v{config.version})")
    typer.echo(f"Checks: OK ({len(checks)} enabled)")
    for role, provider in (
        ("decision", config.providers.decision),
        ("fallback", config.providers.fallback),
    ):
        if provider.type in {"disabled", "mock"}:
            typer.echo(f"{role.title()} provider: {provider.type}")
        elif provider.api_key_env and os.environ.get(provider.api_key_env):
            typer.echo(
                f"{role.title()} provider: {provider.type}/{provider.model} (credential present)"
            )
        else:
            typer.echo(
                f"{role.title()} provider: {provider.type}/{provider.model} "
                f"(missing {provider.api_key_env})"
            )
    typer.echo(f"Storage: {config.storage.directory}")


@app.command("run")
def run_command(
    dataset: Annotated[Path, typer.Argument(help="JSON or JSONL dataset")],
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Run the full deterministic → decision → fallback cascade."""
    samples = load_samples(dataset)
    with LoopEval.from_config(config_path) as loop:
        report = loop.run(samples)
        summary = report_summary(report)
        typer.echo(json.dumps(summary, indent=2, default=str))
        if output:
            output.write_text(report.model_dump_json(indent=2) + "\n")
            typer.echo(f"Wrote {output}")


@app.command()
def learn(
    run_id: Annotated[str, typer.Argument(help="Completed run id")],
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
) -> None:
    """Extract and deduplicate proposed checks from a completed run."""
    config = load_config(config_path)
    store = _store(config)
    try:
        report = store.get_report(run_id)
        if report is None:
            raise typer.BadParameter(f"run not found or incomplete: {run_id}")
        keys = [key for result in report.results if (key := ingest_candidate(store, result))]
        typer.echo(f"Learned {len(set(keys))} candidate(s) from {run_id}")
    finally:
        store.close()


@app.command()
def candidates(
    status: Annotated[str | None, typer.Option("--status")] = None,
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
) -> None:
    """List proposed, reviewed, shadow, and active learned checks."""
    config = load_config(config_path)
    store = _store(config)
    try:
        rows = store.list_candidates(status)
        if not rows:
            typer.echo("No candidates.")
            return
        for row in rows:
            metrics = row.get("validation") or {}
            suffix = (
                f" precision={metrics.get('precision', 0):.3f} recall={metrics.get('recall', 0):.3f}"
                if metrics
                else ""
            )
            typer.echo(
                f"{row['id']}  {row['status']:<9} evidence={row['evidence_count']}  "
                f"{row['check_id']} — {row['title']}{suffix}"
            )
    finally:
        store.close()


@app.command()
def candidate(
    candidate_id: Annotated[str, typer.Argument()],
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
) -> None:
    """Show a candidate check, validation, and the evidence behind it."""
    config = load_config(config_path)
    store = _store(config)
    try:
        row = store.get_candidate(candidate_id)
        if row is None:
            raise typer.BadParameter(f"candidate not found: {candidate_id}")
        row["evidence"] = store.list_candidate_evidence(candidate_id)
        typer.echo(json.dumps(row, indent=2, default=str))
    finally:
        store.close()


@app.command()
def review(
    candidate_id: Annotated[str, typer.Argument()],
    decision: Annotated[str, typer.Option("--decision", help="approve or reject")],
    notes: Annotated[str | None, typer.Option("--notes")] = None,
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
) -> None:
    """Record a human approval or rejection for a proposed check."""
    config = load_config(config_path)
    store = _store(config)
    try:
        store.review_candidate(candidate_id, decision, notes)
        typer.echo(f"{candidate_id}: {decision}")
    finally:
        store.close()


@app.command()
def validate(
    candidate_id: Annotated[str, typer.Argument()],
    dataset: Annotated[Path, typer.Argument(help="Labeled JSON/JSONL dataset")],
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
) -> None:
    """Shadow a candidate on held-out samples and save precision/recall evidence."""
    config = load_config(config_path)
    samples = load_samples(dataset)
    loop = LoopEval(config)
    try:
        row = loop.store.get_candidate(candidate_id)
        if row is None:
            raise typer.BadParameter(f"candidate not found: {candidate_id}")
        if row["status"] not in {"approved", "shadow"}:
            raise typer.BadParameter("candidate must be approved before validation")
        candidate = CandidateCheck.model_validate(row["check"])
        check = candidate.to_spec()

        async def evaluate_all() -> list[bool | None]:
            predictions: list[bool | None] = []
            for sample in samples:
                result = await loop.evaluator.evaluate_sample(
                    sample, checks=[check], enable_fallback=False, enable_coverage=False
                )
                check_result = result.checks[0]
                if check_result.status == ResultStatus.FAIL:
                    predictions.append(True)
                elif check_result.status == ResultStatus.PASS:
                    predictions.append(False)
                else:
                    predictions.append(None)
            return predictions

        predictions = asyncio.run(evaluate_all())
        labels = [check.id in sample.labels for sample in samples]
        metrics = validation_metrics(predictions, labels)
        passed = meets_promotion_policy(metrics, config.promotion)
        loop.store.save_validation(candidate_id, metrics, passed)
        typer.echo(json.dumps({**metrics, "passed_policy": passed}, indent=2))
    finally:
        loop.close()


@app.command()
def promote(
    candidate_id: Annotated[str, typer.Argument()],
    destination: Annotated[Path, typer.Option("--destination")] = Path("checks/learned"),
    force: Annotated[
        bool,
        typer.Option("--force", help="Bypass review/validation gates; recorded in shell history."),
    ] = False,
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
) -> None:
    """Promote a reviewed, held-out-validated candidate into the active registry."""
    config = load_config(config_path)
    store = _store(config)
    try:
        target = promote_candidate(
            store=store,
            candidate_id=candidate_id,
            destination=destination,
            policy=config.promotion,
            existing_check_ids={check.id for check in load_checks(config.checks)},
            force=force,
        )
        typer.echo(f"Promoted {candidate_id} to {target}")
    finally:
        store.close()


@app.command()
def report(
    run_id: Annotated[str, typer.Argument()],
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
) -> None:
    """Print the durable summary for a completed run."""
    config = load_config(config_path)
    store = _store(config)
    try:
        value = store.get_report(run_id)
        if value is None:
            raise typer.BadParameter(f"run not found or incomplete: {run_id}")
        typer.echo(json.dumps(report_summary(value), indent=2, default=str))
    finally:
        store.close()


@app.command()
def compare(
    before: Annotated[str, typer.Argument()],
    after: Annotated[str, typer.Argument()],
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
) -> None:
    """Compare escalation rate, cost, and verdicts between two runs."""
    config = load_config(config_path)
    store = _store(config)
    try:
        before_report = store.get_report(before)
        after_report = store.get_report(after)
        if before_report is None or after_report is None:
            raise typer.BadParameter("both run ids must refer to completed runs")
        typer.echo(json.dumps(compare_reports(before_report, after_report), indent=2))
    finally:
        store.close()


@app.command()
def version() -> None:
    """Print the installed LoopEval version."""
    typer.echo(__version__)
