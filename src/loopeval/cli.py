from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from . import __version__
from .api import LoopEval
from .bootstrap import propose_initial_checks
from .checks import load_checks
from .config import LoopEvalConfig, load_config
from .io import load_samples
from .learning import (
    ingest_candidate,
    meets_promotion_policy,
    promote_candidate,
    validation_metrics,
)
from .models import CandidateCheck, OverallVerdict, ResultStatus
from .providers import build_generative_provider
from .reporting import compare_reports, gate_failures, report_summary, write_junit_report
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
    "providers": {},
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


DECISION_PRESETS: dict[str, dict[str, object]] = {
    "typesafe": {
        "type": "typesafe",
        "model": "jev-1.13.0",
        "api_key_env": "TYPESAFE_API_KEY",
        "timeout_seconds": 20,
        "input_cost_per_million": 0.042,
        "output_cost_per_million": 0,
    },
    "openrouter": {
        "type": "openrouter_decisions",
        "model": "typesafe/jev-1.13",
        "api_key_env": "OPENROUTER_API_KEY",
        "timeout_seconds": 20,
        "input_cost_per_million": 0.042,
        "output_cost_per_million": 0,
    },
    "mock": {"type": "mock", "model": "mock-decision"},
}


FALLBACK_PRESETS: dict[str, dict[str, object]] = {
    "openai": {
        "type": "openai",
        "model": "gpt-4.1-mini",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 45,
    },
    "openrouter": {
        "type": "openrouter",
        "model": "openai/gpt-4.1-mini",
        "api_key_env": "OPENROUTER_API_KEY",
        "timeout_seconds": 45,
    },
    "none": {"type": "disabled"},
    "mock": {"type": "mock", "model": "mock-generative"},
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


def _provider_configuration(
    *,
    decision: str,
    fallback: str,
    decision_model: str | None,
    fallback_model: str | None,
    decision_key_env: str | None,
    fallback_key_env: str | None,
    fallback_base_url: str | None,
) -> dict[str, object]:
    if decision not in DECISION_PRESETS:
        choices = ", ".join(DECISION_PRESETS)
        raise typer.BadParameter(f"--decision must be one of: {choices}")
    if fallback not in {*FALLBACK_PRESETS, "openai-compatible"}:
        choices = ", ".join([*FALLBACK_PRESETS, "openai-compatible"])
        raise typer.BadParameter(f"--fallback must be one of: {choices}")

    decision_config = json.loads(json.dumps(DECISION_PRESETS[decision]))
    if decision_model:
        decision_config["model"] = decision_model
    if decision_key_env:
        decision_config["api_key_env"] = decision_key_env

    if fallback == "openai-compatible":
        if not fallback_model or not fallback_base_url:
            raise typer.BadParameter(
                "--fallback openai-compatible requires --fallback-model and --fallback-base-url"
            )
        fallback_config: dict[str, object] = {
            "type": "openai_compatible",
            "model": fallback_model,
            "base_url": fallback_base_url,
            "api_key_env": fallback_key_env,
            "timeout_seconds": 45,
            "structured_output": False,
        }
    else:
        fallback_config = json.loads(json.dumps(FALLBACK_PRESETS[fallback]))
        if fallback_model and fallback not in {"none", "mock"}:
            fallback_config["model"] = fallback_model
        if fallback_key_env and fallback not in {"none", "mock"}:
            fallback_config["api_key_env"] = fallback_key_env

    return {"decision": decision_config, "fallback": fallback_config}


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Project directory to initialize")] = Path("."),
    offline: Annotated[
        bool, typer.Option("--offline", help="Use deterministic mock providers; no API key needed.")
    ] = False,
    decision: Annotated[
        str,
        typer.Option("--decision", help="Jev route: typesafe or openrouter."),
    ] = "typesafe",
    fallback: Annotated[
        str,
        typer.Option(
            "--fallback",
            help="Fallback route: openai, openrouter, openai-compatible, or none.",
        ),
    ] = "openai",
    decision_model: Annotated[str | None, typer.Option("--decision-model")] = None,
    fallback_model: Annotated[str | None, typer.Option("--fallback-model")] = None,
    decision_key_env: Annotated[str | None, typer.Option("--decision-key-env")] = None,
    fallback_key_env: Annotated[str | None, typer.Option("--fallback-key-env")] = None,
    fallback_base_url: Annotated[str | None, typer.Option("--fallback-base-url")] = None,
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
        config["providers"] = {
            "decision": DECISION_PRESETS["mock"],
            "fallback": FALLBACK_PRESETS["mock"],
        }
        config["escalation"]["random_audit_rate"] = 0
    else:
        config["providers"] = _provider_configuration(
            decision=decision,
            fallback=fallback,
            decision_model=decision_model,
            fallback_model=fallback_model,
            decision_key_env=decision_key_env,
            fallback_key_env=fallback_key_env,
            fallback_base_url=fallback_base_url,
        )
    check_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    check_path.write_text(yaml.safe_dump(CHECKS_TEMPLATE, sort_keys=False))
    sample_path.write_text("".join(json.dumps(sample) + "\n" for sample in SAMPLES_TEMPLATE))
    typer.echo(f"Initialized LoopEval in {path.resolve()}")
    if not offline:
        decision_provider = config["providers"]["decision"]
        fallback_provider = config["providers"]["fallback"]
        typer.echo(f"Jev: {decision_provider['type']} ({decision_provider['api_key_env']})")
        if fallback_provider["type"] == "disabled":
            typer.echo("LLM fallback: disabled")
        else:
            key_env = fallback_provider.get("api_key_env") or "no credential"
            typer.echo(f"LLM fallback: {fallback_provider['type']} ({key_env})")
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
    missing: list[str] = []
    for role, provider in (
        ("decision", config.providers.decision),
        ("fallback", config.providers.fallback),
    ):
        if provider.type in {"disabled", "mock"}:
            typer.echo(f"{role.title()} provider: {provider.type}")
        elif provider.type == "openai_compatible" and not provider.api_key_env:
            typer.echo(
                f"{role.title()} provider: {provider.type}/{provider.model} (unauthenticated)"
            )
        elif provider.api_key_env and os.environ.get(provider.api_key_env):
            typer.echo(
                f"{role.title()} provider: {provider.type}/{provider.model} (credential present)"
            )
        else:
            missing.append(provider.api_key_env or f"{role} credential")
            typer.echo(
                f"{role.title()} provider: {provider.type}/{provider.model} "
                f"(missing {provider.api_key_env})"
            )
    typer.echo(f"Storage: {config.storage.directory}")
    if missing:
        typer.echo("Missing required credentials: " + ", ".join(missing), err=True)
        raise typer.Exit(code=2)


@app.command("run")
def run_command(
    dataset: Annotated[Path, typer.Argument(help="JSON or JSONL dataset")],
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    junit: Annotated[Path | None, typer.Option("--junit", help="Write a JUnit XML report.")] = None,
    fail_on: Annotated[
        str,
        typer.Option(
            "--fail-on",
            help="Comma-separated verdicts that fail the command: fail, unresolved.",
        ),
    ] = "",
    max_failures: Annotated[
        int | None, typer.Option("--max-failures", min=0, help="Maximum allowed failed samples.")
    ] = None,
    max_unresolved_rate: Annotated[
        float | None,
        typer.Option(
            "--max-unresolved-rate", min=0, max=1, help="Maximum unresolved fraction."
        ),
    ] = None,
    max_cost_usd: Annotated[
        float | None, typer.Option("--max-cost-usd", min=0, help="Maximum reported run cost.")
    ] = None,
) -> None:
    """Run the full deterministic → decision → fallback cascade."""
    requested_verdicts: set[OverallVerdict] = set()
    for raw in filter(None, (item.strip() for item in fail_on.split(","))):
        try:
            requested_verdicts.add(OverallVerdict(raw))
        except ValueError as exc:
            raise typer.BadParameter(
                "--fail-on accepts only 'fail' and 'unresolved'"
            ) from exc
    if OverallVerdict.PASS in requested_verdicts:
        raise typer.BadParameter("--fail-on accepts only 'fail' and 'unresolved'")
    samples = load_samples(dataset)
    with LoopEval.from_config(config_path) as loop:
        report = loop.run(samples)
        summary = report_summary(report)
        typer.echo(json.dumps(summary, indent=2, default=str))
        if output:
            output.write_text(report.model_dump_json(indent=2) + "\n")
            typer.echo(f"Wrote {output}")
        if junit:
            write_junit_report(report, junit)
            typer.echo(f"Wrote {junit}", err=True)
        failures = gate_failures(
            report,
            fail_on=requested_verdicts,
            max_failures=max_failures,
            max_unresolved_rate=max_unresolved_rate,
            max_cost_usd=max_cost_usd,
        )
        if failures:
            for reason in failures:
                typer.echo(f"Evaluation gate failed: {reason}", err=True)
            raise typer.Exit(code=3)


@app.command()
def bootstrap(
    scenarios: Annotated[
        Path,
        typer.Option("--scenarios", help="Representative JSON or JSONL application scenarios."),
    ],
    requirements: Annotated[
        Path,
        typer.Option("--requirements", help="Product requirements or evaluation policy."),
    ],
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path("loopeval.yaml"),
    max_checks: Annotated[
        int, typer.Option("--max-checks", min=1, max=20, help="Maximum proposed checks.")
    ] = 8,
    max_samples: Annotated[
        int, typer.Option("--max-samples", min=1, help="Maximum scenarios sent to the LLM.")
    ] = 25,
) -> None:
    """Propose an initial, reviewable check library from requirements and scenarios."""
    config = load_config(config_path)
    samples = load_samples(scenarios)
    provider = build_generative_provider(config.providers.fallback)
    if provider is None:
        raise typer.BadParameter("bootstrap requires a configured LLM fallback provider")
    store = _store(config)
    try:
        proposal, keys = asyncio.run(
            propose_initial_checks(
                provider=provider,
                requirements=requirements.read_text(),
                samples=samples[:max_samples],
                active_checks=load_checks(config.checks),
                store=store,
                max_checks=max_checks,
                max_chars=config.budgets.max_state_chars,
            )
        )
        typer.echo(proposal.summary)
        for key, item in zip(keys, proposal.candidates, strict=True):
            typer.echo(f"{key}  {item.check.id}  {item.check.name}")
        typer.echo("Next: inspect with 'loopeval candidates' and 'loopeval candidate <id>'.")
        typer.echo("Candidates are not active until reviewed, validated, and promoted.")
    finally:
        store.close()


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
    details: Annotated[
        bool, typer.Option("--details", help="Print the complete per-sample report.")
    ] = False,
) -> None:
    """Print the durable summary for a completed run."""
    config = load_config(config_path)
    store = _store(config)
    try:
        value = store.get_report(run_id)
        if value is None:
            raise typer.BadParameter(f"run not found or incomplete: {run_id}")
        payload = value.model_dump(mode="json") if details else report_summary(value)
        typer.echo(json.dumps(payload, indent=2, default=str))
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
