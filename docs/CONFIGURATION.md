# Configuration, caching, and budgets

`loopeval init evals` generates `evals/loopeval.yaml`. See
[provider configuration](PROVIDERS.md) for models and credentials. Unknown
configuration fields are rejected to catch misspelled settings.

## Runtime defaults

```yaml
escalation:
  on_uncertain: true
  on_decision_error: true
  on_no_applicable_checks: true
  short_circuit_on_deterministic_failure: true
  coverage_check: true
  coverage_pass_threshold: 0.2
  coverage_failure_threshold: 0.65
  random_audit_rate: 0.02
  fail_closed_without_fallback: false
budgets:
  per_sample_seconds: 60
  concurrency: 8
  max_state_chars: 30000
storage:
  directory: .loopeval
  database: loopeval.db
  cache: true
```

Coverage adds a Jev question about material failures outside the applicable
check library. It is a heuristic, not a guarantee of completeness. Random audits
send a deterministic subset of sample ids to the LLM even when checks resolve.
Set `random_audit_rate: 0` for controlled comparisons and document that choice.
This audit sample is separate from trace coverage: every submitted trace still
goes through the cascade.

Disabling escalation does not turn an uncertain check into a pass.
`fail_closed_without_fallback: true` maps unresolved outcomes to failure.

The state limit counts serialized characters, not provider tokens. Oversized
state is not silently truncated. The fallback's fenced data also includes check
results and can hit the limit even when the original state fits. Provider
context limits still apply to state plus questions and instructions.

## Limit spend

Optional controls:

```yaml
escalation:
  max_fallbacks_per_run: 20
budgets:
  run_cost_usd: 1.00
```

The call limit caps newly started fallback requests in a run. Cache hits do not
consume it. The dollar ceiling checks recorded spending before starting another
fallback request. It does not stop Jev requests, reserve unknown costs, or cancel
requests already in flight. Retries can also incur charges. Use provider-side
spending limits for hard account controls.

`--max-cost-usd` is a separate post-run CI gate. It fails on unknown cost; it does
not stop requests during evaluation. Bootstrap, doctor, calibration, and candidate
validation are separate operations, not included in an evaluation run's cost.
Include their charges from provider billing in a total-cost study.

## Cache behavior

Cached responses avoid new provider calls and report zero incremental usage.
Keys include provider identity, configuration, sample state, and questions or
fallback context. Pin models: an alias can change without a local config change,
leaving old cached judgments in place.

Use `loopeval run ... --no-cache` to bypass cache reads and writes for that run.
Set `storage.cache: false` to disable it for other evaluation commands, including
validation and calibration. Bootstrap and live doctor probes always call the
selected provider. Cache entries have no automatic expiry.

## Reproducibility

Reports record configuration, dataset, and check fingerprints. Keep check files,
sanitized fixtures, provider/model versions, and configuration in version control.
Fingerprints detect changes but are not a complete snapshot of a provider or its
weights. Store the source revision alongside experiment results.
