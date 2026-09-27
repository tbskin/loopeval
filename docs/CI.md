# CI and exit codes

LoopEval separates evaluation completion from policy enforcement. A run always
stores its report before applying command-line gates, so a failing build still
has evidence that can be inspected or uploaded as an artifact.

## Gate a run

```bash
loopeval run scenarios.jsonl \
  --fail-on fail,unresolved \
  --max-failures 0 \
  --max-unresolved-rate 0.05 \
  --max-cost-usd 1.00 \
  --junit reports/loopeval.xml
```

Available gates are:

- `--fail-on`: fail when any sample has one of the listed verdicts
- `--max-failures`: allow no more than this many failed samples
- `--max-unresolved-rate`: limit unresolved samples as a fraction from 0 to 1
- `--max-cost-usd`: limit reported provider cost for the run

A cost gate fails when provider cost is unknown. Configure model prices or use a
provider that reports authoritative request cost if the build depends on that
gate.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | The command completed and every configured gate passed |
| `1` | An unexpected runtime or provider error prevented command completion |
| `2` | Configuration, arguments, or required credentials are invalid or missing |
| `3` | Evaluation completed and was stored, but an evaluation gate failed |

Provider failures that the evaluation policy can represent normally become
`unresolved` results rather than command crashes. Include `unresolved` in
`--fail-on` when CI must fail closed.

## JUnit reports

`--junit` writes one test case per evaluation sample. A `fail` verdict becomes a
JUnit failure, while `unresolved` becomes a JUnit error. Most CI systems can
display this file alongside ordinary test results.

JUnit output contains check results, fallback evidence, and escalation reasons.
Treat it with the same access controls as `.loopeval/` run artifacts.

## Pull requests and credentials

Do not expose provider keys to untrusted pull-request code. Common patterns are:

1. Run exact offline smoke checks on every pull request.
2. Run Jev and LLM evaluations only on protected branches or trusted workflows.
3. Use environment-scoped secrets and provider-side spending limits.
4. Upload JUnit and `.loopeval/runs/` artifacts only to access-controlled CI
   storage.
