# Usage and sample reference

LoopEval evaluates captured outcomes. Your app or test harness produces them.
A Python app can call the API; any language can export JSONL for the CLI. For
production monitoring, capture every trace you intend to evaluate and submit it
from a background worker or batch job. LoopEval does not install instrumentation,
provide a durable ingestion queue, or guarantee a production throughput target.

## Sample fields

Only `input` is required by the schema. Checks may require additional evidence.

| Field | Purpose |
| --- | --- |
| `id` | Unique identifier within a run; content-derived if omitted |
| `input` | Scenario input or user request |
| `output` | Actual application response or structured result |
| `context` | Retrieved passages, policies, or other evidence |
| `expected` | Reference answer or expected structured value |
| `trace` | List of objects describing messages, tool calls, and results |
| `metadata` | Model version, experiment id, or tags |
| `data` | Additional application-specific state |
| `labels` | Human-reviewed failure check ids, used locally for metrics |
| `expected_verdict` | Human-reviewed overall `pass`, `fail`, or `unresolved` |

All fields except `id`, `labels`, and `expected_verdict` are eligible for provider
transmission. A `sample_id` is added. `expected` is reference evidence, not a
private scoring label. Redact sensitive content in metadata and traces too.

Use `labels: []` for a reviewed negative. Omit labels for unreviewed data. Changing
labels does not change the content-derived sample id or expose them to the model.
Duplicate sample ids and empty datasets are rejected before a run starts.

JSONL files contain one object per line. `.json` files may contain an array of
samples. Supply file paths relative to your current directory; paths inside
`loopeval.yaml` resolve relative to the configuration file.

## Python API

```python
from loopeval import EvalSample, LoopEval

samples = [EvalSample(input="Question", output="Captured app answer")]
with LoopEval.from_config("evals/loopeval.yaml") as evaluator:
    report = evaluator.run(samples)
    for result in report.results:
        print(result.sample_id, result.verdict, result.escalation_reasons)
```

Pass dictionaries or `EvalSample` objects. Inside an existing event loop, use
`await evaluator.arun(samples)` instead of `run`. Close the evaluator with the
context manager or `close()` to release SQLite. Create separate evaluator
instances for independent concurrent runs.

Checks are loaded when the evaluator is constructed. Create a new instance after
promotion or check edits. Provider configuration and check contents participate
in caching; pin model versions when reproducibility matters.

## Commands

Use `--config evals/loopeval.yaml` when running outside `evals/`.

| Command | Purpose |
| --- | --- |
| `init` | Create config, starter checks, samples, and requirements |
| `doctor` | Check local setup; `--live` makes billable provider probes |
| `bootstrap` | Propose initial checks from requirements and scenarios |
| `run` | Evaluate samples; optional CI gates, JSON export, and JUnit |
| `runs` | List recent run ids and status |
| `report` | Read a summary; `--details` includes sample results |
| `compare` | Compare completed runs on the same dataset |
| `calibrate` | Recommend Noul thresholds from labeled examples |
| `candidates` / `candidate` | List proposals or inspect evidence |
| `revise` | Edit an inactive proposal and reset review and validation |
| `review` | Record approval or rejection |
| `validate` | Evaluate an approved candidate without fallback |
| `promote` | Write a validated active check into the loaded registry |
| `learn` | Re-ingest candidate evidence from an existing run |

`run` already discovers candidates. `learn` is a recovery/reprocessing command,
not a required step after every run. Use `loopeval COMMAND --help` for options.

## Troubleshooting

- **Missing credentials:** set the environment variable named by `api_key_env`
  in the same process that runs LoopEval. `.env` files are not loaded automatically.
- **Everything passes offline:** mocks recognize synthetic markers. They cannot
  judge your app's quality. Switch to real providers for that experiment.
- **Unresolved results:** inspect `report RUN_ID --details`. Reasons distinguish
  uncertainty, coverage gaps, provider failure, and disabled/exhausted fallback.
- **Skipped checks:** supply the fields listed in `missing_required_fields` or
  adjust the check's scope. A skipped check is not a passed check.
- **State too large:** reduce state deliberately or increase `budgets.max_state_chars`
  within provider limits. State is rejected, not silently truncated.
- **New check not running:** it must be enabled, active, and in a configured
  check path. Reconstruct a long-lived Python evaluator after promotion.
- **Unknown cost:** configure current input and output prices, or use reported
  provider cost. Partial prices are not presented as a complete bill.

## Local artifacts

`.loopeval/` contains SQLite state, cached results, candidates, reviews, and
JSON/JSONL run reports. There is no automatic retention policy. Close evaluators
before backing up the directory, and protect backups like the source dataset.
Keep generated state and exported reports out of public commits. Check YAML and
sanitized fixtures can be versioned with the app.
