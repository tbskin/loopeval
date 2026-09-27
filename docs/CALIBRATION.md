# Measurement, calibration, and labeled metrics

Model probabilities are not universal quality thresholds. Calibrate each check
against examples from the application and domain where it will run.

## Label the dataset

Use `expected_verdict` for the overall expected outcome and `labels` for the
specific failure checks present in a sample:

```json
{
  "id": "refund-incorrect-window",
  "input": "What is the refund window?",
  "output": "Refunds are available for 60 days.",
  "context": ["Refunds are available within 30 days."],
  "expected_verdict": "fail",
  "labels": ["grounding.unsupported_claim"]
}
```

Use `labels: []` for a sample that was reviewed and has none of the tracked
failures. Omit `labels` for an unlabeled sample. This distinction prevents an
unknown label from being treated as a human-confirmed negative.

Labels and expected verdicts are excluded from the state sent to providers.

## Read run metrics

`loopeval run` and `loopeval report` include:

- verdict accuracy and a verdict confusion matrix
- precision, recall, F1, accuracy, and resolved coverage per check
- failed and unresolved sample ids
- escalation rate, tokens, latency, and reported cost

`loopeval compare BEFORE AFTER` reports the quality, escalation, and cost changes
between two stored runs. Compare runs on the same frozen dataset.

Comparison rejects changed sample ids, contents, or labels. Legacy reports
without fingerprints are explicitly unverified. Run with `--no-cache` for
request-cost experiments. Report latency uses wall time; `sample_latency_total_ms`
sums overlapping sample durations and is not batch throughput.

## Compare against an LLM judge

The baseline is the cost of evaluating traces, not generating your app's output.
Use the same captured population, policy, evidence, and independent human labels:

| Approach | Trace selection | Purpose |
| --- | --- | --- |
| LLM judge | Every trace | Equal-coverage cost and quality baseline |
| LLM judge | Random 1%, 5%, and 10% samples | Budget-limited monitoring baseline |
| LoopEval before learning | Every trace | Initial cascade cost and quality |
| LoopEval after promotion | Every trace | Effect of the learned check library |

Sampling rates are experiment settings, not a claim that every team samples.
Repeat sampled baselines with multiple recorded random seeds. Give the LLM judge
the same substantive policy and evidence; do not handicap its prompt. Its output
is a comparator, not human ground truth.

Keep three quantities separate:

- **Trace coverage:** evaluated traces divided by the full population. Uncollected
  traces do not count as evaluated.
- **Resolved coverage:** evaluated examples with a decision rather than an
  abstention. The existing per-check `coverage` metric means this quantity.
- **Failure detection:** human-confirmed defects caught across the full population,
  alongside false positives. Unsampled defects are undetected, not correct negatives.

Report cost per 1,000 population traces, total spend, trace coverage, precision,
recall, unresolved results, and wall-clock throughput. At equal coverage, compare
LoopEval with the all-trace LLM judge. At equal budget, compare how many traces
each can inspect and how many real failures each finds. Do not claim lower total
spend than a 1% LLM sample unless measurements show that too.

Account for steady-state and setup costs: Jev, fallback, audits, bootstrap,
validation, calibration, and retries. Report human review effort separately.
Record model versions, pricing date, dataset/check fingerprints, cache settings,
and confidence intervals. Unknown charges stay unknown. More checks increase
prompt cost, so savings need not improve monotonically.

Freeze a test split before proposing checks. Measure throughput separately from
judge quality and include rate limits, oversized traces, and errors. Small
successful fixtures do not prove full-population production capacity.

The CLI currently compares LoopEval runs. LLM-only baselines, repeated sampling,
population-level trace denominators, and statistical intervals require an
experiment harness; `loopeval compare` does not automatically produce them.

## Calibrate a Noul check

```bash
loopeval calibrate grounding.unsupported_claim labeled.jsonl \
  --minimum-precision 0.90 \
  --minimum-coverage 0.80 \
  --output calibration.json
```

LoopEval evaluates the selected active check without fallback, searches pass and
failure threshold pairs, and recommends the pair with the best labeled result
under the requested precision and coverage constraints. Samples inside the
uncertainty band count as unresolved.

The command does not modify the check. Review the recommendation, update the
check YAML deliberately, and rerun a separate final test split.
Reported metrics are fitted on the calibration set, not a held-out test score.
Every row must have labels and distinct content. Recall includes abstained
positives; resolved accuracy excludes abstentions.

## Avoid leakage

Keep at least three roles separate:

1. Development examples used to describe and bootstrap checks.
2. Calibration examples used to choose thresholds.
3. A final test split used only to report quality.

Do not present calibration-set performance as final evaluator quality. Refresh
the labeled data when application behavior, model versions, user populations, or
policies change.
