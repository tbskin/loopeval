# Calibration and labeled metrics

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

## Avoid leakage

Keep at least three roles separate:

1. Development examples used to describe and bootstrap checks.
2. Calibration examples used to choose thresholds.
3. A final test split used only to report quality.

Do not present calibration-set performance as final evaluator quality. Refresh
the labeled data when application behavior, model versions, user populations, or
policies change.
