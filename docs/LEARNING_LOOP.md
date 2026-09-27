# Learning and promotion loop

## Initial bootstrap

Commands below run from the directory containing `loopeval.yaml`. From your app
root, add `--config evals/loopeval.yaml` and prefix dataset paths with `evals/`.
Live bootstrap and validation are billable.

Users should not need to hand-author the first semantic check library. Bootstrap
uses the configured LLM fallback to turn product requirements and representative
scenarios into a small set of candidate deterministic and semantic checks:

```bash
loopeval bootstrap \
  --requirements requirements.md \
  --scenarios samples.jsonl
```

The scenarios are fenced as untrusted data. Provider output is validated against
the same `CandidateCheck` schema used by ongoing discovery. A bootstrap proposal
cannot overwrite an active check or invent executable deterministic code. It can
select only registered deterministic rules.

Bootstrap candidates enter the normal `proposed` state. They require review and
held-out validation before promotion. Bootstrap is therefore a faster way to
express an initial evaluation policy, not a way for a model to approve its own
policy.

## Why promotion is deliberately slow

An LLM fallback is a teacher, not ground truth. Allowing it to write and approve
its own checks would convert its biases into permanent evaluation policy.
LoopEval separates discovery, human judgment, and empirical validation.

## Discovery

Escalated samples receive exactly one structured category:

- `existing_failure`: a current check already describes the problem;
- `novel_failure`: a distinct reusable failure needs a new check;
- `acceptable`: the cheap tier abstained but no material failure exists;
- `insufficient_evidence`: no reliable judgment can be made from supplied data.

A novel response must include a complete candidate check. The candidate is
validated as untrusted data and deduplicated by its stable proposed check id and
kind. The first valid proposal becomes the stable candidate definition. Repeated
model observations increase `evidence_count` and retain per-sample evidence, but
cannot silently rewrite the definition under review.

## Review

The reviewer should verify:

1. the pattern is material and reusable;
2. it does not fit an active check;
3. the question is atomic and failure-oriented;
4. every judgment can be made from declared required fields;
5. exact computation has not been delegated to a model;
6. criteria describe boundary cases rather than merely renaming labels;
7. positive and hard-negative examples exist.

Approval permits shadow evaluation; it does not activate the check.

```bash
loopeval candidates --status proposed
loopeval candidate CANDIDATE_ID
loopeval review CANDIDATE_ID --decision approve --notes "Reviewed the failure boundary."
```

Use an id printed by `candidates` in place of `CANDIDATE_ID`.

If the proposed boundary or wording needs human refinement, export and revise it:

```bash
loopeval candidate cand_abc123 --export candidate.yaml
# Edit candidate.yaml.
loopeval revise cand_abc123 candidate.yaml --notes "Clarified the failure boundary."
```

A revision keeps accumulated evidence but resets the candidate to `proposed` and
deletes its prior validation result. It must be reviewed and validated again.
The check id and kind cannot change because they define candidate identity.
Active checks cannot be revised through the candidate workflow. Edit and version
their YAML deliberately, revalidate the policy, and use source control for rollback.

## Held-out validation

Each sample's `labels` list identifies failures present according to human
review. Every validation sample must include `labels`; use an empty list for a
reviewed negative and omit the field only on unlabeled data. Candidate validation
computes:

- true/false positives and negatives;
- precision and recall;
- F1 and resolved accuracy;
- coverage and unresolved count.

For a candidate with check id `refund.incorrect_window`, two illustrative rows are:

```jsonl
{"id":"holdout-bad","input":"Can I return this after 60 days?","output":"Yes.","context":"Returns are allowed within 30 days.","labels":["refund.incorrect_window"]}
{"id":"holdout-good","input":"Can I return this after 45 days?","output":"No, the limit is 30 days.","context":"Returns are allowed within 30 days.","labels":[]}
```

Replace the check id and evidence for your candidate. Two rows do not satisfy
the default policy. Collect at least 20 distinct reviewed examples, including
failures, clear negatives, and hard boundary cases. Repeated copies of the same
example are not independent evidence.

```bash
loopeval validate CANDIDATE_ID holdout.jsonl
```

Validation uses the decision provider without LLM fallback. It rejects unlabeled
rows, duplicate ids or identical content, and known proposal-source examples
even when their ids were changed. Bootstrap and discovery retain source ids and
content hashes. These guards cannot detect near-duplicates, manual leakage, or
source content absent from legacy records.

Recall counts abstained positive examples as missed failures. Resolved accuracy
excludes abstentions and must be read alongside resolved coverage. At least one
positive and one negative example are required for promotion.

The default gate favors precision because false accusations from an evaluator
are costly: 20 examples, 0.90 precision, 0.50 recall, and 0.80 resolved coverage.
Domains with severe false negatives should raise recall requirements and use
severity-weighted gates.

Never tune and report on the same examples. Keep a final test split invisible to
the proposer and threshold-selection process.

## Promotion

```bash
loopeval promote CANDIDATE_ID
```

Promotion writes a normal active YAML check under `checks/learned/`. The next
run loads it like any other check and includes semantic checks in Jev's batched
questions. This can reduce future fallback; measure the effect rather than
assuming every new check saves money or improves quality.

After promotion:

1. rerun the frozen before-population;
2. compare human agreement and unresolved rate;
3. compare evaluation spend and fallback use against the LLM-judge baseline;
4. inspect false positives introduced by the new check;
5. commit the new file so source control supplies rollback history.

Promotion refuses to overwrite a file or activate an id already present in the
loaded registry, even with `--force`.
The destination must be within a configured checks directory. Normal promotion
also verifies that the candidate and decision-provider configuration still match
the saved validation evidence. Revalidate after changing them. `--force` bypasses
review and metric gates, not duplicate-id or overwrite protection.

## Deprecation and drift

v0.1 represents deprecation in the check's `lifecycle`. A later drift monitor
should periodically shadow active checks against new human-reviewed data and
move degraded checks out of active service. Adding checks forever is not
learning; pruning checks that no longer earn their cost is part of the loop.
