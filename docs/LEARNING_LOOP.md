# Learning and promotion loop

## Initial bootstrap

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

- `existing_failure` — a current check already describes the problem;
- `novel_failure` — a distinct reusable failure needs a new check;
- `acceptable` — the cheap tier abstained but no material failure exists;
- `insufficient_evidence` — no reliable judgment can be made from supplied data.

A novel response must include a complete candidate check. The candidate is
validated as untrusted data and deduplicated by its stable proposed check id and
kind. Wording may improve between sightings without splitting the candidate.
Repeated observations increase `evidence_count` and retain per-sample evidence
rather than creating issue spam.

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

## Held-out validation

Each sample's `labels` list identifies failures present according to human
review. Candidate validation computes:

- true/false positives and negatives;
- precision and recall;
- F1 and resolved accuracy;
- coverage and unresolved count.

The default gate favors precision because false accusations from an evaluator
are costly: 20 examples, 0.90 precision, 0.50 recall, and 0.80 resolved coverage.
Domains with severe false negatives should raise recall requirements and use
severity-weighted gates.

Never tune and report on the same examples. Keep a final test split invisible to
the proposer and threshold-selection process.

## Promotion

Promotion writes a normal active YAML check under `checks/learned/`. The next
run loads it exactly like a hand-authored check and includes it in Jev's batched
questions. That concrete registry mutation is what reduces future fallback.

After promotion:

1. rerun the frozen before-population;
2. compare human agreement and unresolved rate;
3. confirm fallback cost dropped;
4. inspect false positives introduced by the new check;
5. commit the new file so source control supplies rollback history.

Promotion refuses to overwrite a file or activate an id already present in the
loaded registry, even with `--force`.

## Deprecation and drift

v0.1 represents deprecation in the check's `lifecycle`. A later drift monitor
should periodically shadow active checks against new human-reviewed data and
move degraded checks out of active service. Adding checks forever is not
learning; pruning checks that no longer earn their cost is part of the loop.
