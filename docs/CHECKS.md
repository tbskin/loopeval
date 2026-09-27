# Check reference

Start with [bootstrap and the learning loop](LEARNING_LOOP.md) to generate
semantic proposals from requirements. Manual checks are useful for exact
invariants, customization, and reviewing what a proposal actually tests.

## Exact checks

```yaml
id: output.valid_json
name: Valid JSON output
description: The response must be valid JSON.
kind: deterministic
rule: json_valid
field: output
severity: critical
```

Built-in rules are `not_empty`, `exact_match`, `json_valid`, `regex`, `max_length`,
and `required_fields`. Field paths use dots, for example `data.answer`.
Exact checks run before model calls. A definite exact failure stops evaluation
by default, avoiding unnecessary provider calls.

| Rule | Relevant `params` |
| --- | --- |
| `not_empty` | None; `0` and `false` are present values |
| `exact_match` | `expected_field: expected`, `normalize_whitespace: true`, `case_sensitive: true` |
| `json_valid` | None; accepts a JSON string or a JSON-serializable object/list |
| `regex` | Required `pattern`; optional `ignore_case` and `should_match` (default `true`) |
| `max_length` | Required positive `maximum`; counts characters or collection entries |
| `required_fields` | `required: [name, status]`; checks keys on an object, not nonempty values |

Missing evidence normally produces `skipped`. `not_empty` intentionally fails
on an absent or empty value. Use it when evidence presence is a requirement.
`json_valid` rejects non-JSON values such as NaN and Infinity in serialized text.

## Semantic checks

A check asks one failure-oriented question. A Noul result is the probability
that a defect exists, not a quality score where higher is better.

```yaml
id: grounding.unsupported_claim
version: 1.0.0
name: Unsupported claim
description: A material claim is unsupported by the supplied context.
kind: noul
requires: [input, output, context]
instructions: >-
  Does output make a material factual claim contradicted by or absent from context?
criteria:
  "true": At least one material factual claim lacks support or conflicts with context.
  "false": Every material factual claim is supported by context.
pass_threshold: 0.15
failure_threshold: 0.80
severity: critical
```

Thresholds are illustrative. Calibrate them on your data. Values between the pass
and failure thresholds abstain. Choice checks use named criteria, `failure_labels`,
`uncertain_labels`, and `min_confidence`. Score checks use 2 to 10 ordered criteria
and `failure_score_gte` within that range. See
[Jev primitives](https://docs.typesafe.ai/concepts/system-one) for their meaning.

Only enabled checks with `lifecycle: active` run in the normal cascade. An id
identifies the failure category across labels and reports. Only one active
version of each id may be loaded. Bump `version` when changing a tracked policy.

`requires` defines applicability, not an assertion that fields must exist.
Missing required evidence skips that check; other checks may still pass. Add an
exact presence check for evidence your app must always supply.

The `examples` field is review material. It is not automatically added to Jev's
questions. Put operational boundaries in `instructions` and `criteria`.
