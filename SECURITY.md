# Security policy

## Supported versions

LoopEval is unreleased. Security fixes currently target the default branch.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
security-advisory [reporting form](https://github.com/tbskin/loopeval/security/advisories/new)
when available. If private reporting is unavailable, open an issue requesting a
private contact method without disclosing the vulnerability. Include reproduction steps,
impact, affected versions, and any suggested mitigation. Maintainers should
acknowledge a report within five business days.

## Threat model

Evaluation samples are untrusted. Inputs, outputs, retrieved context, and tool
results may contain prompt injection intended to manipulate the fallback judge.
LoopEval:

- labels the fallback payload as untrusted data;
- uses an unpredictable delimiter and neutralizes delimiter occurrences;
- bounds state size;
- requests schema-constrained output where supported;
- validates every provider response locally;
- never executes provider-proposed code;
- requires separate review and validation for normal promotion; explicit
  `--force` bypasses those gates.

These controls reduce risk but do not make model judges authoritative. Keep
deterministic authorization and irreversible-action policy in application code.

## Secrets

Configuration contains environment-variable names, not key values. LoopEval
does not log authorization headers or persist credentials. Users are responsible
for provider data-retention policies and for redacting sensitive sample content
before transmission.
Do not put secrets in custom `headers`, plugin `options`, samples, or check
definitions. Those fields are configuration or evaluation data, not secret
storage. Provider error bodies are not copied into HTTP error messages. Custom
provider plugins are trusted executable code and must apply their own redaction.

## Local files

`.loopeval/` contains verdicts, model-generated evidence, and check results tied
to sample ids. It does not copy the source dataset into run reports, but quoted
evidence may still contain sensitive text. Protect it and do not commit it.
SQLite uses WAL mode for resilience; WAL and shared-memory files inherit the
directory's OS permissions.
There is no automatic retention or encryption at rest. Apply operating-system
access controls and a retention policy. Bootstrap sends its selected scenarios
to the LLM; runtime semantic checks send sample state to Jev. Labels and expected
verdicts are excluded, but reference answers, metadata, and traces are not.

JUnit reports include per-sample check results, escalation reasons, and fallback
evidence. Apply the same access controls and retention policy used for
`.loopeval/` artifacts.
