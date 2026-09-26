# Security policy

## Supported versions

Security fixes are provided for the latest minor release on the default branch.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
security-advisory reporting for this repository. Include reproduction steps,
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
- never activates a proposed check without a separate review and validation step.

These controls reduce risk but do not make model judges authoritative. Keep
deterministic authorization and irreversible-action policy in application code.

## Secrets

Configuration contains environment-variable names, not key values. LoopEval
does not log authorization headers or persist credentials. Users are responsible
for provider data-retention policies and for redacting sensitive sample content
before transmission.

## Local files

`.loopeval/` contains verdicts, model-generated evidence, and check results tied
to sample ids. It does not copy the source dataset into run reports, but quoted
evidence may still contain sensitive text. Protect it and do not commit it.
SQLite uses WAL mode for resilience; WAL and shared-memory files inherit the
directory's OS permissions.
