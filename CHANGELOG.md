# Changelog

All notable changes follow Keep a Changelog and Semantic Versioning.

## [Unreleased]

### Added

- direct Anthropic and OpenAI Responses fallback providers;
- live provider verification through `loopeval doctor --live`;
- HTTP 529 retry handling and bounded `Retry-After` support.

## [0.1.0] - 2026-09-26

### Added

- standalone deterministic → Jev → BYOK LLM evaluation cascade;
- Noul, Choice, and Score check schemas;
- direct TypeSafe and OpenRouter Decisions providers;
- OpenRouter, OpenAI, and OpenAI-compatible fallback provider;
- explicit coverage, uncertainty, provider-error, and random-audit escalation;
- local SQLite provenance and portable JSON/JSONL artifacts;
- candidate discovery, review, held-out validation, and guarded promotion;
- cost, usage, latency, report, and before/after comparison support;
- offline mock workflow, CLI, Python API, tests, and project documentation.
