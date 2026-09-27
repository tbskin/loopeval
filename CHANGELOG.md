# Changelog

All notable changes follow Keep a Changelog and Semantic Versioning.

## [Unreleased]

### Added

- direct Anthropic and OpenAI Responses fallback providers;
- live provider verification through `loopeval doctor --live`;
- installable decision and generative provider entry points;
- verified wheel builds and a trusted-publishing release workflow;
- immutable candidate definitions with accumulating evidence;
- explicit candidate export and human revision with review-state reset;
- HTTP 529 retry handling and bounded `Retry-After` support.

- deterministic, Jev, and BYOK LLM evaluation cascade;
- Noul, Choice, and Score check schemas;
- direct TypeSafe and OpenRouter Decisions providers;
- OpenRouter, OpenAI, and OpenAI-compatible fallback provider;
- explicit coverage, uncertainty, provider-error, and random-audit escalation;
- local SQLite provenance and portable JSON/JSONL artifacts;
- candidate discovery, review, held-out validation, and guarded promotion;
- cost, usage, latency, report, and before/after comparison support;
- offline mock workflow, CLI, Python API, tests, and project documentation.

### Improved

- adopter-first setup, sample reference, provider paths, and measurement guidance;
- holdout source tracking, duplicate protection, and validation provenance;
- strict provider schemas, response validation, and private error handling;
- unknown-cost accounting, cache bypass, and dataset-verified comparisons;
- visible missing evidence, oversized state, and unresolved judgments;
- run discovery and actionable CLI input errors.
