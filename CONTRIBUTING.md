# Contributing

LoopEval welcomes focused issues and pull requests. New integrations should
extend a model-provider protocol, a generic check capability, or the reviewed
learning workflow while preserving the core trust boundaries.

Coding agents modifying this repository should read [AGENTS.md](AGENTS.md)
before making changes.

## Development

```bash
git clone https://github.com/tbskin/loopeval.git
cd loopeval
uv sync --extra dev
uv run ruff check .
uv run mypy src/loopeval
uv run pytest --cov=loopeval --cov-report=term-missing
uv build
```

Live provider tests must be opt-in and must never run on pull requests from
forks. Unit tests should use `httpx.MockTransport` or LoopEval's mock providers.

## Pull requests

- explain the user-visible behavior and trust-boundary impact;
- include tests for success, abstention, and provider failure;
- update schemas and documentation together;
- avoid real credentials, private samples, and generated `.loopeval/` state;
- keep checks atomic and attach evidence to deterministic failures.

Contributions are licensed under Apache-2.0.
