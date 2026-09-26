# Repository guide for coding agents

This file is for coding agents modifying LoopEval itself. Product users should
start with `README.md` and `docs/INTEGRATION_PROMPT.md`.

## Read first

Before changing code, read:

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/LEARNING_LOOP.md` for candidate or promotion changes
4. `docs/PROVIDERS.md` for provider changes
5. `SECURITY.md` for changes involving prompts, credentials, storage, or network calls

## Development commands

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy src/loopeval
uv run pytest --cov=loopeval --cov-report=term-missing
uv build
```

## Invariants

- Exact checks run before network calls.
- Missing evidence never becomes an invented pass.
- Semantic checks remain narrow, typed, and failure-oriented.
- Provider output is untrusted and validated locally.
- Provider failures remain visible in results.
- A fallback model cannot activate its own proposed check.
- Promotion requires review and held-out evidence unless the user explicitly forces it.
- Credentials are referenced by environment-variable name and are never persisted.
- Usage from a batched provider request is counted once.
- Public schemas, CLI behavior, tests, and documentation change together.

## Scope and test discipline

Use mock providers or transport fakes in automated tests. Live provider calls
must be explicit and must not run in pull-request CI. Preserve unrelated user
changes, keep generated `.loopeval/` state out of source control, and add tests
for success, abstention, and relevant failure behavior.
