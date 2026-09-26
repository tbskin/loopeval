## What changed

Describe the user-visible behavior and why it belongs in LoopEval's standalone
core.

## Trust and cost impact

Explain changes to routing, false positives/negatives, review gates, data sent
to providers, caching, or cost accounting.

## Verification

- [ ] Tests cover success, abstention, and relevant provider failures.
- [ ] `uv run ruff check .` passes.
- [ ] `uv run mypy src/loopeval` passes.
- [ ] `uv run pytest --cov=loopeval --cov-report=term-missing` passes.
- [ ] Documentation and schemas changed together.
- [ ] No credentials, private samples, or `.loopeval/` artifacts are included.
