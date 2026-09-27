## What changed

Describe the user-visible behavior and the problem it solves.

## Trust and cost impact

Explain changes to routing, false positives/negatives, review gates, data sent
to providers, caching, or cost accounting.

## Verification

- [ ] Tests cover success, abstention, and relevant provider failures.
- [ ] `uv run ruff check .` passes.
- [ ] `uv run mypy src/loopeval` passes.
- [ ] `uv run pytest --cov=loopeval --cov-report=term-missing` passes.
- [ ] `./scripts/verify-package.sh dist` passes when packaging changed.
- [ ] Documentation and schemas changed together.
- [ ] No credentials, private samples, or `.loopeval/` artifacts are included.
