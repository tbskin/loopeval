# Releasing LoopEval

This is the release procedure, not a statement that a release exists.
LoopEval will publish source distributions and universal Python wheels to PyPI.
GitHub Actions uses PyPI Trusted Publishing, so the repository does not need a
long-lived PyPI API token.

## One-time repository setup

Before the first release, verify real-provider integrations and the learning
loop against independent labeled data. Include an LLM-as-judge baseline, both
at equal trace coverage and equal budget. Offline mocks and a successful package
build are necessary but do not demonstrate judge quality or cost reduction.
Publication requires an explicit maintainer decision after this evidence review.

1. Create the `loopeval` project on PyPI, or configure a pending trusted
   publisher for the first release.
2. In PyPI, add a GitHub trusted publisher with:
   - owner: `tbskin`
   - repository: `loopeval`
   - workflow: `release.yml`
   - environment: `pypi`
3. Create a GitHub environment named `pypi` and require maintainer approval for
   deployments to it.
4. Protect `.github/workflows/release.yml` with normal branch review and, when
   available, CODEOWNERS.
5. Confirm the intended public repository visibility, issue templates, license,
   and an operational private security-reporting channel before announcing it.

A PyPI account or pending publisher does not reserve the project name. Recheck
name availability when preparing the first publication.

The dedicated publish job has only `id-token: write`. Build and test steps run in
a separate job without that permission. The published distributions are the
exact artifacts produced and smoke-tested by the build job.

## Prepare a release

1. Update the version in `pyproject.toml` and `src/loopeval/__init__.py`.
2. Move the relevant entries from `Unreleased` into a dated section in
   `CHANGELOG.md`.
3. Run the full checks:

   ```bash
   uv sync --extra dev --locked
   uv run ruff check .
   uv run mypy src/loopeval
   uv run pytest --cov=loopeval --cov-report=term-missing
   ./scripts/verify-package.sh dist
   ```

4. Inspect the wheel and source archive under `dist/`.
5. Commit the version and changelog with a simple release commit.
6. Create and push an annotated tag matching the package version exactly, such
   as `v0.1.0`.
7. Draft a GitHub release from that tag. Review the notes, then publish it.
8. Approve the `pypi` environment deployment after the build job passes.

The workflow refuses to publish when the GitHub release tag does not equal `v`
plus the version in `pyproject.toml`. PyPI also rejects attempts to overwrite an
existing version.

## Verify the public release

Install into a clean environment and run the offline tour:

```bash
uvx --from loopeval==0.1.0 loopeval version

mkdir loopeval-release-check
cd loopeval-release-check
uvx --from loopeval==0.1.0 loopeval init demo --offline
uvx --from loopeval==0.1.0 loopeval doctor --live --config demo/loopeval.yaml
uvx --from loopeval==0.1.0 loopeval run demo/samples.jsonl --config demo/loopeval.yaml
```

Replace `0.1.0` with the released version. Do not delete or replace a bad PyPI
release. Publish a new patch version with the fix.
