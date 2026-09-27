#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

output_directory="${1:-dist}"
verification_directory="$(mktemp -d)"
trap 'rm -rf "$verification_directory"' EXIT

package_version="$(uv run python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"

uv build --out-dir "$output_directory"
uv run --extra dev twine check "$output_directory"/*

wheel_path="$(find "$output_directory" -maxdepth 1 -name "loopeval-${package_version}-*.whl" -print -quit)"
if [[ -z "$wheel_path" ]]; then
  echo "Built wheel for version $package_version was not found." >&2
  exit 1
fi

uv venv "$verification_directory/venv"
uv pip install --python "$verification_directory/venv/bin/python" "$wheel_path"

loopeval_command="$verification_directory/venv/bin/loopeval"
installed_version="$($loopeval_command version)"
if [[ "$installed_version" != "$package_version" ]]; then
  echo "Installed version $installed_version does not match $package_version." >&2
  exit 1
fi

project_directory="$verification_directory/project"
"$loopeval_command" init "$project_directory" --offline
"$loopeval_command" doctor --live --config "$project_directory/loopeval.yaml"
"$loopeval_command" run \
  "$project_directory/samples.jsonl" \
  --config "$project_directory/loopeval.yaml" \
  --output "$verification_directory/report.json"

echo "Verified LoopEval $package_version from $wheel_path"
