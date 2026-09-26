from __future__ import annotations

import json
import runpy
from pathlib import Path

from loopeval.checks import load_checks
from loopeval.config import load_config
from loopeval.io import load_samples


def test_refund_assistant_example_builds_valid_scenarios(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).parents[1] / "examples" / "refund_assistant"
    monkeypatch.chdir(root)
    monkeypatch.syspath_prepend(str(root))
    namespace = runpy.run_path(str(root / "build_scenarios.py"))
    samples = namespace["build_samples"]()
    assert len(samples) == 2
    assert samples[0]["output"]
    assert samples[0]["context"]
    assert samples[0]["trace"]

    generated = tmp_path / "scenarios.jsonl"
    generated.write_text("\n".join(json.dumps(sample) for sample in samples))
    assert len(load_samples(generated)) == 2
    config = load_config(root / "loopeval.yaml")
    assert config.providers.decision.type == "typesafe"
    assert len(load_checks(config.checks)) == 2
