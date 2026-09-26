from __future__ import annotations

import json
from pathlib import Path

from .models import EvalSample


def load_samples(path: str | Path) -> list[EvalSample]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        samples = []
        for number, line in enumerate(source.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                samples.append(EvalSample.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{source}:{number}: invalid sample: {exc}") from exc
        return samples
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text())
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"{source}: expected a JSON object or array")
        return [EvalSample.model_validate(item) for item in data]
    raise ValueError("datasets must be .jsonl or .json")
