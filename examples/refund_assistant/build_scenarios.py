from __future__ import annotations

import json
from pathlib import Path

from app import answer

SCENARIOS = [
    {
        "id": "refund-window",
        "input": "How long do I have to request a refund?",
        "expected": "The refund window is 30 days.",
        "labels": [],
        "expected_verdict": "pass",
    },
    {
        "id": "unrelated-question",
        "input": "Can you help me change my shipping address?",
        "expected": "The assistant should say this request is outside its refund-policy scope.",
        "labels": [],
        "expected_verdict": "pass",
    },
]


def build_samples() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        response = answer(str(scenario["input"]))
        samples.append({**scenario, **response.as_eval_fields()})
    return samples


if __name__ == "__main__":
    destination = Path(__file__).parent / "scenarios.jsonl"
    destination.write_text("".join(json.dumps(sample) + "\n" for sample in build_samples()))
    print(f"Wrote {destination}")
