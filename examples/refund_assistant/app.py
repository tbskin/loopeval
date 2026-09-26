from __future__ import annotations

from dataclasses import asdict, dataclass

POLICY = "Customers may request a refund within 30 days of purchase."


@dataclass(frozen=True)
class SupportAnswer:
    output: str
    context: list[str]
    trace: list[dict[str, str]]

    def as_eval_fields(self) -> dict[str, object]:
        return asdict(self)


def answer(question: str) -> SupportAnswer:
    """Small stand-in for an application's RAG or agent call."""
    if "refund" in question.casefold():
        output = "You can request a refund within 30 days of purchase."
    else:
        output = "I can help with questions about the refund policy."
    return SupportAnswer(
        output=output,
        context=[POLICY],
        trace=[{"tool": "policy_lookup", "result": POLICY}],
    )
