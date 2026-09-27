from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx
import pytest

from loopeval.config import ProviderConfig
from loopeval.models import CandidateCheck, FallbackVerdict, QuestionKind, TypedQuestion
from loopeval.providers import build_decision_provider, build_generative_provider
from loopeval.providers.anthropic import AnthropicProvider
from loopeval.providers.base import MissingCredentialError, ProviderError
from loopeval.providers.decision import HTTPDecisionProvider
from loopeval.providers.generative import OpenAICompatibleProvider
from loopeval.providers.mock import MockDecisionProvider, MockGenerativeProvider
from loopeval.providers.openai_responses import OpenAIResponsesProvider
from loopeval.providers.usage import parse_usage

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "usage, expected",
    [
        ({"cost": 0, "cost_usd": 12}, 0),
        ({"input_tokens": 10}, None),
        ({"input_tokens": 10, "output_tokens": 2}, 0.000014),
    ],
)
def test_cost_accounting_preserves_zero_and_requires_complete_usage(usage, expected) -> None:
    config = ProviderConfig(type="mock", input_cost_per_million=1, output_cost_per_million=2)
    actual = parse_usage({"usage": usage}, config).cost_usd
    assert actual == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize(
    "usage", [{"cost": -1}, {"cost": float("nan")}, {"input_tokens": -1}, {"output_tokens": True}]
)
def test_invalid_usage_is_not_silently_counted(usage) -> None:
    with pytest.raises(ProviderError):
        parse_usage({"usage": usage}, ProviderConfig(type="mock"))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["openai", "openrouter", "anthropic", "openai_responses"])
async def test_full_candidate_schema_through_each_native_transport(kind, monkeypatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    candidate = CandidateCheck(
        id="quality.bad",
        name="Bad",
        description="Defect",
        kind="noul",
        instructions="Is it bad?",
        criteria={"true": "bad", "false": "good"},
    )
    verdict = FallbackVerdict(
        category="novel_failure",
        evidence="Observed defect",
        confidence=0.9,
        candidate_check=candidate,
    )
    wire = verdict.model_dump(mode="json")
    wire["candidate_check"]["criteria"] = json.dumps(candidate.criteria)
    wire["candidate_check"]["params"] = "{}"
    content = json.dumps(wire)
    if kind == "anthropic":
        payload = {"content": [{"type": "text", "text": content}], "stop_reason": "end_turn"}
    elif kind == "openai_responses":
        payload = {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": content}]}],
        }
    else:
        payload = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    FakeAsyncClient.responses = [response(200, payload)]
    provider = build_generative_provider(
        ProviderConfig(type=kind, model="test", api_key_env="TEST_KEY")
    )
    parsed, _, _ = await provider.generate_structured(
        system="Evaluate",
        user="Sample",
        schema=FallbackVerdict.model_json_schema(),
        schema_name="verdict",
    )
    assert FallbackVerdict.model_validate(parsed) == verdict
    body = FakeAsyncClient.requests[0][2]
    if kind in {"openai", "openai_responses"}:
        assert body["store"] is False


@pytest.mark.asyncio
async def test_provider_errors_never_echo_response_body(monkeypatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.responses = [response(400, {"error": "secret private sample"})]
    provider = OpenAICompatibleProvider(
        ProviderConfig(
            type="openai",
            model="test",
            api_key_env="TEST_KEY",
            max_retries=0,
        )
    )
    with pytest.raises(ProviderError) as caught:
        await provider.generate_structured(
            system="x", user="x", schema=SIMPLE_SCHEMA, schema_name="test"
        )
    assert "400" in str(caught.value)
    assert "secret" not in str(caught.value)
    assert "private sample" not in str(caught.value)


class FakeAsyncClient:
    responses: ClassVar[list[httpx.Response]] = []
    requests: ClassVar[list[tuple[str, dict[str, str], dict[str, Any]]]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any]
    ) -> httpx.Response:
        self.requests.append((url, headers, json))
        return self.responses.pop(0)


def response(
    status: int,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    request = httpx.Request("POST", "https://example.test")
    return httpx.Response(status, json=payload or {}, headers=headers, request=request)


@pytest.fixture(autouse=True)
def clear_fake() -> None:
    FakeAsyncClient.responses = []
    FakeAsyncClient.requests = []


def test_provider_factories_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    with pytest.raises(MissingCredentialError):
        build_decision_provider(
            ProviderConfig(
                type="openrouter_decisions",
                model="typesafe/jev-1.13",
                api_key_env="MISSING_TEST_KEY",
            )
        )
    assert build_generative_provider(ProviderConfig(type="disabled")) is None
    with pytest.raises(ValueError, match="not a decision provider"):
        build_decision_provider(
            ProviderConfig(type="openai", model="gpt-test", api_key_env="MISSING_TEST_KEY")
        )
    with pytest.raises(ValueError, match="not a generative provider"):
        build_generative_provider(
            ProviderConfig(type="typesafe", model="jev-test", api_key_env="MISSING_TEST_KEY")
        )


@pytest.mark.asyncio
async def test_decision_provider_parses_official_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.responses = [
        response(
            200,
            {
                "model": "typesafe/jev-1.13-20260917",
                "answers": {"bad": {"type": "noul", "noul": 0.91}},
                "usage": {"input_tokens": 100, "output_tokens": 2, "cost": 0.0000042},
            },
        )
    ]
    provider = HTTPDecisionProvider(
        ProviderConfig(
            type="openrouter_decisions",
            model="typesafe/jev-1.13",
            api_key_env="TEST_KEY",
        )
    )
    result = await provider.decide(
        {"output": "bad"},
        [TypedQuestion(id="bad", kind=QuestionKind.NOUL, instructions="Is it bad?")],
    )
    assert result.answers["bad"].noul == 0.91
    assert result.usage.cost_usd == 0.0000042
    url, headers, body = FakeAsyncClient.requests[0]
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert headers["Authorization"] == "Bearer secret"
    assert body["questions"]["bad"]["type"] == "noul"


@pytest.mark.asyncio
async def test_decision_provider_errors_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    provider = HTTPDecisionProvider(
        ProviderConfig(
            type="typesafe",
            model="jev-1.13.0",
            api_key_env="TEST_KEY",
            max_retries=0,
        )
    )
    with pytest.raises(ProviderError, match="at least one"):
        await provider.decide({}, [])
    FakeAsyncClient.responses = [response(401, {"error": "unauthorized"})]
    with pytest.raises(ProviderError, match="returned 401"):
        await provider.decide({}, [TypedQuestion(id="x", kind="noul", instructions="Is X?")])
    FakeAsyncClient.responses = [response(200, {"answers": {}, "usage": {}})]
    with pytest.raises(ProviderError, match="omitted answer"):
        await provider.decide({}, [TypedQuestion(id="x", kind="noul", instructions="Is X?")])


@pytest.mark.asyncio
async def test_decision_provider_retries_typesafe_overload_and_honors_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    delays: list[float] = []

    async def capture_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("loopeval.providers.http.asyncio.sleep", capture_sleep)
    FakeAsyncClient.responses = [
        response(529, {"error": "overloaded"}, headers={"Retry-After": "1.5"}),
        response(
            200,
            {"answers": {"x": {"type": "noul", "noul": 0.1}}, "usage": {}},
        ),
    ]
    provider = HTTPDecisionProvider(
        ProviderConfig(
            type="typesafe",
            model="jev-1.13.0",
            api_key_env="TEST_KEY",
            max_retries=1,
        )
    )
    result = await provider.decide({}, [TypedQuestion(id="x", kind="noul", instructions="Is X?")])
    assert result.answers["x"].noul == 0.1
    assert delays == [1.5]


@pytest.mark.asyncio
async def test_generative_provider_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.responses = [
        response(
            200,
            {
                "choices": [{"message": {"content": 'prefix {"category":"acceptable"} suffix'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        )
    ]
    provider = OpenAICompatibleProvider(
        ProviderConfig(
            type="openai_compatible",
            model="local-model",
            api_key_env="TEST_KEY",
            base_url="https://local.test/v1/",
            structured_output=False,
            input_cost_per_million=1,
            output_cost_per_million=2,
        )
    )
    parsed, usage, _ = await provider.generate_structured(
        system="system", user="user", schema={"type": "object"}, schema_name="answer"
    )
    assert parsed == {"category": "acceptable"}
    assert usage.cost_usd == pytest.approx(0.000016)
    assert FakeAsyncClient.requests[0][0] == "https://local.test/v1/chat/completions"


@pytest.mark.asyncio
async def test_openai_compatible_provider_allows_unauthenticated_local_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.responses = [
        response(200, {"choices": [{"message": {"content": '{"ok":true}'}}]})
    ]
    provider = OpenAICompatibleProvider(
        ProviderConfig(
            type="openai_compatible",
            model="local-model",
            base_url="http://localhost:11434/v1",
            structured_output=False,
        )
    )
    parsed, _, _ = await provider.generate_structured(
        system="system", user="user", schema={"type": "object"}, schema_name="answer"
    )
    assert parsed == {"ok": True}
    assert "Authorization" not in FakeAsyncClient.requests[0][1]


def test_generative_parse_rejects_bad_content() -> None:
    with pytest.raises(ProviderError, match="non-text"):
        OpenAICompatibleProvider._parse([])
    with pytest.raises(ProviderError, match="no JSON"):
        OpenAICompatibleProvider._parse("not json")


@pytest.mark.asyncio
async def test_anthropic_provider_uses_native_structured_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "anthropic-secret")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.responses = [
        response(
            200,
            {
                "content": [{"type": "text", "text": '{"ok":true}'}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 4},
            },
        )
    ]
    provider = AnthropicProvider(
        ProviderConfig(
            type="anthropic",
            model="claude-haiku-4-5",
            api_key_env="ANTHROPIC_TEST_KEY",
            input_cost_per_million=1,
            output_cost_per_million=2,
        )
    )
    parsed, usage, _ = await provider.generate_structured(
        system="system",
        user="user",
        schema=SIMPLE_SCHEMA,
        schema_name="answer",
    )
    assert parsed == {"ok": True}
    assert usage.cost_usd == pytest.approx(0.00002)
    url, headers, body = FakeAsyncClient.requests[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "anthropic-secret"
    assert headers["anthropic-version"] == "2023-06-01"
    assert body["output_config"]["format"]["schema"] == SIMPLE_SCHEMA


@pytest.mark.asyncio
async def test_anthropic_provider_rejects_truncated_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "anthropic-secret")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.responses = [
        response(
            200,
            {
                "content": [{"type": "text", "text": "{}"}],
                "stop_reason": "max_tokens",
            },
        )
    ]
    provider = AnthropicProvider(
        ProviderConfig(
            type="anthropic",
            model="claude-haiku-4-5",
            api_key_env="ANTHROPIC_TEST_KEY",
            max_retries=0,
        )
    )
    with pytest.raises(ProviderError, match="max_tokens"):
        await provider.generate_structured(
            system="system",
            user="user",
            schema=SIMPLE_SCHEMA,
            schema_name="answer",
        )


@pytest.mark.asyncio
async def test_openai_responses_provider_uses_native_structured_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_TEST_KEY", "openai-secret")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.responses = [
        response(
            200,
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"ok":true}'}],
                    }
                ],
                "usage": {"input_tokens": 9, "output_tokens": 3},
            },
        )
    ]
    provider = OpenAIResponsesProvider(
        ProviderConfig(
            type="openai_responses",
            model="gpt-4.1-mini",
            api_key_env="OPENAI_TEST_KEY",
        )
    )
    parsed, usage, _ = await provider.generate_structured(
        system="system",
        user="user",
        schema=SIMPLE_SCHEMA,
        schema_name="answer",
    )
    assert parsed == {"ok": True}
    assert usage.input_tokens == 9
    url, headers, body = FakeAsyncClient.requests[0]
    assert url == "https://api.openai.com/v1/responses"
    assert headers["Authorization"] == "Bearer openai-secret"
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "answer",
        "strict": True,
        "schema": SIMPLE_SCHEMA,
    }
    assert body["store"] is False


def test_direct_generative_provider_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    anthropic = build_generative_provider(
        ProviderConfig(type="anthropic", model="claude-haiku-4-5", api_key_env="TEST_KEY")
    )
    responses = build_generative_provider(
        ProviderConfig(type="openai_responses", model="gpt-test", api_key_env="TEST_KEY")
    )
    assert isinstance(anthropic, AnthropicProvider)
    assert isinstance(responses, OpenAIResponsesProvider)


def test_provider_plugins_load_from_role_specific_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEntryPoint:
        def __init__(self, factory: Any) -> None:
            self.factory = factory

        def load(self) -> Any:
            return self.factory

    def fake_entry_points(*, group: str, name: str) -> list[FakeEntryPoint]:
        assert name == "example"
        factories = {
            "loopeval.decision_providers": lambda config: MockDecisionProvider(config),
            "loopeval.generative_providers": lambda config: MockGenerativeProvider(config),
        }
        return [FakeEntryPoint(factories[group])]

    monkeypatch.setattr("loopeval.providers.plugins.entry_points", fake_entry_points)
    config = ProviderConfig(type="plugin", plugin="example", model="plugin-model")
    assert isinstance(build_decision_provider(config), MockDecisionProvider)
    assert isinstance(build_generative_provider(config), MockGenerativeProvider)


def test_provider_plugin_reports_missing_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loopeval.providers.plugins.entry_points", lambda **kwargs: [])
    config = ProviderConfig(type="plugin", plugin="missing")
    with pytest.raises(ProviderError, match="is not installed"):
        build_generative_provider(config)
