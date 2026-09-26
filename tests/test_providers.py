from __future__ import annotations

from typing import Any, ClassVar

import httpx
import pytest

from loopeval.config import ProviderConfig
from loopeval.models import QuestionKind, TypedQuestion
from loopeval.providers import build_decision_provider, build_generative_provider
from loopeval.providers.base import MissingCredentialError, ProviderError
from loopeval.providers.decision import HTTPDecisionProvider
from loopeval.providers.generative import OpenAICompatibleProvider


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


def response(status: int, payload: dict[str, Any] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://example.test")
    return httpx.Response(status, json=payload or {}, request=request)


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


def test_generative_parse_rejects_bad_content() -> None:
    with pytest.raises(ProviderError, match="non-text"):
        OpenAICompatibleProvider._parse([])
    with pytest.raises(ProviderError, match="no JSON"):
        OpenAICompatibleProvider._parse("not json")
