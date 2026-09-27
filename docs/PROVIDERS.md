# Provider configuration

LoopEval distinguishes decision providers from generative fallback providers.
Both use small, explicit contracts so provider integrations remain easy to
implement and test.

## Choose a provider path

| Jev route | LLM fallback | Keys |
| --- | --- | --- |
| [TypeSafe direct](https://docs.typesafe.ai/introduction/quickstart) | [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat) | `TYPESAFE_API_KEY`, `OPENAI_API_KEY` |
| TypeSafe direct | [OpenAI Responses](https://platform.openai.com/docs/api-reference/responses) | `TYPESAFE_API_KEY`, `OPENAI_API_KEY` |
| TypeSafe direct | [Anthropic Messages](https://platform.claude.com/docs/en/api/messages) | `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY` |
| TypeSafe direct | [OpenRouter](https://openrouter.ai/docs/quickstart) | `TYPESAFE_API_KEY`, `OPENROUTER_API_KEY` |
| [OpenRouter Decisions](https://openrouter.ai/typesafe/jev-1.13/api) | OpenRouter | `OPENROUTER_API_KEY` |
| TypeSafe direct | OpenAI-compatible hosted or local endpoint | Provider-specific or none |
| TypeSafe direct | Disabled | `TYPESAFE_API_KEY` |
| Mock | Mock | None |

Generate the direct TypeSafe and OpenAI configuration:

```bash
loopeval init evals --decision typesafe --fallback openai
```

Use one OpenRouter key for both roles:

```bash
loopeval init evals --decision openrouter --fallback openrouter
```

Use direct Anthropic:

```bash
loopeval init evals --decision typesafe --fallback anthropic
```

Use OpenAI's Responses API instead of Chat Completions:

```bash
loopeval init evals --decision typesafe --fallback openai-responses
```

Use Jev without the generative learning loop:

```bash
loopeval init evals --decision typesafe --fallback none
```

Provider keys are read from the environment variables named in
`loopeval.yaml`. LoopEval does not read or persist the key itself.

Validate the files and credential references locally:

```bash
loopeval doctor
```

Then verify the real endpoints before a run:

```bash
loopeval doctor --live
```

The live command makes one small, billable request to each configured provider.
It catches invalid keys, inaccessible models, incorrect base URLs, and unsupported
structured-output settings.

## TypeSafe Jev

Create a key in the [TypeSafe dashboard](https://console.typesafe.ai/). The
[TypeSafe model reference](https://docs.typesafe.ai/models) lists current model
names, pricing, and limits.

```yaml
providers:
  decision:
    type: typesafe
    model: jev-1.13.0
    api_key_env: TYPESAFE_API_KEY
    timeout_seconds: 20
    max_retries: 2
    input_cost_per_million: 0.042
    output_cost_per_million: 0
```

Endpoint: `POST https://api.typesafe.ai/v1/systemone`.

## Jev through OpenRouter Decisions

Create a key in [OpenRouter settings](https://openrouter.ai/settings/keys).

```yaml
providers:
  decision:
    type: openrouter_decisions
    model: typesafe/jev-1.13
    api_key_env: OPENROUTER_API_KEY
```

Endpoint: `POST https://openrouter.ai/api/alpha/decisions`.

LoopEval does not silently fail over between providers. Hidden fallback changes
data handling, cost, and calibration. Choose the provider explicitly and let a
provider failure become an observable escalation reason.

## OpenRouter fallback

```yaml
providers:
  fallback:
    type: openrouter
    model: anthropic/claude-haiku-4.5
    api_key_env: OPENROUTER_API_KEY
    timeout_seconds: 45
```

## OpenAI Chat Completions fallback

Create a key in the [OpenAI dashboard](https://platform.openai.com/api-keys).

```yaml
providers:
  fallback:
    type: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
```

This route calls `POST https://api.openai.com/v1/chat/completions` and requests
strict JSON Schema output by default.

## OpenAI Responses fallback

Use the native [Responses API](https://platform.openai.com/docs/api-reference/responses):

```yaml
providers:
  fallback:
    type: openai_responses
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
    max_output_tokens: 4096
```

This route calls `POST https://api.openai.com/v1/responses` and uses
`text.format` with a strict JSON Schema.

## Anthropic fallback

Create a key in the [Anthropic Console](https://console.anthropic.com/settings/keys).

```yaml
providers:
  fallback:
    type: anthropic
    model: claude-haiku-4-5
    api_key_env: ANTHROPIC_API_KEY
    max_output_tokens: 4096
```

This route calls `POST https://api.anthropic.com/v1/messages` and uses Anthropic's
native `output_config.format` JSON Schema support. LoopEval rejects responses
that stop because of a refusal or output-token limit instead of treating partial
JSON as a valid verdict.

## Any OpenAI-compatible endpoint

Generate a compatible configuration with:

```bash
loopeval init evals \
  --fallback openai-compatible \
  --fallback-model my-model \
  --fallback-base-url https://inference.example.com/v1 \
  --fallback-key-env MY_MODEL_API_KEY
```

```yaml
providers:
  fallback:
    type: openai_compatible
    model: my-model
    base_url: https://inference.example.com/v1
    api_key_env: MY_MODEL_API_KEY
    structured_output: false
```

Set `structured_output: false` when the endpoint does not implement
`response_format.json_schema`; LoopEval then includes the schema in the system
message and still validates the returned JSON locally.

For a local endpoint that does not require authentication, omit
`api_key_env`. LoopEval will not send an `Authorization` header.

### Google Gemini through its OpenAI-compatible endpoint

[Gemini's OpenAI compatibility layer](https://ai.google.dev/gemini-api/docs/openai)
uses a regular bearer token:

```yaml
providers:
  fallback:
    type: openai_compatible
    model: gemini-3.8-flash
    base_url: https://generativelanguage.googleapis.com/v1beta/openai
    api_key_env: GEMINI_API_KEY
    structured_output: false
```

Set `structured_output: true` only after `loopeval doctor --live` confirms that
the selected model supports `response_format.json_schema` through the
compatibility endpoint.

### Local Ollama

[Ollama exposes an OpenAI-compatible endpoint](https://docs.ollama.com/api/openai-compatibility)
and does not require an API key for its default local server:

```yaml
providers:
  fallback:
    type: openai_compatible
    model: llama3.2
    base_url: http://localhost:11434/v1
    structured_output: false
```

Start Ollama and pull the configured model before running `loopeval doctor
--live`.

## Retries and overloads

Provider calls retry timeouts, transport failures, rate limits, server errors,
and HTTP 529 overload responses. A provider's `Retry-After` header is honored up
to 30 seconds. Configure `max_retries` and `timeout_seconds` per provider. After
the retry limit, LoopEval records the provider error and follows the configured
escalation behavior.

## Model versions

The generated configuration uses convenient current model names. Before
calibrating production thresholds, pin a dated or otherwise immutable model
version when the provider offers one. Changing a model can change probabilities,
accuracy, cost, and escalation rate. Re-run labeled metrics and calibration after
any provider or model change.

## Custom providers

There are two extension paths for an SDK or endpoint that is not
OpenAI-compatible.

### Installable provider plugins

A Python package can register provider factories through standard package entry
points:

```toml
[project.entry-points."loopeval.decision_providers"]
my_decision = "my_package.providers:build_decision_provider"

[project.entry-points."loopeval.generative_providers"]
my_fallback = "my_package.providers:build_generative_provider"
```

Each factory receives the validated `ProviderConfig` and returns a
`DecisionProvider` or `GenerativeProvider`. After installing the package in the
same environment as LoopEval, select it in `loopeval.yaml`:

```yaml
providers:
  decision:
    type: plugin
    plugin: my_decision
    model: my-model-version
    api_key_env: MY_DECISION_API_KEY
    options:
      region: us-east
  fallback:
    type: plugin
    plugin: my_fallback
    model: my-fallback-version
    api_key_env: MY_FALLBACK_API_KEY
```

Plugin-specific settings belong under `options`. Authentication may use the
referenced environment variable or be managed by the plugin. Run `loopeval
doctor --live` to load and verify both plugins. A provider plugin is executable
code, so install only packages you trust.

### Direct Python injection

The Python API also accepts implementations of LoopEval's small provider
contracts directly:

```python
from loopeval import DecisionProvider, LoopEval


class MyDecisionProvider(DecisionProvider):
    name = "my-provider"
    model = "my-decision-model-v1"

    async def decide(self, state, questions):
        # Call your provider and return loopeval.models.DecisionResponse.
        ...


with LoopEval.from_config(
    "loopeval.yaml",
    decision_provider=MyDecisionProvider(),
) as evaluator:
    report = evaluator.run([{"input": "...", "output": "..."}])
```

`GenerativeProvider` is the equivalent contract for structured fallback calls.
Passing `fallback_provider=None` explicitly disables fallback for that instance.
Provider identity (`name` plus pinned `model`) participates in cache keys.

## Mock providers

Mock providers power tests and the offline tour. They make no network calls and
must never be confused with quality measurements. `loopeval init --offline`
selects them explicitly.

## Price configuration

Prices change. LoopEval therefore calculates cost only from values in your
configuration or a provider-reported cost. Update the configuration when prices
change and keep historical config hashes with reports. An absent price yields an
unknown cost rather than an invented zero.

`budgets.run_cost_usd` is an observed-cost runtime ceiling for the fallback tier.
LoopEval checks it before starting a fallback call and stops starting additional
calls once recorded cost reaches the ceiling. It cannot reserve an unknown
request cost in advance, and already concurrent calls may finish above the
ceiling. Use `escalation.max_fallbacks_per_run` for a hard call-count limit and a
provider-side spending limit for a hard account-level financial control.

The CLI's `--max-cost-usd` is different: it is a post-run CI gate over the final
reported cost. It fails when cost is unknown rather than assuming zero.
