# Reproducing the registered reasoning modes

The configuration JSON files are the authoritative settings. Use explicit modes
when running these models outside the campaign as well.

| Model | Registered request settings |
|---|---|
| GPT-6 Luna | `MTE_OPENAI_RESPONSES_REASONING_EFFORT=none` or `low`; Responses API, standard reasoning mode; Priority requested, Fast echoed by the provider |
| Sonnet 5.5 | `MTE_ANTHROPIC_THINKING=between_tools` or `adaptive`, with `MTE_ANTHROPIC_EFFORT=low` |
| Haiku 5.5 | `MTE_ANTHROPIC_THINKING=disabled` or `adaptive`, with `MTE_ANTHROPIC_EFFORT=low` |
| Muse Spark 1.3 | OpenRouter model `meta/muse-spark-1.3`, `MTE_OPENROUTER_REASONING_EFFORT=minimal` or `low`, `MTE_OPENROUTER_PROVIDER=Meta` |
| Muse Spark Open | Baseten model `meta-models/muse-spark-open`, `MTE_BASETEN_REASONING_EFFORT=low` |
| Grok Voice 3.0 | `MTE_XAI_VOICE_REASONING_EFFORT=none`, model `grok-voice-3.0`, grok-realtime pipeline and speech-v2 benchmark |

The registered modes are explicit. After all capture finished and judge inputs
were frozen, the factory's unset Sonnet 5.5 default was repaired: it now uses
`between_tools`, replacing Pipecat 1.12's rejected `disabled` default. Explicit
thinking modes and effort settings keep precedence. This compatibility repair
was not used to generate or reinterpret any campaign capture; original source
bytes remain in the verified runtime snapshot. See POST-CAPTURE-COMPAT.json.
`between_tools` has no display field; adaptive runs use `display=omitted`, with
signed thinking blocks preserved by the patched Anthropic adapter.

For example, with the Anthropic credential configured:

```sh
MTE_ANTHROPIC_THINKING=between_tools MTE_ANTHROPIC_EFFORT=low MTE_ANTHROPIC_MAX_TOKENS=16384 uv run multi-turn-eval run aiwf_medium_context --model claude-sonnet-5-5 --service anthropic
```

Grok's `none` setting is injected into the actual session.update payload after
Pipecat's OpenAI-specific reasoning filter. The direct probe verified the server's
session.updated echo. OpenRouter routing allows only Meta, disables fallback, and
requires parameter support. These settings avoid implicit provider reasoning defaults.

See [protocol](PROTOCOL.md), the [Sonnet configuration](../../../ops/recent-models-20261018/sonnet/configuration.json), and
[smoke audit (local archive)](../README.md) for the rest of the sampler, output-cap, pacing,
credit-retry, and provenance settings. The setup registration script is frozen;
use the checked-in JSON configurations, which include the final registered options.

[Evidence storage and local audit instructions](../README.md).
