# Estimated text-model cost per conversation minute — 2026-08-06

One completed freshness conversation per configuration; conversation minutes are actual user + assistant words at 150 words/minute.

| Model configuration | Provider | Billing basis | Est. sample cost | Est. cost / conversation min |
|---|---|---|---:|---:|
| claude-sonnet-4-6 | Anthropic | tokens | $0.2610 | $0.0194 |
| claude-fable-5 (low) | Anthropic | tokens | $1.1844 | $0.1104 |
| claude-fable-5 (default) | Anthropic | tokens | $1.0517 | $0.0850 |
| glm-5.2 (none) | BaseTen | tokens | $0.1169 | $0.0077 |
| claude-haiku-4-5 | Anthropic | tokens | $0.0957 | $0.0053 |
| gpt-5.1 | OpenAI | tokens | $0.1395 | $0.0063 |
| gpt-5.6-terra (medium) | OpenAI | tokens | $0.1594 | $0.0230 |
| gpt-5.5 (none) | OpenAI | tokens | $0.4793 | $0.0499 |
| qwen3.6-27b (thinking off) | BaseTen | 1x H100, 1 active conversation | $1.3289 | $0.1083 |
| gemini-3.6-flash (minimal) | AI Studio | tokens | $0.3340 | $0.0416 |
| gpt-5.4 (low) | OpenAI | tokens | $0.2103 | $0.0204 |
| gpt-5.6-sol (none) | OpenAI | tokens | $0.3079 | $0.0428 |
| gpt-4.1 | OpenAI | tokens | $0.2812 | $0.0212 |
| gpt-5.4 (none, +96 dots) | OpenAI | tokens | $0.2209 | $0.0205 |
| inkling (none) | BaseTen | tokens | $0.1036 | $0.0117 |
| gpt-4o | OpenAI | tokens | $0.5783 | $0.0510 |
| kimi-k2.6 (thinking off) | BaseTen | tokens | $0.1086 | $0.0090 |
| gemini-3.5-flash (minimal) | AI Studio | tokens | $0.3583 | $0.0315 |
| claude-sonnet-5 | Anthropic | tokens | $0.2507 | $0.0201 |
| qwen3.6-35b-a3b (thinking off, FP8) | BaseTen | 1x H100, 1 active conversation | $1.6849 | $0.1083 |
| gpt-5.6-terra (none) | OpenAI | tokens | $0.1609 | $0.0220 |
| gpt-5.4 (none) | OpenAI | tokens | $0.1912 | $0.0181 |
| gemini-2.5-flash (thinking off) | AI Studio | tokens | $0.0673 | $0.0070 |
| gpt-5.2 | OpenAI | tokens | $0.1408 | $0.0132 |
| gpt-5.6-luna (none) | OpenAI | tokens | $0.0655 | $0.0083 |
| gpt-oss-120b (groq) | Groq | tokens | $0.0607 | $0.0035 |
| poolside/laguna-s-2.1 (thinking off) | OpenRouter | tokens | $0.0088 | $0.0006 |
| gpt-4.1-mini | OpenAI | tokens | $0.1298 | $0.0105 |
| gpt-5-mini | OpenAI | tokens | $0.0242 | $0.0015 |
| gpt-4o-mini | OpenAI | tokens | $0.0348 | $0.0031 |
| qwen3-8b (thinking off, BaseTen) | BaseTen | 1x H100, 1 active conversation | $1.9131 | $0.1083 |
| gemma-4-26b-a4b-it (thinking off) | BaseTen | 1x H100, 1 active conversation | $1.3787 | $0.1083 |
| gemini-3.5-flash-lite (minimal) | AI Studio | tokens | $0.1386 | $0.0142 |

Dedicated BaseTen values assume one active conversation has exclusive use of the listed GPU. They exclude cold-start and idle scale-down overhead; they are utilization estimates, not token prices.

See `COST-METHODOLOGY.md` and `pricing-2026-08-06.json` for cache accounting, exclusions, assumptions, rates, and official sources.
