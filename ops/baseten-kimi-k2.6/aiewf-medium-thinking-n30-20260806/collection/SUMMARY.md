# BaseTen Kimi K2.6 collection summary

This immutable summary covers the frozen 30-conversation thinking-on,
no-filler AIEWF medium-context cohort. It describes collection reliability,
not judged model accuracy.

| Measure | Result |
|---|---:|
| Canonical complete conversations | 30/30 |
| Fixed scripted-turn denominator | 900 |
| Conversation attempts recorded | 30 |
| Canonical yield per conversation attempt | 100.0% |
| Canonical on slot's first attempt | 30/30 |
| Slots requiring retries | none |
| `end_session` on scripted turn 0–29 | 30/30 |
| `end_session` on recovery turn 30+ | 0/30 |
| No `end_session` | 0/30 |

## Attempt outcomes

| Outcome | Attempts |
|---|---:|
| canonical_or_eligible_complete | 30 |


## Frozen request signature

- Endpoint: `https://inference.baseten.co/v1`
- Model: `moonshotai/Kimi-K2.6`
- Reasoning effort: `omit`
- Chat-template args: `{"enable_thinking": true}`
- Temperature: 1.0
- Top-p: 0.95
- Max tokens: 8192
- Filler: none
- Provider concurrency: 1
- Inter-attempt cooldown: 30 seconds

`input_hashes` in `summary.json` pins the exact collection manifests, lifecycle
log, source-integrity manifest, configuration, and this summarizer.
