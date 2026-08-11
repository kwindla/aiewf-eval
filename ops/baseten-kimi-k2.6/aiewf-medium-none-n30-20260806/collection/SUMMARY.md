# BaseTen Kimi K2.6 collection summary

This immutable summary covers the frozen 30-conversation thinking-off,
no-filler AIEWF medium-context cohort. It describes collection reliability,
not judged model accuracy.

| Measure | Result |
|---|---:|
| Canonical complete conversations | 30/30 |
| Fixed scripted-turn denominator | 900 |
| Conversation attempts recorded | 41 |
| Canonical yield per conversation attempt | 73.2% |
| Canonical on slot's first attempt | 25/30 |
| Slots requiring retries | K26-02, K26-03, K26-08, K26-12, K26-15 |
| `end_session` on scripted turn 0–29 | 9/30 |
| `end_session` on recovery turn 30+ | 16/30 |
| No `end_session` | 5/30 |

## Attempt outcomes

| Outcome | Attempts |
|---|---:|
| canonical_or_eligible_complete | 30 |
| incomplete_scheduled_turns | 2 |
| out_of_cohort_duplicate_complete | 1 |
| out_of_cohort_interrupted | 1 |
| provider_429 | 3 |
| provider_stream_or_502 | 4 |


## Frozen request signature

- Endpoint: `https://inference.baseten.co/v1`
- Model: `moonshotai/Kimi-K2.6`
- Reasoning effort: `none`
- Temperature: 0.6
- Max tokens: 8192
- Filler: none
- Provider concurrency: 1
- Inter-attempt cooldown: 30 seconds

`input_hashes` in `summary.json` pins the exact collection manifests, lifecycle
log, source-integrity manifest, configuration, and this summarizer.
