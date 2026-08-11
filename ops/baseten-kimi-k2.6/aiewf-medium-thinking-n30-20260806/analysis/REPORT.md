# BaseTen Kimi K2.6 — AIEWF medium-context result

The fixed denominator is 30 canonical conversations × scripted turns 0–29 =
900. Recovery rows are excluded from both scores and README TTFAT, but their
tokens remain in the billed-token totals.

## README-format result

| Measure | Result |
|---|---:|
| Strict turn pass | 885/900 (98.3%) |
| Conversation-cluster bootstrap 95% CI | 97.7–99.0% |
| Any error | 1.7% |
| Tool error | 1.4% |
| Instruction error | 1.7% |
| KB error | 0.0% |
| Scripted-turn TTFAT P50 / P95 / max | 1596 / 5252 / 13622 ms |
| Raw first-chunk TTFB P50 / P95 / max | 666 / 1146 / 2163 ms |
| Reasoning delay P50 / P95 / max | 854 / 4586 / 12100 ms |
| Scripted rows with positive thinking tokens | 900/900 |
| Scripted thinking tokens | 225,391 |

```text
| kimi-k2.6 (thinking on) | 98.3% | 1.7% | 1.4% | 1.7% | 0.0% | 1596ms | 5252ms | 13622ms | BaseTen |
```

## Collection and protocol reliability

| Measure | Result |
|---|---:|
| Canonical complete conversations | 30/30 |
| Conversation attempts recorded | 30 |
| Canonical yield per conversation attempt | 100.0% |
| Canonical on slot's first attempt | 30/30 |
| `end_session` on scripted turn | 30/30 |
| `end_session` on recovery turn | 0/30 |
| No `end_session` | 0/30 |
| Recovery rows excluded from score/TTFAT | 0 |

## Highest-error scripted turns

| Turn | Any strict failures | Tool failures | Instruction failures | KB failures |
|---:|---:|---:|---:|---:|
| 16 | 13/30 | 13/30 | 13/30 | 0/30 |
| 19 | 2/30 | 0/30 | 2/30 | 0/30 |
| 0 | 0/30 | 0/30 | 0/30 | 0/30 |
| 1 | 0/30 | 0/30 | 0/30 | 0/30 |
| 2 | 0/30 | 0/30 | 0/30 | 0/30 |
| 3 | 0/30 | 0/30 | 0/30 | 0/30 |
| 4 | 0/30 | 0/30 | 0/30 | 0/30 |
| 5 | 0/30 | 0/30 | 0/30 | 0/30 |
| 6 | 0/30 | 0/30 | 0/30 | 0/30 |
| 7 | 0/30 | 0/30 | 0/30 | 0/30 |

The 15 strict failures are tightly concentrated. Thirteen runs called
`request_tech_support` on turn 16 before gathering the specific app problem;
two runs used an inappropriate generic event-scope deflection for the Salon 2
directions question on turn 19. No other scripted turn failed.

## Usage accounting

All canonical transcript rows, including recovery, total
12,352,497 prompt tokens,
296,816 completion tokens, and
12,158,240 cache-read input tokens. Recovery
rows alone account for 0 prompt and
0 completion tokens.
