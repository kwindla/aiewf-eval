# BaseTen Kimi K2.6 — AIEWF medium-context result

The fixed denominator is 30 canonical conversations × scripted turns 0–29 =
900. Recovery rows are excluded from both scores and README TTFAT, but their
tokens remain in the billed-token totals.

## README-format result

| Measure | Result |
|---|---:|
| Strict turn pass | 845/900 (93.9%) |
| Conversation-cluster bootstrap 95% CI | 92.3–95.3% |
| Any error | 6.1% |
| Tool error | 6.0% |
| Instruction error | 3.9% |
| KB error | 0.0% |
| Scripted-turn TTFAT P50 / P95 / max | 480 / 854 / 4458 ms |

```text
| kimi-k2.6 (thinking off) | 93.9% | 6.1% | 6.0% | 3.9% | 0.0% | 480ms | 854ms | 4458ms | BaseTen |
```

## Collection and protocol reliability

| Measure | Result |
|---|---:|
| Canonical complete conversations | 30/30 |
| Conversation attempts recorded | 41 |
| Canonical yield per conversation attempt | 73.2% |
| Canonical on slot's first attempt | 25/30 |
| `end_session` on scripted turn | 9/30 |
| `end_session` on recovery turn | 16/30 |
| No `end_session` | 5/30 |
| Recovery rows excluded from score/TTFAT | 54 |

## Highest-error scripted turns

| Turn | Any strict failures | Tool failures | Instruction failures | KB failures |
|---:|---:|---:|---:|---:|
| 29 | 21/30 | 21/30 | 1/30 | 0/30 |
| 15 | 18/30 | 18/30 | 18/30 | 0/30 |
| 17 | 8/30 | 8/30 | 8/30 | 0/30 |
| 24 | 7/30 | 7/30 | 7/30 | 0/30 |
| 19 | 1/30 | 0/30 | 1/30 | 0/30 |
| 0 | 0/30 | 0/30 | 0/30 | 0/30 |
| 1 | 0/30 | 0/30 | 0/30 | 0/30 |
| 2 | 0/30 | 0/30 | 0/30 | 0/30 |
| 3 | 0/30 | 0/30 | 0/30 | 0/30 |
| 4 | 0/30 | 0/30 | 0/30 | 0/30 |

## Usage accounting

All canonical transcript rows, including recovery, total
13,102,869 prompt tokens,
68,638 completion tokens, and
12,930,304 cache-read input tokens. Recovery
rows alone account for 778,306 prompt and
1,039 completion tokens.
