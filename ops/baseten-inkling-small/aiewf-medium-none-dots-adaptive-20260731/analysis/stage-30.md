# Inkling Small +96 dots — stage 30

| arm | conversations | strict pass | strict completion | observed / fixed turns | TTFAT P50 / P95 |
|---|---:|---:|---:|---:|---:|
| control none | 30 | 75.1% | 17/30 (56.7%) | 723 / 900 | 279 / 828 ms |
| +96 dots | 30 | 76.9% | 10/30 (33.3%) | 795 / 900 | 368 / 999 ms |

Dots minus control strict-pass effect: **+1.8 points** (whole-conversation bootstrap 95% CI -9.0 to +13.1).

Adaptive recommendation: **terminal_at_30**. This analysis did not execute the stage gate.

## Dot-arm error concentrations

| turn | any-error count | rate |
|---:|---:|---:|
| 16 | 30 | 100.0% |
| 14 | 24 | 80.0% |
| 29 | 20 | 66.7% |
| 28 | 19 | 63.3% |
| 10 | 14 | 46.7% |
| 15 | 10 | 33.3% |
| 26 | 10 | 33.3% |
| 20 | 9 | 30.0% |
| 21 | 9 | 30.0% |
| 24 | 9 | 30.0% |

The denominator is 30 scripted turns per conversation. Missing future turns count as errors. Controls were collected earlier, so provider/deployment-time drift remains a limitation.
