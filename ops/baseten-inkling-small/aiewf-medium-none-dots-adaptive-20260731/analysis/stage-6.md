# Inkling Small +96 dots — stage 6

| arm | conversations | strict pass | strict completion | observed / fixed turns | TTFAT P50 / P95 |
|---|---:|---:|---:|---:|---:|
| control none | 30 | 75.1% | 17/30 (56.7%) | 723 / 900 | 279 / 828 ms |
| +96 dots | 6 | 85.6% | 3/6 (50.0%) | 172 / 180 | 368 / 930 ms |

Dots minus control strict-pass effect: **+10.4 points** (whole-conversation bootstrap 95% CI -1.8 to +22.6).

Adaptive recommendation: **extend_to_10**. This analysis did not execute the stage gate.

## Dot-arm error concentrations

| turn | any-error count | rate |
|---:|---:|---:|
| 16 | 6 | 100.0% |
| 14 | 5 | 83.3% |
| 15 | 3 | 50.0% |
| 28 | 3 | 50.0% |
| 29 | 3 | 50.0% |
| 9 | 1 | 16.7% |
| 10 | 1 | 16.7% |
| 20 | 1 | 16.7% |
| 25 | 1 | 16.7% |
| 26 | 1 | 16.7% |

The denominator is 30 scripted turns per conversation. Missing future turns count as errors. Controls were collected earlier, so provider/deployment-time drift remains a limitation.
