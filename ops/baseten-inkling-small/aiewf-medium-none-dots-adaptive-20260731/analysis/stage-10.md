# Inkling Small +96 dots — stage 10

| arm | conversations | strict pass | strict completion | observed / fixed turns | TTFAT P50 / P95 |
|---|---:|---:|---:|---:|---:|
| control none | 30 | 75.1% | 17/30 (56.7%) | 723 / 900 | 279 / 828 ms |
| +96 dots | 10 | 75.0% | 4/10 (40.0%) | 258 / 300 | 369 / 962 ms |

Dots minus control strict-pass effect: **-0.1 points** (whole-conversation bootstrap 95% CI -16.3 to +15.0).

Adaptive recommendation: **extend_to_30**. This analysis did not execute the stage gate.

## Dot-arm error concentrations

| turn | any-error count | rate |
|---:|---:|---:|
| 16 | 10 | 100.0% |
| 14 | 9 | 90.0% |
| 15 | 7 | 70.0% |
| 28 | 6 | 60.0% |
| 29 | 6 | 60.0% |
| 10 | 4 | 40.0% |
| 20 | 4 | 40.0% |
| 26 | 4 | 40.0% |
| 25 | 3 | 30.0% |
| 27 | 3 | 30.0% |

The denominator is 30 scripted turns per conversation. Missing future turns count as errors. Controls were collected earlier, so provider/deployment-time drift remains a limitation.
