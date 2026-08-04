# Gemma 4 26B A4B +96 dots — full stage

Every conversation contributes 30 scheduled turns. Missing future turns are errors. Arm intervals resample whole conversations; effect intervals resample the frozen temporal pairs.

| arm | conversations | strict pass | whole-conversation 95% CI | strict completion | observed / fixed turns | TTFAT P50 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| no filler | 30 | 80.7% | 79.4 to 81.6% | 29/30 (96.7%) | 894 / 900 | 580 / 803 ms |
| +96 dots | 30 | 79.8% | 76.2 to 82.2% | 29/30 (96.7%) | 882 / 900 | 582 / 812 ms |

Dots minus control strict-pass effect: **-0.9 points** (paired bootstrap 95% CI -4.6 to +1.8).

## Error concentration

| arm | total strict errors | turns affected | top-3 turn share | top-5 turn share |
|---|---:|---:|---:|---:|
| no filler | 174 | 12 | 51.7% | 84.5% |
| +96 dots | 182 | 19 | 48.9% | 77.5% |

### Highest-error turns

| arm | turn | errors | conversations |
|---|---:|---:|---:|
| no filler | 12 | 30 | 100.0% |
| no filler | 21 | 30 | 100.0% |
| no filler | 24 | 30 | 100.0% |
| no filler | 15 | 29 | 96.7% |
| no filler | 17 | 28 | 93.3% |
| no filler | 19 | 18 | 60.0% |
| no filler | 11 | 3 | 10.0% |
| no filler | 25 | 2 | 6.7% |
| no filler | 26 | 1 | 3.3% |
| no filler | 27 | 1 | 3.3% |
| +96 dots | 21 | 30 | 100.0% |
| +96 dots | 24 | 30 | 100.0% |
| +96 dots | 12 | 29 | 96.7% |
| +96 dots | 15 | 27 | 90.0% |
| +96 dots | 17 | 25 | 83.3% |
| +96 dots | 19 | 20 | 66.7% |
| +96 dots | 11 | 8 | 26.7% |
| +96 dots | 13 | 2 | 6.7% |
| +96 dots | 14 | 1 | 3.3% |
| +96 dots | 16 | 1 | 3.3% |

## Sample-size decision

The full 30-pair stage is terminal; no promotion rule applies.
