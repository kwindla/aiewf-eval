# Gemma 4 26B A4B +96 dots — initial stage

Every conversation contributes 30 scheduled turns. Missing future turns are errors. Arm intervals resample whole conversations; effect intervals resample the frozen temporal pairs.

| arm | conversations | strict pass | whole-conversation 95% CI | strict completion | observed / fixed turns | TTFAT P50 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| no filler | 10 | 80.0% | 76.7 to 82.3% | 9/10 (90.0%) | 294 / 300 | 580 / 715 ms |
| +96 dots | 10 | 81.0% | 79.3 to 83.0% | 10/10 (100.0%) | 300 / 300 | 584 / 804 ms |

Dots minus control strict-pass effect: **+1.0 points** (paired bootstrap 95% CI -2.0 to +4.3).

## Error concentration

| arm | total strict errors | turns affected | top-3 turn share | top-5 turn share |
|---|---:|---:|---:|---:|
| no filler | 60 | 12 | 50.0% | 80.0% |
| +96 dots | 57 | 7 | 52.6% | 84.2% |

### Highest-error turns

| arm | turn | errors | conversations |
|---|---:|---:|---:|
| no filler | 12 | 10 | 100.0% |
| no filler | 21 | 10 | 100.0% |
| no filler | 24 | 10 | 100.0% |
| no filler | 15 | 9 | 90.0% |
| no filler | 17 | 9 | 90.0% |
| no filler | 19 | 5 | 50.0% |
| no filler | 11 | 2 | 20.0% |
| no filler | 25 | 1 | 10.0% |
| no filler | 26 | 1 | 10.0% |
| no filler | 27 | 1 | 10.0% |
| +96 dots | 12 | 10 | 100.0% |
| +96 dots | 15 | 10 | 100.0% |
| +96 dots | 21 | 10 | 100.0% |
| +96 dots | 24 | 10 | 100.0% |
| +96 dots | 17 | 8 | 80.0% |
| +96 dots | 19 | 6 | 60.0% |
| +96 dots | 11 | 3 | 30.0% |

## Sample-size decision

Recommendation: **promote to 30 pairs**.

- `ci_excludes_zero`: did not fire
- `absolute_effect_ge_3_and_aligned_same_turn_recurs_ge_3`: did not fire
- `completion_differs`: fired

The analyzer does not launch collection. A promotion file is written only when a trigger fires and an explicit reviewer is supplied.
