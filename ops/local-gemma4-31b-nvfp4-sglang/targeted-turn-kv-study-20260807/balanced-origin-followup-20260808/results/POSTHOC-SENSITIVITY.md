# Post-hoc sensitivity notes

These checks were run only after the preregistered primary analysis was sealed
and completed. They do not alter its estimands or decision rule.

## Original four-prefix subsets under the new seeds

| Original source subset | Four current history effects (pp) | Current four-prefix mean | Full 150-history mean |
|---|---|---:|---:|
| Local BF16 | 0.0, +43.8, +68.8, -12.5 | +25.0 | +11.1 |
| Local FP8 | -75.0, +62.5, -62.5, +6.2 | -17.2 | +7.1 |

The original local-FP8 sign reversal remains visible on the exact same four
histories with independent seeds, so it is a history-bank sampling issue rather
than a late-seed artifact. In 200,000 deterministic random four-history draws
from the observed 150-history FP8-origin bank, 35.1% had a negative mean and
11.8% were at least as negative as -17.2 points.

## Cohort split

History-cluster bootstrap intervals below are post-hoc and unadjusted.

| Origin and cohort | Histories | BF16 - FP8 | 95% bootstrap interval |
|---|---:|---:|---:|
| BF16 origin, N=30 | 30 | +2.1 pp | -9.0 to +13.3 pp |
| FP8 origin, N=30 | 30 | +2.1 pp | -12.7 to +16.9 pp |
| BF16 origin, N=120 | 120 | +13.4 pp | +6.9 to +19.8 pp |
| FP8 origin, N=120 | 120 | +8.3 pp | +1.4 to +15.4 pp |
| Origin interaction, N=30 | 30+30 | 0.0 pp | -18.3 to +18.3 pp |
| Origin interaction, N=120 | 120+120 | +5.1 pp | -4.5 to +14.5 pp |

Neither cohort supports a sign-reversing history-origin interaction.
