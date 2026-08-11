# Balanced history-origin crossed-prefix follow-up

Date: 2026-08-08

## Bottom line

The earlier four-prefix-per-source sign reversal did not generalize. In a
balanced census of all 150 local-BF16-origin and all 150 local-FP8-origin
turn-12 histories, BF16 inference outperformed FP8 inference within both
history strata. The history-origin interaction was +4.0 percentage points,
with a 95% history-cluster interval of -4.5 to +12.5 points. It failed the
preregistered rule for a practically important compatibility interaction.

| Frozen history origin | BF16 success | FP8 success | BF16 - FP8 | 95% history-cluster interval |
|---|---:|---:|---:|---:|
| Local BF16 | 52.9% | 41.8% | +11.1 pp | +5.4 to +16.8 pp |
| Local FP8 | 52.6% | 45.5% | +7.1 pp | +0.8 to +13.5 pp |
| Origin interaction | — | — | +4.0 pp | -4.5 to +12.5 pp |
| Balanced 50/50 origin mixture | 52.8% | 43.7% | +9.1 pp | +4.9 to +13.4 pp |

The crossed replay therefore supports a direct, task-specific BF16 advantage
at this frozen turn-12 tool decision, averaged over the existing local history
population. It does not support the stronger claim that each inference path is
generally advantaged on histories it generated. It is not evidence of general
"intelligence divergence."

## No quantization-boundary crossing

No KV state crossed between configurations. The wall-clock macro-block order
was FP8, BF16, BF16, FP8 only to balance drift. The combined audit records
three container IDs:

- block 1 used one FP8-only container;
- that container was stopped and removed before blocks 2 and 3 used one
  BF16-only container; and
- the BF16 container was stopped and removed before block 4 used a fresh
  FP8-only container.

Every snapshot was independently rebuilt from the same frozen token IDs in
the active dtype. No cache was serialized, transferred, converted, or reused
across a server boundary. No generated reply from one arm became input to the
other arm. The complete 300-snapshot token-ID gate passed with zero mismatches.

## What changed relative to the 12-prefix replay

The original local source strata contained only four histories each. Those
four local-FP8 histories were unusually favorable to FP8: their original
512-seed estimate was -9.7 points BF16-minus-FP8. Under the new independent
16-seed allocations, the same four histories still average -17.2 points, even
though all 150 FP8-origin histories average +7.1 points. Their individual new
effects are -75.0, +62.5, -62.5, and +6.2 points.

A deterministic 200,000-draw post-hoc resampling check found that 35.1% of
random four-history subsets from the observed FP8-origin bank have a negative
mean, and 11.8% are at least as FP8-favorable as the originally selected four
under the new allocations. The sign reversal was therefore quite plausible
with four heterogeneous histories and should not be treated as a population
interaction.

The broader history-level distributions remain heterogeneous:

| Origin | BF16-favoring histories | FP8-favoring histories | Ties | Median effect |
|---|---:|---:|---:|---:|
| Local BF16 | 84 | 49 | 17 | +6.2 pp |
| Local FP8 | 76 | 62 | 12 | +6.2 pp |

Heterogeneity is real; the claim that it aligns reliably with history origin
is not supported.

## Robustness checks

The preregistered two-stage bootstrap, which also resamples paired seeds within
histories, closely matches the primary history-cluster result:

| Estimand | Two-stage bootstrap 95% interval |
|---|---:|
| BF16-origin effect | +5.0 to +17.2 pp |
| FP8-origin effect | +0.3 to +13.8 pp |
| Origin interaction | -5.0 to +13.1 pp |
| Balanced origin mixture | +4.5 to +13.7 pp |

The mirrored block halves are also stable:

| Arm and origin | First-half success | Second-half success |
|---|---:|---:|
| FP8 on BF16-origin histories | 41.6% | 42.0% |
| FP8 on FP8-origin histories | 46.2% | 44.9% |
| BF16 on BF16-origin histories | 53.4% | 52.4% |
| BF16 on FP8-origin histories | 52.2% | 53.0% |

The N=30/N=120 cohort split is imprecise but does not reveal an origin
reversal. Both N=30 source strata estimate +2.1 points. In the larger N=120
cohorts, the effects are +13.4 points on BF16-origin histories and +8.3 points
on FP8-origin histories; their interaction is +5.1 points with a post-hoc
bootstrap interval of -4.5 to +14.5.

## Error shift

Across the balanced 4,800 rows per inference arm, correct calls increase from
2,096/4,800 (43.7%) under FP8 to 2,533/4,800 (52.8%) under BF16. Redundant
confirmation/questions fall from 1,332/4,800 (27.8%) to 944/4,800 (19.7%).
That 388-case reduction accounts numerically for 88.8% of the 437-call success
gain. The broad-bank effect therefore remains concentrated at the structured
tool-call-versus-prose decision boundary identified in the original study.

## Evidence decision

No additional BF16-only or FP8-only full trajectories are needed to resolve
the suspected history-origin sign reversal; the balanced 300-history crossed
study has answered that question, and it does not show a reliable reversal.

More on-policy conversations would answer a different question: they would
narrow the production-level aggregate difference, currently +0.67 points with
a whole-conversation bootstrap interval of -0.07 to +1.40 points. Replays
cannot replace independent trajectories for that estimand. If the goal is a
roughly +/-0.5-point production interval, the earlier estimate of about 175
additional conversations per arm still applies. That expense is not required
to establish the direct turn-12 tool-use effect found here.

## Provenance

- `PREREGISTRATION.md` freezes the corpus, estimands, seed allocation, block
  order, inference, and decision rule.
- `preregistration.sha256` seals the preregistered sources.
- `results/token-identity-audit.json` proves 300/300 cross-arm token matches.
- Each block has an immutable `.plan.json` and passing `.audit.json` sidecar.
- `results/combined-integrity-audit.json` proves all 9,600 cells, block order,
  and server-instance separation.
- `results/preanalysis.sha256` seals raw data and analysis code before the first
  aggregate analysis.
- `results/analysis.{json,md}` contains the preregistered result.
