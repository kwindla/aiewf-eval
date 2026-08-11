# Targeted KV replay analysis

## Paired KV effects

Positive differences favor BF16 KV. The real-prefix bank weights every prefix equally. Because each prefix reuses the same position-keyed seed stream, the primary uncertainty calculation jointly resamples seed clusters across prefixes. The simultaneous interval is Bonferroni-safe across two turns and all three permitted sample-size looks and governs the precision continuation rule.

| Cell | Pairs / seed clusters | BF16 success | FP8 success | Difference (seed-cluster 95% CI) | Two-turn/three-look simultaneous CI | Cluster score p |
|---|---:|---:|---:|---:|---:|---:|
| warm_turn12_golden | 128 / 128 | 99.2% | 53.9% | +45.3 pp (+36.7, +53.9) | (+33.6, +57.0) | 0 |
| warm_turn12_bank | 384 / 32 | 65.4% | 48.7% | +16.7 pp (+10.7, +22.4) | (+8.6, +24.2) | 5.278e-08 |
| warm_turn15_golden | 128 / 128 | 0.8% | 0.8% | +0.0 pp (-2.3, +2.3) | (-3.1, +3.1) | 1 |
| warm_turn15_bank | 384 / 32 | 56.8% | 56.2% | +0.5 pp (-3.4, +4.2) | (-4.9, +5.5) | 0.7923 |

Holm-adjusted seed-cluster-robust primary bank p-values (report as confirmatory only at the final sample size): `warm_turn12_bank`=1.056e-07, `warm_turn15_bank`=0.7923.

## Turn interaction

This secondary contrast is `(BF16−FP8 at turn 12) − (BF16−FP8 at turn 15)` on the percentage-point scale.

| Cache | Seed clusters | Interaction (seed-cluster 95% CI) | Cluster score p |
|---|---:|---:|---:|
| warm | 32 | +16.1 pp (+8.6, +23.7) | 4.534e-05 |

## Teacher-forced canonical tool sequence

Positive log-probability differences favor BF16 KV for the exact expected tool-call suffix. The decision margin compares `<|tool_call>` with each arm's best first-token alternative.

| Cache | Turn | Prefix kind | Matched snapshots | BF16−FP8 mean logp/token | BF16−FP8 first-decision margin |
|---|---:|---|---:|---:|---:|
| warm | 12 | golden_mechanism | 1 | +0.19910 | +8.36454 |
| warm | 12 | real_prefix_bank | 12 | +0.01587 | +1.40994 |
| warm | 15 | golden_mechanism | 1 | -0.06667 | -1.83374 |
| warm | 15 | real_prefix_bank | 12 | -0.01150 | -0.10555 |
