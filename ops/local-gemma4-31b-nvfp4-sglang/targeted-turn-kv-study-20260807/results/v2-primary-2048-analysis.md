# Targeted KV replay analysis

## Paired KV effects

Positive differences favor BF16 KV. The real-prefix bank weights every prefix equally. Because each prefix reuses the same position-keyed seed stream, the primary uncertainty calculation jointly resamples seed clusters across prefixes. The simultaneous interval is Bonferroni-safe across two turns and all three permitted sample-size looks and governs both the precision continuation rule and confirmatory inference. Ordinary fixed-look p-values are intentionally omitted.

| Cell | Pairs / seed clusters | BF16 success | FP8 success | Difference (seed-cluster 95% CI) | Two-turn/three-look simultaneous CI |
|---|---:|---:|---:|---:|---:|
| warm_turn12_golden | 512 / 512 | 99.8% | 51.0% | +48.8 pp (+44.5, +53.1) | (+43.0, +54.7) |
| warm_turn12_bank | 1536 / 128 | 65.2% | 47.2% | +18.0 pp (+14.9, +21.0) | (+13.9, +21.9) |
| warm_turn15_golden | 512 / 512 | 1.0% | 0.4% | +0.6 pp (-0.4, +1.6) | (-0.8, +2.0) |
| warm_turn15_bank | 1536 / 128 | 55.6% | 56.1% | -0.5 pp (-2.5, +1.5) | (-3.2, +2.2) |

## Primary-look integrity and stopping decision

The exact cumulative 2,048-case-per-turn/arm allocation passed: 8,192 rows, 12 bank prefixes per turn, 128 paired seed clusters, warm cache, and the frozen ABBA stage mapping.

| Cell | Estimate | Simultaneous interval | Interval half-width | ±2 pp target met |
|---|---:|---:|---:|:---:|
| warm_turn12_bank | +18.0 pp | (+13.9, +21.9) | 4.04 pp | no |
| warm_turn15_bank | -0.5 pp | (-3.2, +2.2) | 2.70 pp | no |

Stopping decision: **continue**. The next cumulative look is 4,096 cases per turn/arm.

Confirmatory interpretation uses the simultaneous intervals above. Earlier pilot Holm p-values are exploratory and are not reused.

## Turn interaction

This secondary contrast is `(BF16−FP8 at turn 12) − (BF16−FP8 at turn 15)` on the percentage-point scale. Its p-value is a fixed-look exploratory cluster-Wald approximation, not the confirmatory primary test.

| Cache | Seed clusters | Interaction (seed-cluster 95% CI) | Exploratory fixed-look cluster-Wald p |
|---|---:|---:|---:|
| warm | 128 | +18.4 pp (+14.6, +22.1) | 2.811e-22 |

## Teacher-forced canonical tool sequence

Positive log-probability differences favor BF16 KV for the exact expected tool-call suffix. The decision margin compares `<|tool_call>` with each arm's best first-token alternative.

| Cache | Turn | Prefix kind | Matched snapshots | BF16−FP8 mean logp/token | BF16−FP8 first-decision margin |
|---|---:|---|---:|---:|---:|
| warm | 12 | golden_mechanism | 1 | +0.19910 | +8.36454 |
| warm | 12 | real_prefix_bank | 12 | +0.01587 | +1.40994 |
| warm | 15 | golden_mechanism | 1 | -0.06667 | -1.83374 |
| warm | 15 | real_prefix_bank | 12 | -0.01150 | -0.10555 |
