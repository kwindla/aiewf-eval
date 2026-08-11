# Targeted KV replay analysis

## Paired KV effects

Positive differences favor BF16 KV. The real-prefix bank weights every prefix equally. Because each prefix reuses the same position-keyed seed stream, the primary uncertainty calculation jointly resamples seed clusters across prefixes. The simultaneous interval is Bonferroni-safe across two turns and all three permitted sample-size looks and governs both the precision continuation rule and confirmatory inference. Ordinary fixed-look p-values are intentionally omitted.

| Cell | Pairs / seed clusters | BF16 success | FP8 success | Difference (seed-cluster 95% CI) | Two-turn/three-look simultaneous CI |
|---|---:|---:|---:|---:|---:|
| warm_turn12_golden | 1024 / 1024 | 99.9% | 50.2% | +49.7 pp (+46.7, +52.7) | (+45.6, +53.8) |
| warm_turn12_bank | 3072 / 256 | 64.9% | 47.9% | +17.0 pp (+14.8, +19.1) | (+14.1, +19.8) |
| warm_turn15_golden | 1024 / 1024 | 0.7% | 0.3% | +0.4 pp (-0.2, +1.0) | (-0.4, +1.3) |
| warm_turn15_bank | 3072 / 256 | 56.1% | 55.5% | +0.6 pp (-0.8, +2.0) | (-1.4, +2.5) |

## Primary-look integrity and stopping decision

The exact cumulative 4,096-case-per-turn/arm allocation passed: 16,384 rows, 12 bank prefixes per turn, 256 paired seed clusters, warm cache, and the frozen ABBA stage mapping.

| Cell | Estimate | Simultaneous interval | Interval half-width | ±2 pp target met |
|---|---:|---:|---:|:---:|
| warm_turn12_bank | +17.0 pp | (+14.1, +19.8) | 2.83 pp | no |
| warm_turn15_bank | +0.6 pp | (-1.4, +2.5) | 1.94 pp | yes |

Stopping decision: **continue**. The next cumulative look is 8,192 cases per turn/arm.

Confirmatory interpretation uses the simultaneous intervals above. Earlier pilot Holm p-values are exploratory and are not reused.

## Turn interaction

This secondary contrast is `(BF16−FP8 at turn 12) − (BF16−FP8 at turn 15)` on the percentage-point scale. Its p-value is a fixed-look exploratory cluster-Wald approximation, not the confirmatory primary test.

| Cache | Seed clusters | Interaction (seed-cluster 95% CI) | Exploratory fixed-look cluster-Wald p |
|---|---:|---:|---:|
| warm | 256 | +16.4 pp (+13.7, +19.0) | 2.026e-33 |

## Teacher-forced canonical tool sequence

Positive log-probability differences favor BF16 KV for the exact expected tool-call suffix. The decision margin compares `<|tool_call>` with each arm's best first-token alternative.

| Cache | Turn | Prefix kind | Matched snapshots | BF16−FP8 mean logp/token | BF16−FP8 first-decision margin |
|---|---:|---|---:|---:|---:|
| warm | 12 | golden_mechanism | 1 | +0.19910 | +8.36454 |
| warm | 12 | real_prefix_bank | 12 | +0.01587 | +1.40994 |
| warm | 15 | golden_mechanism | 1 | -0.06667 | -1.83374 |
| warm | 15 | real_prefix_bank | 12 | -0.01150 | -0.10555 |
