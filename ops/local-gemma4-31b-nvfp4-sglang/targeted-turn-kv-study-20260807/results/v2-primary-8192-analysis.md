# Targeted KV replay analysis

## Paired KV effects

Positive differences favor BF16 KV. The real-prefix bank weights every prefix equally. Because each prefix reuses the same position-keyed seed stream, the primary uncertainty calculation jointly resamples seed clusters across prefixes. The simultaneous interval is Bonferroni-safe across two turns and all three permitted sample-size looks and governs both the precision continuation rule and confirmatory inference. Ordinary fixed-look p-values are intentionally omitted.

| Cell | Pairs / seed clusters | BF16 success | FP8 success | Difference (seed-cluster 95% CI) | Two-turn/three-look simultaneous CI |
|---|---:|---:|---:|---:|---:|
| warm_turn12_golden | 2048 / 2048 | 99.9% | 48.7% | +51.2 pp (+49.0, +53.4) | (+48.3, +54.2) |
| warm_turn12_bank | 6144 / 512 | 64.2% | 48.3% | +15.9 pp (+14.5, +17.3) | (+13.9, +17.8) |
| warm_turn15_golden | 2048 / 2048 | 0.7% | 0.2% | +0.5 pp (+0.0, +0.9) | (-0.0, +1.1) |
| warm_turn15_bank | 6144 / 512 | 56.5% | 56.0% | +0.5 pp (-0.6, +1.6) | (-0.9, +2.0) |

## Primary-look integrity and stopping decision

The exact cumulative 8,192-case-per-turn/arm allocation passed: 32,768 rows, 12 bank prefixes per turn, 512 paired seed clusters, warm cache, and the frozen ABBA stage mapping.

| Cell | Estimate | Simultaneous interval | Interval half-width | ±2 pp target met |
|---|---:|---:|---:|:---:|
| warm_turn12_bank | +15.9 pp | (+13.9, +17.8) | 1.93 pp | yes |
| warm_turn15_bank | +0.5 pp | (-0.9, +2.0) | 1.46 pp | yes |

Stopping decision: **stop_precision_met**. No continuation is required by the frozen precision rule.

Confirmatory interpretation uses the simultaneous intervals above. Earlier pilot Holm p-values are exploratory and are not reused.

## Turn interaction

This secondary contrast is `(BF16−FP8 at turn 12) − (BF16−FP8 at turn 15)` on the percentage-point scale. Its p-value is a fixed-look exploratory cluster-Wald approximation, not the confirmatory primary test.

| Cache | Seed clusters | Interaction (seed-cluster 95% CI) | Exploratory fixed-look cluster-Wald p |
|---|---:|---:|---:|
| warm | 512 | +15.3 pp (+13.5, +17.2) | 6.24e-62 |

## Teacher-forced canonical tool sequence

Positive log-probability differences favor BF16 KV for the exact expected tool-call suffix. The decision margin compares `<|tool_call>` with each arm's best first-token alternative.

| Cache | Turn | Prefix kind | Matched snapshots | BF16−FP8 mean logp/token | BF16−FP8 first-decision margin |
|---|---:|---|---:|---:|---:|
| warm | 12 | golden_mechanism | 1 | +0.19910 | +8.36454 |
| warm | 12 | real_prefix_bank | 12 | +0.01587 | +1.40994 |
| warm | 15 | golden_mechanism | 1 | -0.06667 | -1.83374 |
| warm | 15 | real_prefix_bank | 12 | -0.01150 | -0.10555 |
