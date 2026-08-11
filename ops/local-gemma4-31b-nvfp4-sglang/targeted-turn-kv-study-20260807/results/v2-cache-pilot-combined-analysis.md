# Targeted KV replay analysis

## Paired KV effects

Positive differences favor BF16 KV. The real-prefix bank weights every prefix equally. Because each prefix reuses the same position-keyed seed stream, the primary uncertainty calculation jointly resamples seed clusters across prefixes. The simultaneous interval is Bonferroni-safe across two turns and all three permitted sample-size looks and governs the precision continuation rule.

| Cell | Pairs / seed clusters | BF16 success | FP8 success | Difference (seed-cluster 95% CI) | Two-turn/three-look simultaneous CI | Cluster score p |
|---|---:|---:|---:|---:|---:|---:|
| warm_turn12_golden | 128 / 128 | 99.2% | 53.9% | +45.3 pp (+36.7, +53.9) | (+33.6, +57.0) | 0 |
| warm_turn12_bank | 384 / 32 | 65.4% | 48.7% | +16.7 pp (+10.7, +22.4) | (+8.6, +24.2) | 5.278e-08 |
| warm_turn15_golden | 128 / 128 | 0.8% | 0.8% | +0.0 pp (-2.3, +2.3) | (-3.1, +3.1) | 1 |
| warm_turn15_bank | 384 / 32 | 56.8% | 56.2% | +0.5 pp (-3.4, +4.2) | (-4.9, +5.5) | 0.7923 |
| cold_turn12_golden | 128 / 128 | 58.6% | 33.6% | +25.0 pp (+13.3, +36.7) | (+8.6, +41.4) | 5.164e-05 |
| cold_turn12_bank | 384 / 32 | 60.2% | 52.3% | +7.8 pp (+2.6, +13.0) | (+0.8, +14.8) | 0.003766 |
| cold_turn15_golden | 128 / 128 | 0.0% | 0.0% | +0.0 pp (+0.0, +0.0) | (+0.0, +0.0) | 1 |
| cold_turn15_bank | 384 / 32 | 55.5% | 57.8% | -2.3 pp (-6.8, +2.1) | (-8.3, +3.6) | 0.3047 |

Holm-adjusted seed-cluster-robust primary bank p-values (report as confirmatory only at the final sample size): `warm_turn12_bank`=1.056e-07, `warm_turn15_bank`=0.7923.

## Turn interaction

This secondary contrast is `(BF16−FP8 at turn 12) − (BF16−FP8 at turn 15)` on the percentage-point scale.

| Cache | Seed clusters | Interaction (seed-cluster 95% CI) | Cluster score p |
|---|---:|---:|---:|
| warm | 32 | +16.1 pp (+8.6, +23.7) | 4.534e-05 |
| cold | 32 | +10.2 pp (+3.6, +16.4) | 0.002221 |

## Warm-versus-cold outcome discordance

| Arm | Turn | Prefix kind | Pairs | Discordance |
|---|---:|---|---:|---:|
| bf16 | 12 | golden_mechanism | 128 | 40.6% |
| bf16 | 12 | real_prefix_bank | 384 | 38.0% |
| bf16 | 15 | golden_mechanism | 128 | 0.8% |
| bf16 | 15 | real_prefix_bank | 384 | 20.6% |
| fp8 | 12 | golden_mechanism | 128 | 50.0% |
| fp8 | 12 | real_prefix_bank | 384 | 35.9% |
| fp8 | 15 | golden_mechanism | 128 | 0.8% |
| fp8 | 15 | real_prefix_bank | 384 | 18.2% |

## Cache difference-in-differences

The estimate is `(BF16−FP8) warm − (BF16−FP8) cold`; equivalence requires the full seed-cluster interval to lie inside ±3 points.

| Cell | Four-cell pairs / seed clusters | DiD (seed-cluster 95% CI) | ±3 pp equivalent |
|---|---:|---:|:---:|
| turn12_golden | 128 / 128 | +20.3 pp (+5.5, +35.2) | no |
| turn12_bank | 384 / 32 | +8.9 pp (+1.0, +16.7) | no |
| turn15_golden | 128 / 128 | +0.0 pp (-2.3, +2.3) | yes |
| turn15_bank | 384 / 32 | +2.9 pp (-2.6, +8.3) | no |

## Teacher-forced canonical tool sequence

Positive log-probability differences favor BF16 KV for the exact expected tool-call suffix. The decision margin compares `<|tool_call>` with each arm's best first-token alternative.

| Cache | Turn | Prefix kind | Matched snapshots | BF16−FP8 mean logp/token | BF16−FP8 first-decision margin |
|---|---:|---|---:|---:|---:|
| cold | 12 | golden_mechanism | 1 | +0.18572 | +6.43266 |
| cold | 12 | real_prefix_bank | 12 | +0.03166 | +0.65612 |
| cold | 15 | golden_mechanism | 1 | -0.03966 | -1.15146 |
| cold | 15 | real_prefix_bank | 12 | +0.00204 | +0.21346 |
| warm | 12 | golden_mechanism | 1 | +0.19910 | +8.36454 |
| warm | 12 | real_prefix_bank | 12 | +0.01587 | +1.40994 |
| warm | 15 | golden_mechanism | 1 | -0.06667 | -1.83374 |
| warm | 15 | real_prefix_bank | 12 | -0.01150 | -0.10555 |
