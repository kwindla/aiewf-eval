# Historical upstream-history mediation screen

This post-hoc screen uses the 300 completed local conversations. It asks whether
the KV arms produced different lead-in histories, and whether those features are
associated with the target-turn failures. It is descriptive, not a formal causal
mediation estimate.

## Arm summaries

| Arm | N | Turn 12 failure | Turn 15 failure | Turns 9–11 assistant words | Turns 13–14 assistant words |
|---|---:|---:|---:|---:|---:|
| local_fp8 | 150 | 65.3% | 57.3% | 50.7 | 69.2 |
| local_bf16 | 150 | 46.0% | 51.3% | 50.6 | 69.4 |

## Feature associations pooled across arms

| Feature | Outcome | Present N | Absent N | Failure if present | Failure if absent | Difference (95% bootstrap CI) |
|---|---|---:|---:|---:|---:|---:|
| turn12_recovery_recorded | turn15_failure | 167 | 133 | 58.7% | 48.9% | +9.8 pp (-1.7, +21.1) |
| turn14_mentions_jennifer | turn15_failure | 82 | 218 | 13.4% | 69.7% | -56.3 pp (-65.5, -46.4) |
| turn14_asks_question | turn15_failure | 299 | 1 | 54.5% | 0.0% | +54.5 pp (+48.8, +60.2) |

See `historical-mediation.json` for all arm feature rates.
