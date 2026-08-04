# Inkling Small on BaseTen: low versus none

Final fixed-denominator results for 30 `reasoning_effort=none` and 30 `reasoning_effort=low` conversations. The 60 requests were strictly sequential in 30 frozen temporal pairs. Each conversation contributes 30 scheduled turns; missing future turns after any short canonical run are failures, regardless of whether the immediate cause was model behavior or serving.

## Accuracy

| Arm | Strict pass | 95% conversation-bootstrap CI | Any error | Tool error | Instruction error | KB error |
|---|---:|---:|---:|---:|---:|---:|
| none | 75.1% | 65.3 to 84.0 | 24.9% | 23.8% | 24.2% | 20.4% |
| low | 51.7% | 44.6 to 58.9 | 48.3% | 48.0% | 48.3% | 43.6% |

Low minus none strict-pass effect: **-23.4 points** (paired whole-conversation bootstrap 95% CI -36.8 to -9.1).

## Completion

| Arm | All scheduled turns | Strict terminal completion | Missing future turns | Recovery terminal calls |
|---|---:|---:|---:|---:|
| none | 17/30 (56.7%) | 17/30 (56.7%) | 177 | 0 |
| low | 3/30 (10.0%) | 3/30 (10.0%) | 392 | 8 |

Low minus none strict-completion effect: **-46.7 points** (paired 95% CI -70.0 to -23.3).

Strict completion requires all scheduled turns 0–29 and `end_session` exactly on scheduled turn 29. A synthetic recovery terminal call does not count.

## Latency and reasoning

| Arm | TTFAT P50 | TTFAT P95 | TTFAT Max | Raw TTFB P50 | Reasoning tokens P50 | Reasoning observations |
|---|---:|---:|---:|---:|---:|---:|
| none | 279ms | 828ms | 2024ms | 279ms | 0 | 723 |
| low | 277ms | 849ms | 5770ms | 277ms | 0 | 508 |

Observed-response P50 TTFAT effect, low minus none: **-2ms** (paired conversation-bootstrap 95% CI -4.0 to +2.0 ms).

TTFAT is content-aware: provider-separated reasoning-only chunks do not stop the clock. Raw TTFB measures the first streamed chunk. Timing and reasoning-token summaries are conditional on recorded observed values; missing turns are not assigned invented latency or token counts.

## Concentrated error turns

- `none`: turns 16 (26/30), 17 (14/30), 20 (14/30), 24 (13/30), 28 (13/30), 29 (13/30), 14 (12/30), 25 (12/30).
- `low`: turns 16 (30/30), 24 (27/30), 25 (27/30), 26 (27/30), 27 (27/30), 28 (27/30), 29 (27/30), 17 (26/30).

## Candidate README rows

| Model | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| inkling-small (none) | 75.1% | 24.9% | 23.8% | 24.2% | 20.4% | 279ms | 828ms | 2024ms | BaseTen |
| inkling-small (low) | 51.7% | 48.3% | 48.0% | 48.3% | 43.6% | 277ms | 849ms | 5770ms | BaseTen |

## Methods and audit trail

Strict pass is `tool_use_correct AND instruction_following AND kb_grounding`. Turn-taking is a supplementary dimension retained in the machine-readable artifacts.

Rate intervals use 100,000 deterministic bootstrap draws over whole conversations. Low-minus-none intervals resample the 30 frozen temporal pairs. Quantile intervals use 20,000 paired or clustered draws.

`included-runs.tsv` fixes exact membership and artifact hashes. `turn-errors.tsv` contains all 30 turn-level counts, including missing future turns under the fixed denominator.
