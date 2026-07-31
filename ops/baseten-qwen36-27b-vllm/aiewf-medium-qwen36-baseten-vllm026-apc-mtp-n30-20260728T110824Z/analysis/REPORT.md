# Qwen3.6-27B: native thinking-on versus thinking-off

Final fixed-denominator results for the BaseTen vLLM 0.26 APC+MTP deployment. Each arm contains 30 canonical conversations and every conversation contributes 30 scheduled turns. After an early model exit, every unobserved future turn is scored as a failure on every dimension.

## Accuracy

| Arm | Strict pass | 95% cluster-bootstrap CI | Any error | Tool error | Instruction error | KB error |
|---|---:|---:|---:|---:|---:|---:|
| High / native thinking-on | 86.0% | 76.4–94.1 | 14.0% | 13.8% | 13.9% | 12.7% |
| None / thinking-off | 97.3% | 96.6–98.1 | 2.7% | 2.7% | 2.7% | 0.0% |

High minus none strict-pass effect: **-11.3 points** (paired whole-conversation bootstrap 95% CI -20.8 to -3.1).

## Completion

| Arm | All 30 scheduled turns | Wilson 95% CI | `end_session` exactly at scheduled turn 29 | Wilson 95% CI | Missing future turns |
|---|---:|---:|---:|---:|---:|
| High / native thinking-on | 76.7% | 59.1–88.2 | 76.7% | 59.1–88.2 | 114 |
| None / thinking-off | 100.0% | 88.6–100.0 | 100.0% | 88.6–100.0 | 0 |

“All 30 scheduled turns” measures response coverage. Strict protocol completion additionally requires the terminal tool on scripted turn 29; a synthetic recovery call at turn 30 does not count.

## Latency

| Arm | TTFAT observations | Coverage | P50 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| High / native thinking-on | 786 | 87.3% | 1924 ms | 9468 ms | 18023 ms |
| None / thinking-off | 900 | 100.0% | 668 ms | 822 ms | 5429 ms |

Observed-response P50 TTFAT effect, high minus none: **1256 ms** (paired conversation-bootstrap 95% CI 1144 to 1402 ms).

TTFAT is the content-aware time to the first assistant text or tool-call token. It is summarized only where a scheduled model response was observed; missing turns remain accuracy failures but are not assigned fictitious latency.

## Methods and audit trail

Strict pass is the conjunction of tool-use correctness, instruction following, and KB grounding, matching the README benchmark definition. Turn-taking error is retained as a supplementary metric in `aggregates.json` and `effects.tsv`.

Arm confidence intervals resample whole conversations. High-minus-none intervals resample the 30 frozen high/none pairs, preserving the campaign's balanced temporal blocks. Rate intervals use 100,000 deterministic bootstrap draws; latency-quantile intervals use 20,000.

`included-runs.tsv` records the exact manifest membership, per-run fixed-denominator counts, judge identity, and SHA-256 hashes of every transcript, judgment, and judge summary.
