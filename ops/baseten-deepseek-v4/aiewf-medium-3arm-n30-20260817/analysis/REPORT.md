# DeepSeek V4 Flash 0731 & V4 Pro 0813 — AIEWF medium-context, three arms

Campaign 2026-08-17T21:43Z → 2026-08-18T04:40Z through Baseten's
OpenAI-compatible Model API with the exact model IDs, native
`reasoning_effort`, temperature 1.0, model-default top-p, and an
8,192-token cap. Each arm is an independent 30-conversation cohort against
the fixed 900-turn denominator; absent scripted turns count as failures for
strict and every component. Strict pass requires tool use, instruction
following, KB grounding, and turn taking all correct. Following the
leaderboard convention, TTFAT excludes each conversation's first scripted
response and includes recovery responses.

| Measure | Flash (low) | Flash (high) | Pro (low) |
|---|---:|---:|---:|
| Strict pass | 870/900 (96.7%) | 845/900 (93.9%) | 876/900 (97.3%) |
| Tool error | 2.8% | 5.9% | 2.7% |
| Instruction error | 3.2% | 6.1% | 2.6% |
| KB error | 0.7% | 4.8% | 2.0% |
| TTFAT P50 / P95 / Max (ms) | 677 / 1444 / 4687 | 762 / 1871 / 8702 | 752 / 1477 / 3545 |
| TTFAT observations | 887 | 836 | 857 |

Values recomputed 2026-08-18 from the per-run judged artifacts listed in
`../manifest.tsv` (30 runs per arm under `runs/aiwf_medium_context/`); they
match the published leaderboard rows exactly on strict counts, component
error rates, and TTFAT max. P50/P95 differ from the published summaries by
at most 1/8 ms due to quantile-interpolation method; the published values
came from the campaign's original analyzer.

Raw campaign console logs remain local-only in
`runs/deepseek-v4-baseten-full3-20260818T044040Z/` (gitignored, like all
run transcripts); `manifest.tsv` records the log-to-run-directory mapping
so each published number is traceable to its judged artifacts.
