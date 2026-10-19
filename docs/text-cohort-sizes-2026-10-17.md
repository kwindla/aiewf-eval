# Complete-conversation cohort sizes, October 17–18, 2026

> Recovery-context migration, October 19: quality now uses the [reconciled grades](model-evaluations-2026-10/judge-recovery-context-20261019/REPORT.md). Cohort membership, completion counts and recorded latency are unchanged. Inventories below remain historical.

> Responses adapter publication, October 18: the current table has 50 ranked configurations, 1,344 complete conversations, and 17 shortfalls. Twelve affected configurations received 30 prospective assignments each; 1 have zero completes and remain unranked. See [fresh cohort results](model-evaluations-2026-10/responses-repair-20261018/REPORT.md). The inventories below are historical.

> Earlier publication update, October 18, before the new-model campaign: the leaderboard had 42 text configurations, 1,072 complete conversations, and 13 shortfalls. Fresh Baseten Ultra thinking-on (N=30) replaces the historical Modal 128/96 rows. The inventory below preserves the original 43-configuration refresh.

The exhaustive local search supersedes the earlier summary-only size assessment. It includes unjudged transcripts, nested projects, other checkouts, and compressed Nemotron sweeps. Historical study counts are not the selected sample size: the new analysis uses the latest complete conversations at a matching configuration, up to 30 text / 25 voice.

There are **43 text configurations, 1,091 selected complete conversations, and 14 text shortfalls**. All nine voice profiles have 25, for 225 voice conversations. No new conversations were generated. Incomplete traces and their observed prefixes are excluded from accuracy, so this is conditional-on-completion quality. Incorrect tools are graded rather than excluded.

The primary inventory has **8,060 source files, 7,370 distinct transcript byte hashes, and 690 copies across 35 text model families**. A supplementary filename-independent check found 203 complete Qwen timing/DFlash files or copies; all fail the selected serving/prompt profile. It found no additional standard transcripts outside new campaign/test files, no orphan complete judged captures, and no parse errors. These supplementary files are not counted as independent repetitions.

Most importantly, the old Super archive contains **40 complete native-budget 512 conversations**, despite an old summary reporting only 38 judged conversations. Its newest 30 are remapped to an explicit AWS B200 NVFP4 row. Nano has seven complete native-budget 512 NVFP4 conversations and nine older complete legacy-parameter 512 captures on a different stack; they are not pooled. Ultra 128 has 33 matching native-budget Modal completions, of which 30 are selected; Ultra 96 has 19. Across all serving/sampling variants the corresponding budget counts are much larger (Ultra 128: 77, Ultra 96: 56, Super 512: 81, Nano 512: 16), but these are not coherent cohorts.

The same distinction explains the OpenAI shortfalls: alternate effort levels, 96-dot filler, incomplete trials, and copied files do not count toward a matching 30. Counts below are selected complete conversations; available is the matching complete pool before taking the newest 30. See each manifest for exclusions, source timestamps, recorded request settings, and input hashes.

| Configuration | Selected N | Available | Shortfall | Earliest selected | Latest selected | Evidence |
|---|---:|---:|---:|---|---|---|
| deepseek-v4-flash-0731 (high) | 30 | 32 | 0 | 2026-08-17 | 2026-08-18 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| deepseek-v4-flash-0731 (low) | 30 | 31 | 0 | 2026-08-17 | 2026-08-25 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| deepseek-v4-pro-0813 (low) | 30 | 34 | 0 | 2026-08-17 | 2026-08-25 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gemini-2.5-flash (thinking off) | 30 | 31 | 0 | 2026-07-22 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gemini-3.5-flash (minimal) | 30 | 47 | 0 | 2026-05-19 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gemini-3.5-flash-lite (minimal) | 30 | 49 | 0 | 2026-08-04 | 2026-08-25 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gemini-3.6-flash (minimal) | 30 | 32 | 0 | 2026-07-22 | 2026-08-25 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gemini-3.7-flash (low) | 30 | 32 | 0 | 2026-08-13 | 2026-08-25 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gemini-3.8-flash (low) | 30 | 33 | 0 | 2026-10-17 | 2026-10-17 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gemma-4-26b-a4b-it (thinking off) | 30 | 60 | 0 | 2026-07-31 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gemma-4-31b-it (thinking off) | 30 | 153 | 0 | 2026-08-07 | 2026-08-07 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| glm-5.2 (none) | 30 | 31 | 0 | 2026-07-19 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| glm-5.3 (low reasoning) | 30 | 33 | 0 | 2026-08-19 | 2026-08-19 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-4.1 | 26 | 26 | 4 | 2026-02-18 | 2026-08-26 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-4.1-mini | 7 | 7 | 23 | 2026-03-11 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-4o | 11 | 11 | 19 | 2026-02-19 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-4o-mini | 11 | 11 | 19 | 2026-02-19 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5-mini | 11 | 11 | 19 | 2026-02-19 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5.1 | 25 | 25 | 5 | 2026-02-18 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5.2 | 11 | 11 | 19 | 2026-02-19 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5.4 (low) | 19 | 19 | 11 | 2026-03-05 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5.4 (none) | 30 | 72 | 0 | 2026-07-21 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5.4 (none, +96 dots) | 30 | 31 | 0 | 2026-07-19 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5.5 (none) | 30 | 31 | 0 | 2026-07-20 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5.6-luna (none) | 21 | 21 | 9 | 2026-07-16 | 2026-08-26 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5.6-sol (none) | 30 | 31 | 0 | 2026-07-19 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5.6-terra (medium) | 8 | 8 | 22 | 2026-07-16 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| gpt-5.6-terra (none) | 30 | 33 | 0 | 2026-07-16 | 2026-08-26 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| inkling (none) | 30 | 30 | 0 | 2026-07-19 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| kimi-k2.6 (thinking off) | 30 | 32 | 0 | 2026-08-06 | 2026-08-25 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| kimi-k2.6 (thinking on) | 30 | 30 | 0 | 2026-08-06 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| muse-glimmer-30b (thinking low, GGUF) | 30 | 165 | 0 | 2026-09-25 | 2026-09-25 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| nemotron-3-nano-30b (512, NVFP4) | 7 | 7 | 23 | 2026-06-05 | 2026-06-05 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| nemotron-3-super-120b (512, NVFP4) | 30 | 40 | 0 | 2026-06-05 | 2026-06-05 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| nemotron-3-ultra (128) | 30 | 33 | 0 | 2026-06-03 | 2026-06-04 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| nemotron-3-ultra (96) | 19 | 19 | 11 | 2026-06-03 | 2026-06-03 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| nemotron-3.5-lightning (thinking off, NVFP4) | 16 | 16 | 14 | 2026-08-12 | 2026-08-13 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| nemotron-3.5-lightning (thinking on, NVFP4) | 30 | 33 | 0 | 2026-08-12 | 2026-08-13 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| poolside/laguna-s-2.1 (thinking off) | 29 | 29 | 1 | 2026-07-22 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| qwen3.6-27b (thinking off) | 30 | 31 | 0 | 2026-07-28 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| qwen3.6-35b-a3b (thinking off, FP8) | 30 | 31 | 0 | 2026-07-28 | 2026-08-06 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| qwen3.8-27b (thinking off, FP8) | 30 | 30 | 0 | 2026-08-16 | 2026-08-17 | [manifest (local archive)](model-evaluations-2026-10/README.md) |
| qwen3.8-27b (thinking off, NVFP4) | 30 | 31 | 0 | 2026-08-16 | 2026-08-16 | [manifest (local archive)](model-evaluations-2026-10/README.md) |

The [source audit](model-evaluations-2026-10/judge-refresh-opus55-20261017/SOURCE-AUDIT.md), [directory inventory (local archive)](model-evaluations-2026-10/README.md), and [configuration inventory](model-evaluations-2026-10/judge-refresh-opus55-20261017/configurations.json) preserve the broader search. Different efforts, filler, quantizations, and serving endpoints are not merged merely to reach 30. Voice capture diagnostics and harness defects remain in the original attempt ledger.
