# README rejudging and remapping

Processing finished 2026-10-18T14:25:42.857622+00:00. Scope: **43 text configurations (1,091 conversations)** and **nine voice configurations (225 conversations)**, totaling **1,316 selected conversations / 39,480 scripted turns**. **1316 conversations received valid grades; 0 remain unscored because of provider safeguard rejections.** Full-cohort quality is available for every selected configuration. No new model conversations were generated.

Every quality request used `claude-opus-5-5`, medium effort, adaptive thinking, no tools, and no project settings. The judge received neither generating-model identities nor prior scores. Text finished before voice judging started. The exact rubric hashes and selection/latency policies are in [aggregate.json](summary.json) and [PROTOCOL.md](PROTOCOL.md). Per-conversation outputs are in [results/ (local archive)](../README.md); all attempts are recorded in [attempts.jsonl (local archive)](../README.md).

SDK-reported **successful-request cost: $251.89** ($209.55 text / $42.33 voice); 0 current successful requests lack cost metadata. The ledger contains 16 operational error events. This total is not an invented estimate for any failed request without usage metadata. Credit/balance failures receive a shared wait of at least 90 seconds before retrying. Only the first structurally valid grade per frozen input and preprocessing version is accepted; no score-based retries or edits. Initially, one Nano input received four provider `reasoning_extraction` rejections and remained rejected after an additive task clarification. That note requested grading of literal recorded output with brief score justifications; it did not remove the analysis-like source text, change criteria, or request hidden reasoning. The exact exception is in [task-clarifications.json (local archive)](../README.md), with source/rubric/model/effort unchanged and the attempted clarification hash in the attempt ledger. The original unscored rejections and later retry outcomes remain in [retries/20261018/ (local archive)](../README.md). Provider rejections never count as quality passes or failures.

User-requested retry finished 2026-10-18T14:08:08.866714+00:00: 3 previously rejected inputs retried, 0 graded and 3 rejected again. Sources, rubric, model, effort, and existing per-input clarifications were unchanged. Prior rejection records and completion/verification snapshots are preserved beside the [retry record (local archive)](../README.md).

User-requested reasoning cleanup finished 2026-10-18T14:24:53.481375+00:00: audited 1316 inputs across 52 configurations. Removed delimited analysis from 49 scripted turns in 29 conversations; 29 replacement/new grades and 0 provider rejections. All other grades were reused. The same cleanup rule applies before judging regardless of whether an earlier request succeeded. Empty final text with a tool call remains a tool-only response. Raw sources, selected membership, rubric, model, effort, actual tool calls, and latency observations are unchanged. The obsolete per-input clarification was omitted. Prior grades and rejected dispositions are archived under [reasoning-cleanup/ (local archive)](../README.md); the [manifest](reasoning-cleanup/manifest.json) and result metadata record per-turn source/answer hashes and removed character counts. Cost includes $5.87 for superseded grades. Score differences on rejudged conversations can include judge variability as well as input cleanup.

## Selection and comparability

The [source audit](SOURCE-AUDIT.md) and [configuration inventory](configurations.json) cover copies, archives, diagnostic captures, and configuration variants. Each text row uses its most recent matching complete conversations, up to 30. A complete transcript has every scripted index 0–29 exactly once with an assistant response or tool call. Wrong or absent required tools remain quality issues. Recovery responses are preserved in source snapshots and included in text latency. The existing judge formatter omits recovery records from both its prompt and scored denominators; later scripted responses are evaluated from the scripted transcript and realignment evidence. This limitation is shared across this refresh.

Fourteen text rows have a shortfall, detailed in the [cohort-size audit](../../text-cohort-sizes-2026-10-17.md). We did not pad sample sizes by mixing filler, reasoning, precision, or serving configurations. Voice uses all 25 validated completions of each final published capture profile; newer valid matching captures were not found.

This is a **conditional-on-completion analysis**. Incomplete conversations and observed prefixes are excluded; missing future turns are not errors in this analysis. Earlier fixed-denominator, attempt-based scores answer a different question. Completion selection can favor configurations with high failure rates, especially Lightning off; quality here must not be interpreted as reliability. Voice separately retains its full 245-attempt ledger and all six failures/14 harness defects. Text manifests enumerate incomplete and differently configured candidates; historical attempt-based campaigns remain available for reliability context.

The four Nemotron 3 rows replace old unmapped Baseten summaries. Ultra uses coherent native-budget Modal captures; Super/Nano use coherent native-budget NVFP4 AWS B200 captures. Super uses the newest 30 of a recovered 40-completion study. Nano has seven at its replacement configuration; nine older legacy-parameter captures are not pooled. Ultra 96 has 19 matching native-budget captures, despite many more captures at other settings. These changes affect serving/cohort and timing as well as judging. Provider labels and NVFP4 labels were corrected. Old versus new differences are **not causal judge comparisons**. OpenAI row definitions match model, reasoning, and filler rather than creating separate rows for every historical service tier; recorded priority/fast labels and unlogged older tiers remain explicit in manifests. Their latency is a historical cohort measurement, not a standardized tier experiment. No cross-model significance claims are made: capture dates, hardware, SDKs, caching, samplers, and sometimes prompts differ.

## Quality and uncertainty

Strict pass requires tool use, instruction following, and KB grounding all to pass. Component errors overlap. Turn-taking is reported separately and never enters strict quality, even where available. Intervals use 20,000 whole-conversation bootstrap resamples with deterministic per-cohort seeds. They account for within-conversation turn correlation, but not judge uncertainty, configuration differences, or collection drift; a 100% sample yields a degenerate empirical interval and does not establish population perfection.

| Configuration | Selected N | Graded N | Provider blocked | Known strict passes / graded turns | Full-cohort pass rate | Cluster 95% CI | Current grade cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-v4-flash-0731 (high) | 30 | 30 | 0 | 884/900 observed | 98.2% | 96.9–99.3% | $5.42 |
| deepseek-v4-flash-0731 (low) | 30 | 30 | 0 | 868/900 observed | 96.4% | 95.1–97.7% | $5.50 |
| deepseek-v4-pro-0813 (low) | 30 | 30 | 0 | 888/900 observed | 98.7% | 98.0–99.3% | $5.21 |
| gemini-2.5-flash (thinking off) | 30 | 30 | 0 | 734/900 observed | 81.6% | 79.2–83.9% | $5.31 |
| gemini-3.5-flash (minimal) | 30 | 30 | 0 | 851/900 observed | 94.6% | 93.2–95.8% | $5.38 |
| gemini-3.5-flash-lite (minimal) | 30 | 30 | 0 | 842/900 observed | 93.6% | 92.1–94.9% | $5.21 |
| gemini-3.6-flash (minimal) | 30 | 30 | 0 | 858/900 observed | 95.3% | 93.0–97.2% | $5.07 |
| gemini-3.7-flash (low) | 30 | 30 | 0 | 899/900 observed | 99.9% | 99.7–100.0% | $4.72 |
| gemini-3.8-flash (low) | 30 | 30 | 0 | 899/900 observed | 99.9% | 99.7–100.0% | $4.81 |
| gemma-4-26b-a4b-it (thinking off) | 30 | 30 | 0 | 740/900 observed | 82.2% | 80.1–84.0% | $5.44 |
| gemma-4-31b-it (thinking off) | 30 | 30 | 0 | 863/900 observed | 95.9% | 94.8–97.0% | $4.76 |
| glm-5.2 (none) | 30 | 30 | 0 | 861/900 observed | 95.7% | 94.4–96.8% | $5.66 |
| glm-5.3 (low reasoning) | 30 | 30 | 0 | 880/900 observed | 97.8% | 96.9–98.6% | $5.40 |
| gpt-4.1 | 26 | 26 | 0 | 737/780 observed | 94.5% | 92.9–95.9% | $4.69 |
| gpt-4.1-mini | 7 | 7 | 0 | 172/210 observed | 81.9% | 80.5–83.8% | $2.05 |
| gpt-4o | 11 | 11 | 0 | 286/330 observed | 86.7% | 83.3–89.7% | $2.15 |
| gpt-4o-mini | 11 | 11 | 0 | 268/330 observed | 81.2% | 76.7–85.2% | $2.20 |
| gpt-5-mini | 11 | 11 | 0 | 259/330 observed | 78.5% | 74.5–81.8% | $2.25 |
| gpt-5.1 | 25 | 25 | 0 | 696/750 observed | 92.8% | 90.3–95.1% | $5.37 |
| gpt-5.2 | 11 | 11 | 0 | 290/330 observed | 87.9% | 84.8–91.2% | $2.09 |
| gpt-5.4 (low) | 19 | 19 | 0 | 544/570 observed | 95.4% | 93.5–97.2% | $3.33 |
| gpt-5.4 (none) | 30 | 30 | 0 | 803/900 observed | 89.2% | 87.1–91.1% | $5.28 |
| gpt-5.4 (none, +96 dots) | 30 | 30 | 0 | 861/900 observed | 95.7% | 94.3–97.0% | $5.21 |
| gpt-5.5 (none) | 30 | 30 | 0 | 865/900 observed | 96.1% | 94.6–97.6% | $5.07 |
| gpt-5.6-luna (none) | 21 | 21 | 0 | 540/630 observed | 85.7% | 83.5–87.8% | $3.62 |
| gpt-5.6-sol (none) | 30 | 30 | 0 | 859/900 observed | 95.4% | 94.9–96.0% | $5.10 |
| gpt-5.6-terra (medium) | 8 | 8 | 0 | 238/240 observed | 99.2% | 97.9–100.0% | $1.27 |
| gpt-5.6-terra (none) | 30 | 30 | 0 | 801/900 observed | 89.0% | 87.1–90.9% | $5.22 |
| inkling (none) | 30 | 30 | 0 | 844/900 observed | 93.8% | 92.1–95.4% | $5.83 |
| kimi-k2.6 (thinking off) | 30 | 30 | 0 | 787/900 observed | 87.4% | 84.8–90.0% | $5.53 |
| kimi-k2.6 (thinking on) | 30 | 30 | 0 | 889/900 observed | 98.8% | 98.0–99.4% | $5.32 |
| muse-glimmer-30b (thinking low, GGUF) | 30 | 30 | 0 | 750/900 observed | 83.3% | 81.8–84.9% | $6.03 |
| nemotron-3-nano-30b (512, NVFP4) | 7 | 7 | 0 | 146/210 observed | 69.5% | 65.2–73.3% | $1.90 |
| nemotron-3-super-120b (512, NVFP4) | 30 | 30 | 0 | 816/900 observed | 90.7% | 89.0–92.2% | $5.87 |
| nemotron-3-ultra (128) | 30 | 30 | 0 | 617/900 observed | 68.6% | 60.4–76.3% | $7.24 |
| nemotron-3-ultra (96) | 19 | 19 | 0 | 291/570 observed | 51.1% | 42.8–59.5% | $5.23 |
| nemotron-3.5-lightning (thinking off, NVFP4) | 16 | 16 | 0 | 338/480 observed | 70.4% | 66.2–74.2% | $3.75 |
| nemotron-3.5-lightning (thinking on, NVFP4) | 30 | 30 | 0 | 750/900 observed | 83.3% | 80.9–85.7% | $6.68 |
| poolside/laguna-s-2.1 (thinking off) | 29 | 29 | 0 | 687/870 observed | 79.0% | 76.7–81.3% | $5.59 |
| qwen3.6-27b (thinking off) | 30 | 30 | 0 | 828/900 observed | 92.0% | 90.3–93.6% | $5.73 |
| qwen3.6-35b-a3b (thinking off, FP8) | 30 | 30 | 0 | 735/900 observed | 81.7% | 78.7–84.3% | $5.83 |
| qwen3.8-27b (thinking off, FP8) | 30 | 30 | 0 | 865/900 observed | 96.1% | 95.0–97.2% | $5.10 |
| qwen3.8-27b (thinking off, NVFP4) | 30 | 30 | 0 | 845/900 observed | 93.9% | 92.2–95.4% | $5.25 |
| GPT-Live prompt-optimized + Terra medium* | 25 | 25 | 0 | 707/750 observed | 94.3% | 92.7–95.9% | $4.40 |
| GPT-Live standard + Terra medium | 25 | 25 | 0 | 689/750 observed | 91.9% | 90.1–93.6% | $4.40 |
| GPT-Realtime-1.5 | 25 | 25 | 0 | 694/750 observed | 92.5% | 91.2–93.7% | $4.68 |
| GPT-Realtime-2.1 low | 25 | 25 | 0 | 699/750 observed | 93.2% | 92.1–94.3% | $4.69 |
| Gemini 3.1 minimal | 25 | 25 | 0 | 628/750 observed | 83.7% | 80.0–86.9% | $6.23 |
| Gemini 3.8 Extended Thinking low | 25 | 25 | 0 | 672/750 observed | 89.6% | 87.7–91.5% | $4.44 |
| Gemini 3.8 Extended Thinking medium (interaction-aware) | 25 | 25 | 0 | 678/750 observed | 90.4% | 88.8–92.0% | $4.51 |
| Gemini 3.8 Live (regular; default) | 25 | 25 | 0 | 671/750 observed | 89.5% | 87.3–91.5% | $4.41 |
| Grok Voice Think Fast | 25 | 25 | 0 | 691/750 observed | 92.1% | 90.7–93.5% | $4.57 |

Provider rejection is missing judge data, not observed model correctness. No point estimate or bootstrap interval is published for an incomplete quality cohort. Known successful grades remain in results and aggregate counts; bounds in aggregate.json show the possible full-cohort accuracy range without imputing unknown turns. Rows with unavailable quality sort after fully graded rows. The fixed selected membership is not changed based on provider acceptance.

## Latency

All numbers were recalculated from the selected frozen evidence. [latency-observations.json (local archive)](../README.md) contains the individual values, turn indices, recovery flags, and voice tool/silence observations. Text excludes the first scripted response, includes recovery, and publishes positive finite recorded service TTFAT; P95 uses linear interpolation. Every selected text source was checked byte-for-byte against its snapshot. Native thinking paths measure user-visible content/tool output rather than reasoning deltas where exposed by the original service. This does not standardize SDK versions or provider instrumentation retroactively. Local rows omit hosted network time.

Voice recomputation exactly reproduces every historical sample count, mean, median, maximum, missing count, and nonpositive count for all nine profiles. Original audio JSON, original quality reconstruction, and cleaned spoken input are hashed and checked. All scripted turns enter the voice measurement scope. Actual first speech includes acknowledgments and is not necessarily the substantive answer. Silence uses nonnegative original Silero padding; RMS is not substituted. Unavailable values remain null.

| Voice configuration | Positive V2V | Nonpositive | Missing | Tool | Non-tool | Silence | Turn-taking passes / 750 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-Live prompt-optimized + Terra medium* | 748 | 1 | 1 | 150 | 598 | 0 | unscored |
| GPT-Live standard + Terra medium | 748 | 1 | 1 | 123 | 625 | 0 | 749 |
| GPT-Realtime-1.5 | 725 | 0 | 25 | 122 | 603 | 733 | 737 |
| GPT-Realtime-2.1 low | 750 | 0 | 0 | 135 | 615 | 750 | 750 |
| Gemini 3.1 minimal | 721 | 0 | 29 | 129 | 592 | 691 | 750 |
| Gemini 3.8 Extended Thinking low | 743 | 0 | 7 | 143 | 600 | 0 | 750 |
| Gemini 3.8 Extended Thinking medium (interaction-aware) | 750 | 0 | 0 | 151 | 599 | 0 | unscored |
| Gemini 3.8 Live (regular; default) | 749 | 0 | 1 | 156 | 593 | 0 | 749 |
| Grok Voice Think Fast | 730 | 0 | 20 | 138 | 592 | 730 | 730 |

## Selected source coverage

Dates below are recorded source timestamps (or capture-ID timestamps when that field is unavailable). Every selected path and SHA, available generation configuration, candidate exclusion, and archive reference is preserved in the linked manifest.

| Configuration | Provider | N | Earliest | Latest | Evidence |
|---|---:|---:|---:|---:|---:|
| deepseek-v4-flash-0731 (high) | Baseten | 30 | 2026-08-17T23:26:29.831Z | 2026-08-18T05:32:30.836Z | [manifest (local archive)](../README.md) |
| deepseek-v4-flash-0731 (low) | Baseten | 30 | 2026-08-17T23:15:59.566Z | 2026-08-25T18:16:39.257Z | [manifest (local archive)](../README.md) |
| deepseek-v4-pro-0813 (low) | Baseten | 30 | 2026-08-17T23:50:40.376Z | 2026-08-25T18:20:01.210Z | [manifest (local archive)](../README.md) |
| gemini-2.5-flash (thinking off) | AI Studio | 30 | 2026-07-22T19:35:47.524Z | 2026-08-06T15:54:41.012Z | [manifest (local archive)](../README.md) |
| gemini-3.5-flash (minimal) | AI Studio | 30 | 2026-05-19T22:56:14.524Z | 2026-08-06T15:53:51.865Z | [manifest (local archive)](../README.md) |
| gemini-3.5-flash-lite (minimal) | AI Studio | 30 | 2026-08-04T18:35:18.248Z | 2026-08-25T18:20:01.600Z | [manifest (local archive)](../README.md) |
| gemini-3.6-flash (minimal) | AI Studio | 30 | 2026-07-22T00:55:13.675Z | 2026-08-25T17:20:41.641Z | [manifest (local archive)](../README.md) |
| gemini-3.7-flash (low) | AI Studio | 30 | 2026-08-13T21:05:40.635Z | 2026-08-25T17:14:36.753Z | [manifest (local archive)](../README.md) |
| gemini-3.8-flash (low) | AI Studio | 30 | 2026-10-17T16:19:31.624Z | 2026-10-17T16:46:37.299Z | [manifest (local archive)](../README.md) |
| gemma-4-26b-a4b-it (thinking off) | Baseten | 30 | 2026-07-31T19:00:38.885Z | 2026-08-06T16:23:36.406Z | [manifest (local archive)](../README.md) |
| gemma-4-31b-it (thinking off) | Baseten | 30 | 2026-08-07T11:10:51.615Z | 2026-08-07T11:33:15.182Z | [manifest (local archive)](../README.md) |
| glm-5.2 (none) | Baseten | 30 | 2026-07-19T10:44:40.721Z | 2026-08-06T15:49:44.215Z | [manifest (local archive)](../README.md) |
| glm-5.3 (low reasoning) | Z.AI | 30 | 2026-08-19T21:53:26.424Z | 2026-08-19T23:13:01.088Z | [manifest (local archive)](../README.md) |
| gpt-4.1 | OpenAI | 26 | 2026-02-18T12:17:36.956Z | 2026-08-26T14:09:04.579Z | [manifest (local archive)](../README.md) |
| gpt-4.1-mini | OpenAI | 7 | 2026-03-11T14:49:47.073Z | 2026-08-06T15:52:21.745Z | [manifest (local archive)](../README.md) |
| gpt-4o | OpenAI | 11 | 2026-02-19T06:48:03.957Z | 2026-08-06T15:44:38.449Z | [manifest (local archive)](../README.md) |
| gpt-4o-mini | OpenAI | 11 | 2026-02-19T06:37:01.197Z | 2026-08-06T15:57:08.166Z | [manifest (local archive)](../README.md) |
| gpt-5-mini | OpenAI | 11 | 2026-02-19T06:29:19.543Z | 2026-08-06T15:54:11.612Z | [manifest (local archive)](../README.md) |
| gpt-5.1 | OpenAI | 25 | 2026-02-18T12:05:47.484Z | 2026-08-06T15:36:12.303Z | [manifest (local archive)](../README.md) |
| gpt-5.2 | OpenAI | 11 | 2026-02-19T06:57:14.105Z | 2026-08-06T15:49:58.084Z | [manifest (local archive)](../README.md) |
| gpt-5.4 (low) | OpenAI | 19 | 2026-03-05T22:01:13.269Z | 2026-08-06T15:39:00.672Z | [manifest (local archive)](../README.md) |
| gpt-5.4 (none) | OpenAI | 30 | 2026-07-21T02:55:54.450Z | 2026-08-06T15:48:38.018Z | [manifest (local archive)](../README.md) |
| gpt-5.4 (none, +96 dots) | OpenAI | 30 | 2026-07-19T03:51:33.018Z | 2026-08-06T15:43:14.592Z | [manifest (local archive)](../README.md) |
| gpt-5.5 (none) | OpenAI | 30 | 2026-07-20T17:32:09.594Z | 2026-08-06T15:20:07.904Z | [manifest (local archive)](../README.md) |
| gpt-5.6-luna (none) | OpenAI | 21 | 2026-07-16T17:35:31.299Z | 2026-08-26T14:10:46.847Z | [manifest (local archive)](../README.md) |
| gpt-5.6-sol (none) | OpenAI | 30 | 2026-07-19T16:11:20.891Z | 2026-08-06T15:40:20.611Z | [manifest (local archive)](../README.md) |
| gpt-5.6-terra (medium) | OpenAI | 8 | 2026-07-16T17:59:20.608Z | 2026-08-06T15:37:39.170Z | [manifest (local archive)](../README.md) |
| gpt-5.6-terra (none) | OpenAI | 30 | 2026-07-16T17:15:43.992Z | 2026-08-26T14:07:24.797Z | [manifest (local archive)](../README.md) |
| inkling (none) | Baseten | 30 | 2026-07-19T05:14:21.992Z | 2026-08-06T15:51:20.468Z | [manifest (local archive)](../README.md) |
| kimi-k2.6 (thinking off) | Baseten | 30 | 2026-08-06T16:33:49.193Z | 2026-08-25T18:21:12.445Z | [manifest (local archive)](../README.md) |
| kimi-k2.6 (thinking on) | Baseten | 30 | 2026-08-06T17:35:25.150Z | 2026-08-06T18:30:23.656Z | [manifest (local archive)](../README.md) |
| muse-glimmer-30b (thinking low, GGUF) | Local RTX 5090 | 30 | 2026-09-25T01:27:05.462Z | 2026-09-25T01:47:07.527Z | [manifest (local archive)](../README.md) |
| nemotron-3-nano-30b (512, NVFP4) | AWS B200 | 7 | 2026-06-05T19:02:01.455Z | 2026-06-05T19:09:09.841Z | [manifest (local archive)](../README.md) |
| nemotron-3-super-120b (512, NVFP4) | AWS B200 | 30 | 2026-06-05T21:19:19.966Z | 2026-06-05T21:37:27.486Z | [manifest (local archive)](../README.md) |
| nemotron-3-ultra (128) | Modal | 30 | 2026-06-03T16:14:30.155Z | 2026-06-04T00:39:22.554Z | [manifest (local archive)](../README.md) |
| nemotron-3-ultra (96) | Modal | 19 | 2026-06-03T16:02:31.774Z | 2026-06-03T19:08:38.418Z | [manifest (local archive)](../README.md) |
| nemotron-3.5-lightning (thinking off, NVFP4) | Local RTX 5090 | 16 | 2026-08-12T01:14:17.651Z | 2026-08-13T16:18:48.944Z | [manifest (local archive)](../README.md) |
| nemotron-3.5-lightning (thinking on, NVFP4) | Local RTX 5090 | 30 | 2026-08-12T01:19:46.837Z | 2026-08-13T16:19:11.831Z | [manifest (local archive)](../README.md) |
| poolside/laguna-s-2.1 (thinking off) | OpenRouter | 29 | 2026-07-22T22:30:20.666Z | 2026-08-06T16:00:27.484Z | [manifest (local archive)](../README.md) |
| qwen3.6-27b (thinking off) | Baseten | 30 | 2026-07-28T11:20:17.425Z | 2026-08-06T15:49:20.178Z | [manifest (local archive)](../README.md) |
| qwen3.6-35b-a3b (thinking off, FP8) | Baseten | 30 | 2026-07-28T15:00:09.163Z | 2026-08-06T16:06:27.186Z | [manifest (local archive)](../README.md) |
| qwen3.8-27b (thinking off, FP8) | Baseten | 30 | 2026-08-16T19:19:44.745Z | 2026-08-17T00:06:42.910Z | [manifest (local archive)](../README.md) |
| qwen3.8-27b (thinking off, NVFP4) | Local RTX 5090 | 30 | 2026-08-16T04:46:11.786Z | 2026-08-16T12:32:02.152Z | [manifest (local archive)](../README.md) |
| GPT-Live prompt-optimized + Terra medium* | speech profile | 25 | 20260916T123039 | 20260916T141349 | [manifest (local archive)](../README.md) |
| GPT-Live standard + Terra medium | speech profile | 25 | 20260915T222012 | 20260915T225824 | [manifest (local archive)](../README.md) |
| GPT-Realtime-1.5 | speech profile | 25 | 20260915T133854 | 20260915T152001 | [manifest (local archive)](../README.md) |
| GPT-Realtime-2.1 low | speech profile | 25 | 20260915T133854 | 20260915T151759 | [manifest (local archive)](../README.md) |
| Gemini 3.1 minimal | speech profile | 25 | 20260915T133854 | 20260916T140645 | [manifest (local archive)](../README.md) |
| Gemini 3.8 Extended Thinking low | speech profile | 25 | 20260915T133854 | 20260916T140645 | [manifest (local archive)](../README.md) |
| Gemini 3.8 Extended Thinking medium (interaction-aware) | speech profile | 25 | 20260916T073851 | 20260916T141545 | [manifest (local archive)](../README.md) |
| Gemini 3.8 Live (regular; default) | speech profile | 25 | 20260916T145116 | 20260916T163537 | [manifest (local archive)](../README.md) |
| Grok Voice Think Fast | speech profile | 25 | 20260915T133854 | 20260915T141449 | [manifest (local archive)](../README.md) |

## Historical preservation and reproduction

The [previous README (local archive)](../README.md) and [previous standalone leaderboard (local archive)](../README.md) preserve publication context; their relative links are rooted at the repository. The September speech package, source archives, raw captures, old judges, and earlier campaign reports were not rewritten. New scores are separate artifacts. Raw SDK responses and cleaned inputs are local-only under `runs/judge-refresh-opus55-20261017`; committed results and manifests preserve hashes and structured judgments.

Run from the repository root with the existing environment:

```bash
.venv/bin/python ops/judge-refresh-opus55-20261017/campaign.py status
.venv/bin/python ops/judge-refresh-opus55-20261017/finalize.py
```

Reproduction of metrics requires the local source captures and frozen snapshots. Resuming judging validates matching input, rubric, model, effort, and all 30 score records before reusing a result. Publication requires every selected input to have a validated grade or an explicit provider-blocked disposition. The aggregate records processing_complete separately from complete/all_quality_complete; the latter remain false while any selected grade is unavailable. Original archives are read without mutation.

[Evidence storage and local audit instructions](../README.md).
