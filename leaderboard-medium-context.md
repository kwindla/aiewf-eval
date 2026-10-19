# aiwf_medium_context text leaderboard

Text mode models:

The standalone copy is [leaderboard-medium-context.md](leaderboard-medium-context.md).

![Text-model pass rates and recalculated TTFAT P50/P95](docs/text-model-voice-readiness.svg)

Rows are ordered by available pass rate descending, then TTFAT P50 ascending. Rows with unavailable quality appear last; N counts selected complete conversations.

| Model | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| muse-spark-1.3 (low) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2965ms | 9143ms | 34777ms | OpenRouter / Meta | 30 |
| gemini-3.7-flash (low) | 99.9% | 0.1% | 0.1% | 0.0% | 0.0% | 821ms | 1656ms | 15353ms | AI Studio | 30 |
| claude-sonnet-5-5 (low) | 99.9% | 0.1% | 0.0% | 0.0% | 0.1% | 838ms | 2140ms | 4546ms | Anthropic | 30 |
| gpt-5.6-terra (medium) | 99.9% | 0.1% | 0.0% | 0.1% | 0.0% | 1004ms | 1569ms | 4177ms | OpenAI | 26 |
| claude-sonnet-5-5 (no up-front thinking, low) | 99.9% | 0.1% | 0.0% | 0.0% | 0.1% | 1180ms | 2195ms | 8186ms | Anthropic | 30 |
| gemini-3.8-flash (low) | 99.9% | 0.1% | 0.0% | 0.0% | 0.1% | 1411ms | 1967ms | 7588ms | AI Studio | 30 |
| muse-spark-1.3 (minimal) | 99.6% | 0.4% | 0.1% | 0.3% | 0.0% | 1752ms | 5638ms | 40044ms | OpenRouter / Meta | 30 |
| kimi-k2.6 (thinking on) | 98.8% | 1.2% | 0.9% | 0.1% | 0.2% | 1560ms | 5393ms | 13622ms | Baseten | 30 |
| deepseek-v4-flash-0731 (high) | 98.6% | 1.4% | 0.4% | 0.9% | 0.8% | 773ms | 1970ms | 8702ms | Baseten | 30 |
| deepseek-v4-pro-0813 (low) | 98.4% | 1.6% | 0.1% | 1.1% | 1.2% | 755ms | 1474ms | 4506ms | Baseten | 30 |
| gpt-5.5 (none) | 98.2% | 1.8% | 1.8% | 1.8% | 0.0% | 780ms | 1173ms | 2870ms | OpenAI | 29 |
| gpt-5.4 (low) | 97.7% | 2.3% | 2.0% | 2.1% | 0.0% | 932ms | 1458ms | 5712ms | OpenAI | 30 |
| glm-5.3 (low reasoning) | 97.7% | 2.3% | 1.0% | 1.0% | 0.7% | 2376ms | 4757ms | 8007ms | Z.AI | 30 |
| deepseek-v4-flash-0731 (low) | 97.3% | 2.7% | 1.7% | 2.1% | 0.8% | 689ms | 1648ms | 17531ms | Baseten | 30 |
| claude-haiku-5-5 (low) | 97.2% | 2.8% | 2.2% | 1.3% | 0.2% | 651ms | 1563ms | 3717ms | Anthropic | 30 |
| gpt-5.6-sol (none) | 96.9% | 3.1% | 3.1% | 3.1% | 0.0% | 765ms | 1298ms | 36309ms | OpenAI | 28 |
| muse-spark-open (low, NVFP4) | 96.7% | 3.3% | 1.9% | 2.3% | 1.1% | 626ms | 1299ms | 3273ms | Baseten | 30 |
| gemma-4-31b-it (thinking off) | 96.4% | 3.6% | 3.3% | 3.6% | 0.0% | 497ms | 629ms | 33254ms | Baseten | 30 |
| gpt-5.4 (none, +96 dots) | 96.4% | 3.6% | 3.3% | 3.6% | 0.0% | 813ms | 1234ms | 4180ms | OpenAI | 30 |
| qwen3.8-27b (thinking off, FP8) | 96.0% | 4.0% | 1.6% | 3.7% | 0.6% | 649ms | 801ms | 2161ms | Baseten | 30 |
| gpt-4.1 | 95.9% | 4.1% | 3.5% | 4.1% | 0.0% | 743ms | 1086ms | 4655ms | OpenAI | 26 |
| glm-5.2 (none) | 95.8% | 4.2% | 0.1% | 2.3% | 2.1% | 938ms | 2215ms | 7567ms | Baseten | 30 |
| qwen3.8-27b (thinking off, NVFP4) | 95.1% | 4.9% | 1.9% | 4.7% | 0.4% | 101ms | 318ms | 592ms | Local RTX 5090 | 30 |
| gemini-3.6-flash (minimal) | 95.1% | 4.9% | 2.3% | 4.1% | 1.4% | 808ms | 1078ms | 2229ms | AI Studio | 30 |
| claude-haiku-5-5 (thinking off, low) | 94.7% | 5.3% | 3.7% | 5.2% | 0.2% | 556ms | 819ms | 2688ms | Anthropic | 29 |
| gemini-3.5-flash (minimal) | 94.7% | 5.3% | 0.4% | 5.2% | 2.0% | 898ms | 1256ms | 2076ms | AI Studio | 30 |
| inkling (none) | 94.4% | 5.6% | 2.6% | 4.4% | 1.6% | 443ms | 658ms | 1158ms | Baseten | 30 |
| gemini-3.5-flash-lite (minimal) | 94.3% | 5.7% | 4.1% | 4.3% | 1.1% | 617ms | 1145ms | 1938ms | AI Studio | 30 |
| gpt-5.1 | 93.5% | 6.5% | 2.0% | 5.2% | 1.9% | 768ms | 1493ms | 5561ms | OpenAI | 25 |
| gpt-5.4 (none) | 93.0% | 7.0% | 6.6% | 7.0% | 0.0% | 810ms | 1257ms | 3669ms | OpenAI | 30 |
| gpt-5.6-terra (none) | 92.7% | 7.3% | 7.1% | 7.3% | 0.0% | 826ms | 1257ms | 2706ms | OpenAI | 26 |
| qwen3.6-27b (thinking off) | 92.1% | 7.9% | 1.8% | 6.0% | 1.0% | 665ms | 765ms | 1920ms | Baseten | 30 |
| gpt-6-luna (low) | 91.8% | 8.2% | 4.5% | 7.2% | 0.1% | 917ms | 1439ms | 3585ms | OpenAI | 29 |
| nemotron-3-ultra (thinking on) | 91.8% | 8.2% | 0.1% | 4.2% | 5.0% | 926ms | 2116ms | 8679ms | Baseten | 30 |
| nemotron-3-super-120b (512, NVFP4) | 90.9% | 9.1% | 1.2% | 7.4% | 3.9% | 669ms | 1383ms | 2881ms | AWS B200 | 30 |
| kimi-k2.6 (thinking off) | 88.7% | 11.3% | 6.2% | 10.4% | 3.6% | 481ms | 850ms | 4458ms | Baseten | 30 |
| gpt-4o | 88.2% | 11.8% | 3.3% | 11.8% | 1.8% | 556ms | 1319ms | 5313ms | OpenAI | 11 |
| gpt-5.2 | 87.3% | 12.7% | 10.3% | 12.1% | 0.6% | 648ms | 1192ms | 7647ms | OpenAI | 11 |
| muse-glimmer-30b (thinking low, GGUF) | 85.1% | 14.9% | 12.7% | 14.9% | 0.6% | 225ms | 1743ms | 4792ms | Local RTX 5090 | 30 |
| gpt-5.6-luna (none) | 84.4% | 15.6% | 11.0% | 15.6% | 0.4% | 708ms | 1198ms | 4546ms | OpenAI | 27 |
| nemotron-3.5-lightning (thinking on, NVFP4) | 83.9% | 16.1% | 2.0% | 7.8% | 10.3% | 1470ms | 5927ms | 29869ms | Local RTX 5090 | 30 |
| gemma-4-26b-a4b-it (thinking off) | 83.4% | 16.6% | 13.4% | 16.6% | 0.1% | 578ms | 680ms | 31574ms | Baseten | 30 |
| gpt-4o-mini | 83.0% | 17.0% | 10.3% | 13.9% | 4.8% | 603ms | 3367ms | 16315ms | OpenAI | 11 |
| qwen3.6-35b-a3b (thinking off, FP8) | 82.6% | 17.4% | 5.6% | 15.7% | 3.7% | 762ms | 1134ms | 14389ms | Baseten | 30 |
| gpt-4.1-mini | 81.8% | 18.2% | 13.1% | 18.2% | 0.8% | 638ms | 1010ms | 1704ms | OpenAI | 13 |
| gemini-2.5-flash (thinking off) | 81.4% | 18.6% | 9.2% | 18.3% | 2.8% | 545ms | 824ms | 2352ms | AI Studio | 30 |
| gpt-5-mini | 80.9% | 19.1% | 13.9% | 18.8% | 0.9% | 690ms | 1191ms | 2269ms | OpenAI | 11 |
| poolside/laguna-s-2.1 (thinking off) | 79.3% | 20.7% | 9.7% | 16.1% | 7.2% | 296ms | 600ms | 3140ms | OpenRouter | 29 |
| nemotron-3-nano-30b (512, NVFP4) | 71.9% | 28.1% | 7.6% | 13.3% | 18.6% | 934ms | 1436ms | 1538ms | AWS B200 | 7 |
| nemotron-3.5-lightning (thinking off, NVFP4) | 69.6% | 30.4% | 12.7% | 22.3% | 10.6% | 63ms | 70ms | 80ms | Local RTX 5090 | 16 |

Every new grade uses **Opus 5.5 at medium effort**, with the same frozen text rubric. **— means full-cohort quality is unavailable because the provider rejected at least one selected input**; those rows remain visible with recalculated latency. Rejected judge requests are neither model failures nor passing grades, and their conversations are not dropped to manufacture a score. **N is complete conversations**, with 30 scripted turns each. The historical refresh selected the most recent matching complete conversations, up to 30 per configuration. Fresh Ultra, the October 18 new-model cohorts, and the Responses adapter refresh are prospective; their linked reports give assignment and completion policies. 17 ranked rows have fewer than 30; the [cohort audit](docs/text-cohort-sizes-2026-10-17.md) explains the shortfalls. Selection does not require correct tools or a passing grade.

Accuracy is **conditional on completion**. Incomplete conversations and their observed prefixes are excluded, so these results differ from earlier attempt-based, fixed-900-turn studies. They measure quality among completed conversations and do not establish completion reliability. Any Error is the complement of Pass Rate; component errors overlap. The [refresh report](docs/model-evaluations-2026-10/judge-refresh-opus55-20261017/REPORT.md) includes exact counts, conversation-cluster confidence intervals, selected source dates, and configuration mappings.

Judge inputs exclude delimited `<think>` reasoning, including analysis before a closing `</think>` whose opening tag was supplied by the chat template. Final answers and actual tool calls are evaluated; raw captures are preserved. The [cleanup summary](docs/model-evaluations-2026-10/judge-refresh-opus55-20261017/reasoning-cleanup/manifest.json) describes the normalization; the full per-input audit is preserved in the [local evidence archive](docs/model-evaluations-2026-10/README.md). Historical latency measurements are unchanged; leaked reasoning in content may have started their timer before the final answer.

TTFAT is the recorded service time to the first user-visible answer token or tool-call output. Each conversation's first scripted response is excluded; later scripted and recovery responses are included. P95 uses linear interpolation. All latency cells were recalculated from the selected captures. Local GPU measurements use localhost and omit provider network time; capture dates, SDK versions, hardware, caching, and sampling settings differ across rows. OpenAI cohorts retain recorded service-tier/SDK variation; older unlogged tiers are not inferred as defaults. The ~700ms line is a voice-use guideline, not a statistical threshold or a measured end-to-end speech latency.

The published Nemotron 3 Ultra row uses **30 fresh thinking-on conversations on Baseten**, with `reasoning_effort=high`, temperature 1.0, top-p 0.95, and max_tokens 16384; all 30 completed. No numeric thinking budget was requested. This replaces the older Modal 128/96 rows. Thinking-off is omitted from the current leaderboard; the completed comparison, original grades, and early-ending runs remain in the [Ultra study](docs/model-evaluations-2026-10/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018/REPORT.md). Three-second request pacing occurs before latency measurement. Super/Nano 512 retain their mapped **NVFP4 on AWS B200** cohorts. The [failure audit](docs/model-evaluations-2026-10/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018/failure-analysis/REPORT.md) and [same-output rubric comparison](docs/model-evaluations-2026-10/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018/rubric-comparison/REPORT.md) explain the current score. The [earlier configuration inventory](docs/model-evaluations-2026-10/judge-refresh-opus55-20261017/configurations.json) and [source audit](docs/model-evaluations-2026-10/judge-refresh-opus55-20261017/SOURCE-AUDIT.md) preserve the retired Ultra cohorts; they are historical evidence.

Reasoning, filler, precision, and serving configurations are kept separate. Gemini minimal/low labels retain their recorded thinking settings; Gemini 2.5 explicitly uses budget zero. Kimi on/off retain their different vendor-mode samplers. Muse uses its recorded GGUF low-strength configuration; Lightning and local Qwen use NVFP4. The manifests preserve available request settings and endpoint origins; missing settings are not inferred as defaults.

Earlier scores, denominators, serving details and study links are preserved in the [local evidence archive](docs/model-evaluations-2026-10/README.md), under `ops/judge-refresh-opus55-20261017/historical/README.md`. The [October latency audit](docs/ttfat-reanalysis-2026-10/REPORT.md) and original campaign reports remain historical evidence. The requested Groq, Grok 4.6 text, Fable, Sonnet 4.6, Haiku 4.5, and Sonnet 5 removals remain in the [row archive](docs/text-rows-archived-2026-10-17.md) and [Groq archive](docs/groq-gpt-oss-120b-archived-2026-10-17.md).

<!-- recent-models-20261018:start -->
The [October 18 new-model campaign](docs/model-evaluations-2026-10/recent-models-20261018/REPORT.md) adds GPT-6 Luna, Sonnet/Haiku 5.5, Muse Spark 1.3, and Muse Spark Open at the approved lowest reasoning settings. Each text configuration has 30 assigned conversations. The table reports completed-conversation quality and latency; the campaign report also gives completion counts and fixed-denominator scores, retaining premature endings. Sonnet’s “no up-front thinking” uses `between_tools`, with low effort; it can still emit tool progress updates. Spark 1.3 is routed through OpenRouter pinned to Meta; Spark Open uses Baseten NVFP4. OpenAI requested Priority; direct probes reported effective Fast service.
<!-- recent-models-20261018:end -->

<!-- luna-responses-integration-20261018:start -->
The earlier GPT-6 Luna captures used a defective Responses history adapter. Current affected OpenAI rows now use the [repaired adapter refresh](docs/model-evaluations-2026-10/responses-repair-20261018/REPORT.md); the [original investigation](docs/model-evaluations-2026-10/luna-completion-investigation-20261018/REPORT.md) remains available. The repair preserves native phases and reasoning, but early model endings can still occur.
<!-- luna-responses-integration-20261018:end -->

<!-- responses-repair-20261018:start -->
The [Responses adapter refresh](docs/model-evaluations-2026-10/responses-repair-20261018/REPORT.md) replaces twelve affected configurations using 30 fresh assignments each, Opus 5.5 Medium judging, and recalculated latency. Ordinary conversations use cloud-linked native history; the +96 dots arm replays native outputs with earlier user text filler-free. These are independent prospective cohorts, explicitly requesting Priority; older tiers varied or were unlogged. The table remains conditional on completion. All early endings and missing future turns are retained in the report’s fixed-900 scores.

| Configuration | Complete / assigned | Fixed-900 pass |
|---|---:|---:|
| gpt-4.1-mini | 13/30 | 76.4% |
| gpt-4.1 | 26/30 | 95.6% |
| gpt-5.4 (low) | 30/30 | 97.7% |
| gpt-5.4 (none, +96 dots) | 30/30 | 96.4% |
| gpt-5.4 (none) | 30/30 | 93.0% |
| gpt-5.5 (none) | 29/30 | 97.8% |
| gpt-5.6-luna (none) | 27/30 | 81.7% |
| gpt-5.6-sol (none) | 28/30 | 92.4% |
| gpt-5.6-terra (medium) | 26/30 | 99.0% |
| gpt-5.6-terra (none) | 26/30 | 87.2% |
| gpt-6-luna (none) | 0/30 | 25.0% |
| gpt-6-luna (low) | 29/30 | 91.8% |

Zero-completion configurations are unranked; completed-only quality and latency are unavailable.
<!-- responses-repair-20261018:end -->

<!-- recovery-context-20261019:start -->
The [recovery-context migration](docs/model-evaluations-2026-10/judge-recovery-context-20261019/REPORT.md) retains all public recovery exchanges and recorded tool outcomes as unscored chronological evidence. Opus 5.5 Medium rejudged all 698 affected active assignments; 988 byte-identical requests reuse their verified prior grades. Scoring still uses only the original scripted turns, and later recovery does not erase an earlier unnecessary deferral. The 24-conversation text/voice validation used independent Astra/Fable reference labels and two diagnostic repetitions; these are model reviews, not human ground truth or a population accuracy estimate. Generation cohorts, completion counts, latency and audio evidence are unchanged. Current exact scores and confidence intervals are in the [reconciled aggregate](docs/model-evaluations-2026-10/judge-recovery-context-20261019/summary.json); earlier linked studies preserve historical grades.
<!-- recovery-context-20261019:end -->
