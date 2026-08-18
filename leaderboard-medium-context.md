# AIEWF Medium-Context Text Leaderboard

Text-mode results for `aiwf_medium_context`, sorted by strict turn pass rate and
then TTFAT P50. Each canonical conversation has 30 scripted turns. TTFAT is
measured from request start to the first user-visible answer token or tool-call
output; separately streamed reasoning deltas do not stop the timer. Selected
historically important self-hosted results remain in the table with the provider
that served the published runs. The reporting convention excludes each
conversation's first scripted response; historical exceptions are noted below.

| Model | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nemotron-3-ultra (128) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 541ms | 712ms | 1302ms | Baseten |
| claude-sonnet-4-6 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 850ms | 4126ms | 9396ms | Anthropic |
| claude-fable-5 (low) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3535ms | 5148ms | 8815ms | Anthropic |
| claude-fable-5 (default) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3956ms | 6496ms | 13602ms | Anthropic |
| glm-5.2 (none) | 99.7% | 0.3% | 0.2% | 0.2% | 0.0% | 936ms | 2140ms | 7567ms | Baseten |
| nemotron-3-ultra (96) | 98.3% | 1.7% | 1.3% | 1.3% | 0.3% | 529ms | 655ms | 1259ms | Baseten |
| kimi-k2.6 (thinking on) | 98.3% | 1.7% | 1.4% | 1.7% | 0.0% | 1560ms | 5404ms | 13622ms | Baseten |
| qwen3.8-27b (thinking off, FP8) | 98.2% | 1.8% | 1.8% | 1.8% | 0.0% | 649ms | 801ms | 2161ms | Baseten |
| claude-haiku-4-5 | 98.0% | 2.0% | 0.7% | 2.0% | 0.0% | 637ms | 1615ms | 3152ms | Anthropic |
| gpt-5.1 | 98.0% | 2.0% | 2.0% | 2.0% | 0.0% | 739ms | 1492ms | 4244ms | OpenAI |
| qwen3.8-27b (thinking off, NVFP4) | 97.8% | 2.2% | 1.9% | 2.2% | 0.1% | 101ms | 318ms | 592ms | Local RTX 5090 |
| gpt-5.6-terra (medium) | 97.8% | 2.2% | 1.9% | 2.2% | 0.0% | 927ms | 2149ms | 4167ms | OpenAI |
| gpt-5.5 (none) | 97.4% | 2.6% | 2.0% | 2.6% | 0.0% | 875ms | 2177ms | 5623ms | OpenAI |
| qwen3.6-27b (thinking off) | 97.3% | 2.7% | 2.7% | 2.7% | 0.0% | 667ms | 769ms | 1920ms | Baseten |
| deepseek-v4-pro-0813 (low) | 97.3% | 2.7% | 2.7% | 2.6% | 2.0% | 752ms | 1477ms | 3545ms | Baseten |
| gemini-3.6-flash (minimal) | 97.1% | 2.9% | 2.4% | 2.8% | 0.1% | 798ms | 984ms | 1472ms | AI Studio |
| nemotron-3-super-120b (512) | 97.0% | 3.0% | 1.0% | 3.0% | 0.3% | 687ms | 1210ms | 2254ms | Baseten |
| gpt-5.4 (low) | 97.0% | 3.0% | 3.0% | 3.0% | 0.0% | 782ms | 1706ms | 2698ms | OpenAI |
| deepseek-v4-flash-0731 (low) | 96.7% | 3.3% | 2.8% | 3.2% | 0.7% | 677ms | 1452ms | 4687ms | Baseten |
| gemma-4-31b-it (thinking off) | 96.6% | 3.4% | 3.3% | 3.4% | 0.0% | 489ms | 609ms | 38250ms | Baseten |
| gpt-5.6-sol (none) | 96.6% | 3.4% | 3.3% | 3.3% | 0.1% | 1098ms | 2625ms | 6344ms | OpenAI |
| gpt-4.1 | 96.3% | 3.7% | 3.7% | 3.3% | 0.0% | 536ms | 1771ms | 5056ms | OpenAI |
| gpt-5.4 (none, +96 dots) | 95.2% | 4.8% | 4.7% | 4.6% | 0.1% | 694ms | 2273ms | 17264ms | OpenAI |
| inkling (none) | 94.8% | 5.2% | 5.1% | 4.8% | 1.3% | 447ms | 727ms | 1813ms | Baseten |
| gpt-4o | 94.7% | 5.3% | 3.0% | 5.0% | 0.3% | 546ms | 1369ms | 4897ms | OpenAI |
| kimi-k2.6 (thinking off) | 93.9% | 6.1% | 6.0% | 3.9% | 0.0% | 475ms | 842ms | 4458ms | Baseten |
| deepseek-v4-flash-0731 (high) | 93.9% | 6.1% | 5.9% | 6.1% | 4.8% | 763ms | 1871ms | 8702ms | Baseten |
| nemotron-3.5-lightning (thinking on, NVFP4) | 93.6% | 6.4% | 2.8% | 5.7% | 0.9% | 1464ms | 5787ms | 29869ms | Local RTX 5090 |
| gemini-3.5-flash (minimal) | 93.3% | 6.7% | 5.3% | 6.7% | 4.9% | 892ms | 1183ms | 1721ms | AI Studio |
| claude-sonnet-5 | 93.0% | 7.0% | 7.0% | 7.0% | 0.0% | 1204ms | 2465ms | 6955ms | Anthropic |
| qwen3.6-35b-a3b (thinking off, FP8) | 91.6% | 8.4% | 6.8% | 7.8% | 0.4% | 764ms | 1233ms | 35664ms | Baseten |
| gpt-5.6-terra (none) | 91.3% | 8.7% | 8.1% | 8.6% | 0.3% | 621ms | 1870ms | 5665ms | OpenAI |
| nemotron-3-nano-30b (512) | 90.6% | 9.4% | 5.0% | 6.1% | 4.0% | 940ms | 1912ms | 2821ms | Baseten |
| gpt-5.4 (none) | 90.2% | 9.8% | 9.4% | 9.7% | 0.1% | 689ms | 1723ms | 6571ms | OpenAI |
| gemini-2.5-flash (thinking off) | 89.9% | 10.1% | 9.1% | 10.1% | 0.0% | 550ms | 850ms | 2352ms | AI Studio |
| gpt-5.2 | 89.3% | 10.7% | 10.0% | 10.7% | 0.7% | 624ms | 1171ms | 2509ms | OpenAI |
| gpt-5.6-luna (none) | 88.3% | 11.7% | 11.7% | 11.7% | 0.0% | 671ms | 2304ms | 12017ms | OpenAI |
| gpt-oss-120b (groq) | 86.3% | 13.7% | 9.3% | 13.0% | 0.7% | 98ms | 217ms | 2117ms | Groq |
| muse-glimmer-30b (thinking low, GGUF) | 86.1% | 13.9% | 13.0% | 13.7% | 0.0% | 231ms | 1752ms | 5474ms | Local RTX 5090 |
| poolside/laguna-s-2.1 (thinking off) | 85.6% | 14.4% | 13.7% | 11.2% | 5.7% | 295ms | 620ms | 21032ms | OpenRouter |
| gpt-4.1-mini | 85.3% | 14.7% | 14.7% | 14.7% | 0.0% | 851ms | 2135ms | 5945ms | OpenAI |
| gpt-5-mini | 83.7% | 16.3% | 14.0% | 16.3% | 1.0% | 682ms | 1132ms | 1904ms | OpenAI |
| gpt-4o-mini | 82.7% | 17.3% | 10.3% | 13.7% | 2.3% | 553ms | 1947ms | 6497ms | OpenAI |
| gemma-4-26b-a4b-it (thinking off) | 80.7% | 19.3% | 13.9% | 19.3% | 0.9% | 578ms | 634ms | 31574ms | Baseten |
| gemini-3.5-flash-lite (minimal) | 68.6% | 31.4% | 30.8% | 31.4% | 28.1% | 591ms | 679ms | 928ms | AI Studio |
| nemotron-3.5-lightning (thinking off, NVFP4) | 50.9% | 49.1% | 49.0% | 47.9% | 38.9% | 62ms | 70ms | 80ms | Local RTX 5090 |

Muse Glimmer's row is the selected low-strength arm of the uncapped N=30 sweep
documented in [`runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md`](runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md).

The two Nemotron 3.5 Lightning rows are the official NVFP4 checkpoint on one RTX
5090, with N=30 per native thinking setting. See the
[`campaign report`](ops/local-nemotron35-lightning-sglang/aiewf-medium-binary-n30-20260811/artifacts/analysis/REPORT.md).

The four Nemotron rows are historical measurements from Baseten deployments that
no longer exist in their original form. They remain here because the results are
useful benchmark comparisons; `Baseten` is the provider that served those runs.
Their latency values are legacy TTFT summaries retained verbatim. The original
commits did not preserve exact canonical manifests, so they cannot be converted
reliably to content-aware, first-response-excluded TTFAT and should not be treated
as latency-comparable with the corrected canonical campaigns.

The README contains serving notes, cohort qualifications, historical-route
context, and the speech-to-speech leaderboard.
