# AIEWF Medium-Context Text Leaderboard

Current-production text-mode results for `aiwf_medium_context`, sorted by strict
turn pass rate. Each canonical conversation has 30 scripted turns. TTFAT is
measured from request start to the first user-visible answer token or tool-call
output; separately streamed reasoning deltas do not stop the timer.

| Model | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| claude-sonnet-4-6 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 850ms | 4126ms | 9396ms | Anthropic |
| claude-fable-5 (low) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3535ms | 5148ms | 8815ms | Anthropic |
| claude-fable-5 (default) | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3956ms | 6496ms | 13602ms | Anthropic |
| glm-5.2 (none) | 99.7% | 0.3% | 0.2% | 0.2% | 0.0% | 936ms | 2140ms | 7567ms | BaseTen |
| kimi-k2.6 (thinking on) | 98.3% | 1.7% | 1.4% | 1.7% | 0.0% | 1596ms | 5252ms | 13622ms | BaseTen |
| claude-haiku-4-5 | 98.0% | 2.0% | 0.7% | 2.0% | 0.0% | 637ms | 1615ms | 3152ms | Anthropic |
| gpt-5.1 | 98.0% | 2.0% | 2.0% | 2.0% | 0.0% | 739ms | 1492ms | 4244ms | OpenAI |
| gpt-5.6-terra (medium) | 97.8% | 2.2% | 1.9% | 2.2% | 0.0% | 927ms | 2149ms | 4167ms | OpenAI |
| gpt-5.5 (none) | 97.4% | 2.6% | 2.0% | 2.6% | 0.0% | 875ms | 2177ms | 5623ms | OpenAI |
| qwen3.6-27b (thinking off) | 97.3% | 2.7% | 2.7% | 2.7% | 0.0% | 668ms | 822ms | 5429ms | BaseTen |
| gemini-3.6-flash (minimal) | 97.1% | 2.9% | 2.4% | 2.8% | 0.1% | 798ms | 984ms | 1472ms | AI Studio |
| gpt-5.4 (low) | 97.0% | 3.0% | 3.0% | 3.0% | 0.0% | 782ms | 1706ms | 2698ms | OpenAI |
| gemma-4-31b-it (thinking off) | 96.6% | 3.4% | 3.3% | 3.4% | 0.0% | 490ms | 718ms | 38250ms | BaseTen |
| gpt-5.6-sol (none) | 96.6% | 3.4% | 3.3% | 3.3% | 0.1% | 1098ms | 2625ms | 6344ms | OpenAI |
| gpt-4.1 | 96.3% | 3.7% | 3.7% | 3.3% | 0.0% | 536ms | 1771ms | 5056ms | OpenAI |
| gpt-5.4 (none, +96 dots) | 95.2% | 4.8% | 4.7% | 4.6% | 0.1% | 694ms | 2273ms | 17264ms | OpenAI |
| inkling (none) | 94.8% | 5.2% | 5.1% | 4.8% | 1.3% | 447ms | 727ms | 1813ms | BaseTen |
| gpt-4o | 94.7% | 5.3% | 3.0% | 5.0% | 0.3% | 546ms | 1369ms | 4897ms | OpenAI |
| kimi-k2.6 (thinking off) | 93.9% | 6.1% | 6.0% | 3.9% | 0.0% | 480ms | 854ms | 4458ms | BaseTen |
| gemini-3.5-flash (minimal) | 93.3% | 6.7% | 5.3% | 6.7% | 4.9% | 892ms | 1183ms | 1721ms | AI Studio |
| claude-sonnet-5 | 93.0% | 7.0% | 7.0% | 7.0% | 0.0% | 1204ms | 2465ms | 6955ms | Anthropic |
| qwen3.6-35b-a3b (thinking off, FP8) | 91.6% | 8.4% | 6.8% | 7.8% | 0.4% | 765ms | 1255ms | 35664ms | BaseTen |
| gpt-5.6-terra (none) | 91.3% | 8.7% | 8.1% | 8.6% | 0.3% | 621ms | 1870ms | 5665ms | OpenAI |
| gpt-5.4 (none) | 90.2% | 9.8% | 9.4% | 9.7% | 0.1% | 689ms | 1723ms | 6571ms | OpenAI |
| gemini-2.5-flash (thinking off) | 89.9% | 10.1% | 9.1% | 10.1% | 0.0% | 550ms | 850ms | 2352ms | AI Studio |
| gpt-5.2 | 89.3% | 10.7% | 10.0% | 10.7% | 0.7% | 624ms | 1171ms | 2509ms | OpenAI |
| gpt-5.6-luna (none) | 88.3% | 11.7% | 11.7% | 11.7% | 0.0% | 671ms | 2304ms | 12017ms | OpenAI |
| gpt-oss-120b (groq) | 86.3% | 13.7% | 9.3% | 13.0% | 0.7% | 98ms | 217ms | 2117ms | Groq |
| poolside/laguna-s-2.1 (thinking off) | 85.6% | 14.4% | 13.7% | 11.2% | 5.7% | 295ms | 620ms | 21032ms | OpenRouter |
| gpt-4.1-mini | 85.3% | 14.7% | 14.7% | 14.7% | 0.0% | 851ms | 2135ms | 5945ms | OpenAI |
| muse-glimmer-30b (thinking high, GGUF) | 84.9% | 15.1% | 14.0% | 14.0% | 0.3% | 232ms | 6488ms | 11586ms | Local RTX 5090 |
| gpt-5-mini | 83.7% | 16.3% | 14.0% | 16.3% | 1.0% | 682ms | 1132ms | 1904ms | OpenAI |
| gpt-4o-mini | 82.7% | 17.3% | 10.3% | 13.7% | 2.3% | 553ms | 1947ms | 6497ms | OpenAI |
| qwen3-8b (thinking off, BaseTen) | 81.3% | 18.7% | 15.7% | 15.9% | 3.4% | 564ms | 678ms | 1563ms | BaseTen |
| gemma-4-26b-a4b-it (thinking off) | 80.7% | 19.3% | 13.9% | 19.3% | 0.9% | 580ms | 803ms | 31574ms | BaseTen |
| gemini-3.5-flash-lite (minimal) | 68.6% | 31.4% | 30.8% | 31.4% | 28.1% | 591ms | 679ms | 928ms | AI Studio |

Muse Glimmer's row is the uncapped N=30 campaign documented in
[`runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/REPORT.md`](runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/REPORT.md).

The README contains serving notes, cohort qualifications, historical-route
context, and the speech-to-speech leaderboard.
