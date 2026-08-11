# Gemma 4 31B BaseTen SGLang campaign

The frozen no-filler, thinking-off campaign scored
872/900 strict turns (96.9%,
conversation-cluster bootstrap 95% CI 95.8–97.9%).

| Metric | Result |
|---|---:|
| Canonical conversations | 30 |
| Full 30-turn conversations | 30/30 |
| Strict pass | 872/900 (96.9%) |
| Tool error | 3.1% |
| Instruction error | 3.1% |
| KB error | 0.0% |
| TTFAT P50 / P95 / max | 430 / 564 / 3718 ms |
| Thinking tokens | 0 |

README row:

| Model | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gemma-4-31b-it (thinking off) | 96.9% | 3.1% | 3.1% | 3.1% | 0.0% | 430ms | 564ms | 3718ms | BaseTen |

Most error-prone scripted turns:

| Turn | Conversations with any error | Error rate |
|---:|---:|---:|
| 12 | 19/30 | 63.3% |
| 15 | 6/30 | 20.0% |
| 14 | 1/30 | 3.3% |
| 17 | 1/30 | 3.3% |
| 24 | 1/30 | 3.3% |

Missing future turns after a model-caused early exit count as failures in all
four displayed accuracy measures. Latency is reported only where a scripted
turn produced a measured response. Whole conversations, not individual turns,
are the bootstrap resampling unit.
