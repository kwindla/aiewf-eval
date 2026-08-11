# Gemma 4 31B BaseTen pooled N=150 campaign

The pooled canonical no-filler, thinking-off cohort scored
4346/4500 strict turns (96.58%,
conversation-cluster bootstrap 95% CI 96.13–97.02%).

| Cohort | Conversations | Strict turns | Pass rate | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Original | 30 | 872/900 | 96.89% | 95.78–97.89% |
| Extension | 120 | 3474/3600 | 96.50% | 96.00–96.97% |
| Pooled | 150 | 4346/4500 | 96.58% | 96.13–97.02% |

| Metric | Pooled result |
|---|---:|
| Full 30-turn conversations | 150/150 |
| Any error | 3.42% |
| Tool error | 3.33% |
| Instruction error | 3.40% |
| KB error | 0.02% |
| TTFAT P50 / P95 / max | 490 / 718 / 38250 ms |
| Thinking tokens | 0 |

Latency by cohort (observed scripted turns):

| Cohort | Turns | P50 ms | P95 ms | Max ms | >10s |
|---|---:|---:|---:|---:|---:|
| Original | 900 | 430 | 564 | 3718 | 0 |
| Extension | 3600 | 495 | 778 | 38250 | 27 |
| Pooled | 4500 | 490 | 718 | 38250 | 27 |

Most error-prone scripted turns:

| Turn | Conversations with any error | Error rate | Tool | Instruction | KB |
|---:|---:|---:|---:|---:|---:|
| 12 | 105/150 | 70.0% | 105 | 105 | 0 |
| 15 | 38/150 | 25.3% | 38 | 38 | 0 |
| 17 | 4/150 | 2.7% | 3 | 4 | 0 |
| 14 | 2/150 | 1.3% | 2 | 2 | 0 |
| 16 | 2/150 | 1.3% | 0 | 2 | 0 |
| 24 | 2/150 | 1.3% | 2 | 2 | 0 |
| 13 | 1/150 | 0.7% | 0 | 0 | 1 |

Missing future turns after a model-caused early exit count as failures in all
accuracy measures. Latency is reported only where a scripted turn produced a
measured response. Whole conversations, not turns, are the bootstrap unit
(20,000 deterministic resamples).
