# Nemotron 3.5 Lightning AIEWF medium-context results

Each arm contains 30 canonical conversations and 900 fixed-denominator scripted turns. Missing future turns after a model-caused early exit fail all displayed accuracy criteria. Latency excludes the first scripted response of each conversation and is summarized over the remaining observed scripted and recovery responses.

| Mode | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Raw TTFT P50 | Full 30-turn conversations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| on-unbounded | 93.6% | 6.4% | 2.8% | 5.7% | 0.9% | 1464ms | 5786ms | 29869ms | 65ms | 30/30 |
| off | 50.9% | 49.1% | 49.0% | 47.9% | 38.9% | 62ms | 70ms | 80ms | 62ms | 13/30 |

The two request modes use NVIDIA's recommended temperature 1.0 and top-p 0.95, no output-token cap, and no thinking-budget cap. Both send `force_nonempty_content=true`; only `enable_thinking` changes.
