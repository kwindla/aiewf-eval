# Gemma 4 31B local NVFP4 + BF16-KV campaign

The frozen batch-one BF16-KV cohort scored 863/900
strict turns (95.9%, conversation-cluster bootstrap 95%
CI 94.9–96.8%).

| Configuration | Strict pass | 95% CI | Tool error | Instruction error | KB error | TTFAT P50 | TTFAT P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BaseTen BF16 weights/KV + MTP, 2xH100 | 96.9% | 95.8–97.9% | 3.1% | 3.1% | 0.0% | 430ms | 564ms |
| Local NVFP4 weights + FP8 KV, RTX 5090 | 94.7% | 93.4–95.9% | 5.3% | 5.3% | 0.0% | 102ms | 295ms |
| Local NVFP4 weights + BF16 KV, RTX 5090 | 95.9% | 94.9–96.8% | 4.0% | 4.1% | 0.0% | 125ms | 326ms |

Local BF16 KV minus local FP8 KV is +1.2 percentage points
(independent conversation-cluster bootstrap 95% CI -0.3 to
+2.8). This is the cleanest available estimate of KV-cache
precision impact: weights, hardware, SGLang image, sampling, batch size, and MTP
setting are held fixed. The compact BF16 arm necessarily uses smaller,
asymmetric static KV pools.

Local BF16 KV minus BaseTen BF16 weights/KV + MTP is -1.0 points
(95% CI -2.3 to +0.4). This remains an
end-to-end deployment comparison because weight precision, MTP, hardware, and
SGLang version differ.

| Turn | BaseTen BF16 + MTP errors | Local FP8-KV errors | Local BF16-KV errors |
|---:|---:|---:|---:|
| 12 | 19/30 | 21/30 | 17/30 |
| 14 | 1/30 | 0/30 | 0/30 |
| 15 | 6/30 | 22/30 | 16/30 |
| 16 | 0/30 | 0/30 | 1/30 |
| 17 | 1/30 | 2/30 | 1/30 |
| 24 | 1/30 | 3/30 | 2/30 |

Missing future turns after a model-caused early exit count as failures. Latency
uses observed scripted turns only. Whole conversations, not individual turns,
are the bootstrap unit.
