# Qwen3.8 27B FP8 Baseten H100 AIEWF medium-context none

Fixed denominator: 30 conversations × scripted turns 0–29 = 900. Recovery rows are excluded from judge coverage and accuracy, but included in the current-convention post-first TTFAT pool. Missing scripted turns are failures.

| Measure | Result |
|---|---:|
| Strict pass | 884/900 (98.2%) |
| Conversation-cluster bootstrap 95% CI | 97.4–98.9% |
| Tool error | 1.8% |
| Instruction error | 1.8% |
| KB error | 0.0% |
| Content-aware TTFAT P50 / P95 | 649 / 801 ms (n=883) |
| P50 / P95 vs local NVFP4 | +548 / +483 ms |
| Full scripted conversations | 30/30 |
| Recovery rows excluded from score, included in TTFAT | 13 |
