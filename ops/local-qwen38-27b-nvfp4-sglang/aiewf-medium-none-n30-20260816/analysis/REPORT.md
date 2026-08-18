# Qwen3.8 27B NVFP4 local AIEWF medium-context none

Fixed denominator: 30 conversations × scripted turns 0–29 = 900. Recovery rows are excluded from judge coverage and score. One absent scripted turn is a failure for strict and every component; 899 observed turns are judged.

| Measure | Result |
|---|---:|
| Strict pass | 880/900 (97.8%) |
| Conversation-cluster bootstrap 95% CI | 96.6–98.9% |
| Tool error | 1.9% |
| Instruction error | 2.2% |
| KB error | 0.1% |
| Current-convention pooled post-first content-aware TTFAT P50 / P95 | 101 / 318 ms (n=882, recovery included) |
| Historical Kimi/Gemma scripted-only sensitivity P50 / P95 | 101 / 317 ms (n=869) |
| Full scripted conversations | 29/30 |
| Recovery rows excluded from judge/score, included in headline TTFAT | 13 |
