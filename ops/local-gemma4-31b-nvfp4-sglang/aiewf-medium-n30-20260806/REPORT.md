# Gemma 4 31B local NVFP4 campaign

The local NVFP4-weights + FP8-KV cohort scored 852/900
strict turns (94.7%, conversation-cluster bootstrap 95%
CI 93.4–95.9%).

| Configuration | Strict pass | 95% CI | Tool error | Instruction error | KB error | TTFAT P50 | TTFAT P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BaseTen BF16 + MTP, 2xH100 | 96.9% | 95.8–97.9% | 3.1% | 3.1% | 0.0% | 430ms | 564ms |
| Local NVFP4 + FP8 KV, RTX 5090 | 94.7% | 93.4–95.9% | 5.3% | 5.3% | 0.0% | 102ms | 295ms |

Local minus BaseTen strict pass is -2.2 percentage points (independent
conversation-cluster bootstrap 95% CI -3.8 to -0.7
points).

| Turn | Requirement | BaseTen errors | Local errors | Local − BaseTen |
|---:|---|---:|---:|---:|
| 12 | Submit second session suggestion | 19/30 | 21/30 | +2 |
| 14 | Offer vegan dietary request | 1/30 | 0/30 | -1 |
| 15 | Submit confirmed vegan request | 6/30 | 22/30 | +16 |
| 17 | Submit mobile-app support request | 1/30 | 2/30 | +1 |
| 24 | Submit session vote | 1/30 | 3/30 | +2 |

The difference is highly concentrated: 16 of the 20 additional local errors
occur on turn 15, where the one-word confirmation `Yes.` must retrieve the
previously established name and vegan preference and call
`submit_dietary_request`. All 48 local strict failures are paired tool-use and
instruction-following failures; KB grounding is 900/900. Every conversation
completed all 30 turns and called `end_session` on scripted turn 29.

This is an end-to-end deployment comparison, not a weights-only quantization
ablation: the local arm also uses FP8 KV, omits MTP, runs on one RTX 5090, and
uses SGLang v0.5.15.post1; the BaseTen arm uses BF16 weights/KV, NEXTN MTP,
two H100s, and SGLang v0.5.16. Missing future turns count as failures. Latency
uses observed scripted turns; whole conversations are the bootstrap unit.

A subsequent serving follow-up found a compact BF16-KV layout that fits on the
5090 by reserving 16K full-attention slots and 5.6K sliding-window slots. Its
frozen N=30 cohort scored 863/900 strict (95.9%, cluster 95% CI 94.9–96.8%).
BF16 minus FP8 KV was +1.2 percentage points with an independent
conversation-cluster bootstrap 95% interval of -0.3 to +2.8 points. See
`../aiewf-medium-bf16kv-n30-20260806/REPORT.md` for that comparison.
