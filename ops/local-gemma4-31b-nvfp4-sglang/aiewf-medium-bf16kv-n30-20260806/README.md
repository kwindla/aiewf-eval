# Local RTX 5090 Gemma 4 31B NVFP4 + BF16 KV campaign

This frozen N=30, no-filler, thinking-off cohort repeats the local NVFP4
campaign with BF16 KV. It uses compact batch-one cache allocation: 16,000
full-attention slots and 5,600 sliding-window slots, with Radix prefix caching
enabled.

Run or resume with:

```bash
.venv/bin/python \
  ops/local-gemma4-31b-nvfp4-sglang/aiewf-medium-bf16kv-n30-20260806/run_campaign.py
```

The wrapper owns the Docker lifecycle and always stops the server.

Judge, validate, and aggregate with:

```bash
.venv/bin/python \
  ops/local-gemma4-31b-nvfp4-sglang/aiewf-medium-bf16kv-n30-20260806/judge_campaign.py \
  --execute --workers 4
.venv/bin/python \
  ops/local-gemma4-31b-nvfp4-sglang/aiewf-medium-bf16kv-n30-20260806/analyze.py
```

## Result

All 30 conversations completed all 30 turns on their first collection attempt.
The cohort scored 863/900 strict (95.9%, conversation-cluster bootstrap 95% CI
94.9–96.8%) with 100% KB grounding. TTFAT P50/P95 was 125/326ms.

Relative to the otherwise matched local FP8-KV cohort, BF16 KV is +1.2
percentage points (independent conversation-cluster bootstrap 95% CI -0.3 to
+2.8). The point estimate favors BF16, but the interval includes zero. See
`REPORT.md` and `aggregates.json` for the full three-way comparison.
