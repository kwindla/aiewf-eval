# Local Gemma 4 31B pooled N=150 analysis

This directory contains the incremental judging snapshots and final pooled
analysis for the local FP8-KV and compact BF16-KV arms. Each arm combines its
immutable N=30 cohort with the matching N=120 extension.

Incremental judging only selects completed runs already committed to an
extension's canonical manifest. It freezes each selected transcript hash before
calling the campaign's frozen Claude judge, skips valid existing judgments, and
shares the same lock as the final campaign judge.

```bash
.venv/bin/python \
  ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/incremental_judge.py \
  --kv-cache fp8 --execute --watch --workers 4

.venv/bin/python \
  ops/local-gemma4-31b-nvfp4-sglang/judge_extension.py \
  --kv-cache fp8 --execute --workers 4
```

Run the corresponding two commands with `--kv-cache bf16`, then generate the
pooled report only after both final extension validations succeed:

```bash
.venv/bin/python \
  ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py
```

## Result

| Configuration | Strict pass | Whole-conversation bootstrap 95% CI | TTFAT P50/P95 |
|---|---:|---:|---:|
| BaseTen BF16 weights/KV + MTP | 4346/4500 (96.58%) | 96.13–97.02% | 490/718ms |
| Local NVFP4 + FP8 KV | 4297/4500 (95.49%) | 94.96–96.00% | 105/309ms |
| Local NVFP4 + BF16 KV | 4327/4500 (96.16%) | 95.62–96.64% | 128/336ms |

Local BF16 minus FP8 KV is +0.67 percentage points (independent
whole-conversation bootstrap 95% CI -0.07 to +1.40). The estimate favors BF16,
but the interval narrowly includes zero. See `REPORT.md` and `aggregates.json`
for the turn-level and deployment comparisons.
