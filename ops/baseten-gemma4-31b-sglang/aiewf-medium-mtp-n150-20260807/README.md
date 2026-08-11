# Gemma 4 31B BaseTen pooled N=150 analysis

This directory pools the unchanged canonical N=30 campaign with the separate
frozen N=120 extension. `analyze.py` validates both judging completion markers,
freezes a pooled canonical manifest with artifact hashes, and produces fixed-
4,500-turn accuracy, conversation-cluster bootstrap intervals, latency, token,
and per-turn error results.

```bash
.venv/bin/python \
  ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py
```

## Result

| Strict pass rate | Conversation-cluster bootstrap 95% CI | TTFAT P50 | TTFAT P95 |
|---:|---:|---:|---:|
| 4,346/4,500 (96.58%) | 96.13–97.02% | 490ms | 718ms |

All 150 conversations completed all 30 scripted turns. See `REPORT.md` for
the turn-level error distribution and latency-tail details.
