# Kimi K2.6 fixed-denominator analysis

After `collection/COMPLETE.json` and `judging/COMPLETE.json` exist, run:

```bash
.venv/bin/python \
  ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py
```

The analyzer includes only canonical runs, scores exactly scripted turns 0–29
for a fixed 900-turn denominator, and computes README TTFAT from exactly those
same 900 scripted rows. Recovery rows are reported separately and count only
toward billed token totals. It writes `aggregates.json`, `aggregates.tsv`,
`included-runs.tsv`, `REPORT.md`, and a hash-pinned `COMPLETE.json`; it never
edits the root README.
