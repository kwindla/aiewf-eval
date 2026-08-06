# Fixed-denominator analysis

This workflow uses only the exact run directories recorded in
`../canonical.tsv`. It never scans `runs/` to discover or select inputs.

Read-only preflight:

```bash
cd /home/khkramer/src/aiewf-eval
.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py preflight
```

Final analysis, after `judging/COMPLETE.json` exists:

```bash
cd /home/khkramer/src/aiewf-eval
.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py final
```

The final command writes:

- `aggregates.json`: machine-readable protocol, provenance, aggregate, hashes,
  and per-run audit records.
- `aggregates.tsv`: compact one-arm summary.
- `included-runs.tsv`: exact membership and artifact hashes.
- `REPORT.md`: human-readable result and a candidate README table row.

Each conversation contributes exactly 30 scheduled turns. Missing future turns
after a short model-caused run fail strict pass and all three quality
dimensions. TTFAT statistics remain conditional on observed responses; the
analysis does not invent latency for missing turns.
