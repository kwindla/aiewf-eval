# Balanced origin follow-up

Status: preregistered and corpus frozen on 2026-08-08; inference collection has
not yet been analyzed.

This follow-up is the next experiment identified by the completed targeted KV
study. It crosses all 150 local-BF16-origin and all 150 local-FP8-origin turn-12
histories through both matched KV configurations, using 16 paired seeds per
history. See `PREREGISTRATION.md` for the frozen estimands and decision rule.

Artifacts:

- `snapshot-manifest.json` and `snapshots/`: 300 exact, outcome-blind frozen
  requests.
- `seed-manifest.json`: 4,800 unique per-history seeds, paired across arms.
- `collect_block.py`: fail-closed four-block collection driver using the
  parent study's audited server, replay, scorer, and provenance code.
- `analyze.py`: history-cluster and two-stage bootstrap analysis.

The frozen macro-block order is FP8 first half, BF16 first half, BF16 second
half, FP8 second half. Each block contains 2,400 measured requests.
