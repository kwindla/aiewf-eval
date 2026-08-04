# Gemma 4 26B A4B publication updater

`publication_update.py` is a local-only, fail-closed handoff from the frozen
paired Gemma campaign to the README and filler report. Its default is a dry run.
It does not call BaseTen, a judge, or a collector.

## Final inputs

The updater selects `aggregates-full.json` when present, otherwise
`aggregates-initial.json`. It requires the matching `included-runs-*.tsv` and
`REPORT-*.md`, validates all hashes embedded by `analyze_stage.py`, and requires
`publication-review.json` copied from the example and completed by a reviewer.

An initial result is publishable only when its prespecified rule says not to
promote and the review says `stop_at_initial`. A full result additionally
requires the reviewed `promotion-decision-initial.json`, its hash-linked initial
aggregate and run manifest, and a final review action of
`publish_full_terminal`. A triggered initial result without its full terminal
stage is not publishable.

## Behavior

Dry run:

```bash
.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/publication_update.py
```

The proposed update replaces the existing
`gemma-4-26b-a4b-it (thinking off)` README row with the contemporaneous
no-filler arm, retaining the same label, final `BaseTen` Provider cell,
ten-column table shape, and descending pass-rate order. It adds exactly one
`gemma-4-26b-a4b` row to Section 3 through the canonical
`scripts/build_filler_report.py`; the chart and table use the no-filler P50
TTFAT. The generator rebuilds both report formats, updates the Markdown scope
word, and preserves every prior row and label, including optional Inkling Small
publication markers.

The dry run also stages exact-count compatibility updates for the four current
historical report verifiers. With Gemini 2.5 present, the final screen has 25
rows without Inkling Small or 26 with it. No source or generated file is changed
until an explicit `--apply` after all final inputs validate. Transformations are
idempotent and use git for recovery rather than backup files.
