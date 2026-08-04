# Inkling Small publication updater

`publication_update.py` is the final, local-only handoff from the frozen
Inkling Small analyses to the public README and filler report. It is read-only
by default and must not be applied before both required inputs below exist.

## Required inputs

1. Primary none/low result:
   `../../aiewf-medium-none-low-n30-20260731/analysis/aggregates.json`
2. Raw serving-cause attribution for that same frozen cohort:
   `../../aiewf-medium-none-low-n30-20260731/analysis/FAILURE-ANALYSIS.json`
3. Final judge-sensitivity audit for that same frozen cohort:
   `../../aiewf-medium-none-low-n30-20260731/analysis/JUDGE-AUDIT.json`
4. The highest reached dots result in this directory: `stage-30.json`, else
   `stage-10.json`, else `stage-6.json`.

The primary file must be schema version 1 and `artifact_status=FINAL`, identify
`thinkingmachines/inkling-small` on `BaseTen Model API`, contain exactly 30
conversations and 900 fixed turns in each of `none` and `low`, expose the five
README accuracy fields plus P50/P95/max TTFAT, and have valid hashes for every
file in `input_hashes`.

The raw failure artifact must be the hash-complete, judge-independent schema-1
cause attribution for the same 60-run none/low campaign. It must attribute
exactly 12 `none` and 10 `low` short runs to a BaseTen HTTP 429 followed by the
harness idle timeout, with no unattributed short run. The final judge audit must
cover the same frozen primary cohort and the four disputed
`tool_use_correct` labels, hash its own inputs, and establish that every
audited arm-level rate changes by no more than 0.5 percentage points under the
sensitivity analysis.

The reached-stage file must be schema version 1 for campaign
`aiewf-medium-inkling-small-baseten-none-dots-adaptive-20260731`; contain the
30-conversation frozen `control_none` and exactly 6, 10, or 30 `dots96`
conversations; use fixed 30-turn denominators and 100,000 whole-conversation
bootstrap draws; include a completed-stage adaptive recommendation; and hash
its frozen control, judge-input, and judge-completion files. The frozen dots
control's accuracy fields and P50 TTFAT must exactly match the primary `none`
arm.

## Publication behavior

Dry run:

```bash
.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-dots-adaptive-20260731/analysis/publication_update.py
```

The dry run prints unified diffs only. It proposes:

- exactly one `inkling-small (none)` and one `inkling-small (low)` README row;
- the existing ten-column README shape, without a run-count column and with
  `BaseTen` in the final Provider column;
- descending pass-rate order and refreshed explanatory prose;
- a normalized, hash-linked `publication-input.json`;
- a marker-delimited extension to the canonical
  `scripts/build_filler_report.py` data-loading path;
- one compact Inkling Small robustness disclosure in Section 3's existing
  run-pool provenance paragraph, in both Markdown and HTML: 22/60 retained
  primary attempts were BaseTen 429-plus-idle serving failures, and the
  sensitivity check of four disputed `tool_use_correct` labels changed
  arm-level rates by no more than 0.5 percentage points;
- additive `BaseTen` mappings for both rows in the existing Gemini 2.5 README
  provider verifier, whose exact mapping check would otherwise reject them.

The robustness disclosure is report-only. It does not add a table column or
per-model run count to the top-level README; the Inkling Small README prose
describes the frozen paired/fixed-denominator design without publishing its
conversation counts.

After reviewing the dry run, apply locally with `--apply`. The updater writes
the normalized input and generator extension, invokes the canonical generator
to rebuild both report Markdown and HTML, and then writes the README. It does
not edit generated HTML directly. Validation requires exactly one Inkling Small
row and chart label, the row's `none` P50 TTFAT, `BaseTen` provider, the existing
nine-column Section 3 table shape, and every pre-existing Section 3 model row
and chart label in its original order.

The transformation is idempotent. It uses git for recovery rather than backup
files, never calls a provider, and cannot launch collection or judging.
The canonical generator's existing 24-row count becomes 25 as part of the
staged generator transform; no second report-row counter is edited elsewhere.
