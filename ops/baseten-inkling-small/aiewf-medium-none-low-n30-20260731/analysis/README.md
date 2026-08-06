# Inkling Small fixed-denominator analysis

The analyzer uses only the exact run directories recorded in
`../canonical.tsv`; it never discovers or selects runs by scanning `runs/`.

The default invocation is a read-only progress and integrity audit:

```bash
.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/analysis/analyze.py
```

After collection contains all 30 `none` and 30 `low` conversations and the
judging workflow has written `../judging/COMPLETE.json`, generate the final
artifacts with:

```bash
.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/analysis/analyze.py \
  --write
```

`--write` validates the frozen schedule, runtime provenance, canonical
manifest, frozen judge inputs, judgment coverage, judge identity, and input
hashes before producing:

- `REPORT.md`: human-readable results and candidate README rows.
- `aggregates.json`: complete protocol, arm results, paired effects, turn-level
  results, exact membership, and hashes.
- `aggregates.tsv`: compact arm-level results.
- `effects.tsv`: low-minus-none paired effects and confidence intervals.
- `included-runs.tsv`: exact canonical membership and artifact hashes.
- `turn-errors.tsv`: all fixed-denominator per-turn counts and timing summaries.

Every canonical conversation contributes 30 scheduled turns. Unobserved future
turns after any short canonical run fail tool use, instruction following, KB
grounding, turn-taking, and strict pass, regardless of whether the immediate
cause was model behavior or serving. TTFAT, raw TTFB, and reasoning-token
statistics remain conditional on observed recorded values; no timing or token
count is invented for a missing turn. Cause attribution is reported separately
so fixed-denominator benchmark outcomes are not mistaken for model-only errors.

Arm intervals resample whole conversations. Low-minus-none intervals resample
the 30 frozen temporal pairs, preserving the paired campaign design.

Raw completion and serving causes are analyzed independently of the judge:

```bash
.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/analysis/failure_analysis.py

.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/analysis/failure_analysis.py \
  --write
```

The first command is read-only. `--write` creates `FAILURE-ANALYSIS.json` and
`FAILURE-ANALYSIS.md`. Membership comes only from `../canonical.tsv`; the
artifacts classify generated terminal calls separately from BaseTen 429-plus-idle
serving failures and hash every transcript and run log used. They do not read or
change judge outputs or the main generated aggregates.

A separate post-hoc judge sensitivity audit is available after the final main
aggregate and judging completion marker exist:

```bash
.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/analysis/judge_audit.py

.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/analysis/judge_audit.py \
  --write
```

The audit fails closed before final `aggregates.json` and
`../judging/COMPLETE.json` exist. It pins the IS-18 and IS-47 transcript and
judgment evidence, official judge policy, and final aggregate inputs. Its sole
counterfactual changes four `tool_use_correct` labels in the `none` arm and
emits `JUDGE-AUDIT.json` and `JUDGE-AUDIT.md`; it never edits official
judgments or aggregates.
