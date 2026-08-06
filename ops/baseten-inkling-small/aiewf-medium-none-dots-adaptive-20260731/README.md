# Inkling Small adaptive 96-dot campaign

This additive bundle measures `thinkingmachines/inkling-small` with 96
space-separated trailing dots at `reasoning_effort=none`. It reuses the 30
completed `none` conversations from the primary none/low campaign as its only
control. The control is frozen by transcript hash before the first dot request;
this bundle has no control-run or control-top-up path.

Dot conversations run strictly sequentially against BaseTen's serverless Model
API. The prospective caps are 6, 10, and 30. Start with six:

```bash
.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-dots-adaptive-20260731/collect.py \
  --stage 6

.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-dots-adaptive-20260731/collect.py \
  --stage 6 --execute
```

The first command is read-only. `--execute` requires the completed primary
campaign and freezes its 30 none-arm transcripts into `control-inputs.tsv`.
Collection is resumable and stops exactly at the requested stage cap.

Judge the reached stage after collection. This driver is also read-only by
default and judges only the newly collected dot transcripts. It pins
`claude-opus-4-5` / `claude-agent-sdk-v4-turn-taking`, uses at most two
workers, and passes only `ANTHROPIC_API_KEY` into its provider environment:

```bash
.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-dots-adaptive-20260731/judge_stage.py \
  --stage 6

.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-dots-adaptive-20260731/judge_stage.py \
  --stage 6 --execute
```

After all 30 controls and the reached dot stage are judged, run the analysis.
The default invocation computes and validates without writing. `--execute`
writes the exact gate artifact `analysis/stage-6.json` plus a readable Markdown
summary; it evaluates but never executes the extension gate:

```bash
.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-dots-adaptive-20260731/analyze_stage.py \
  --stage 6 --execute
```

Do not request stage 10 or 30 directly. After judging and analyzing the
completed preceding stage, freeze an extension decision with `gate_stage.py`:

```bash
.venv/bin/python \
  ops/baseten-inkling-small/aiewf-medium-none-dots-adaptive-20260731/gate_stage.py \
  --from-stage 6 --to-stage 10 \
  --analysis ops/baseten-inkling-small/aiewf-medium-none-dots-adaptive-20260731/analysis/stage-6.json \
  --rationale "prespecified stage-6 trigger fired" --execute
```

Then run `collect.py --stage 10 --execute`. The 10-to-30 transition uses the
same procedure. The gate records the analysis artifact's SHA-256; later
preflights reject a changed artifact.

Both the frozen control and dots arm use temperature 1.0, 16,384 maximum
completion tokens, and the primary campaign's eligibility rule. Model-caused
early exits and later timeouts remain outcomes. Only objective provider or
transport failures with zero valid model responses may be replaced, up to four
attempts per dot slot.

See `protocol.md` for the fixed adaptive rule and interpretation constraints.
