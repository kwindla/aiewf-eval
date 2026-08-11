# Muse Glimmer reasoning-strength sweep

This is a four-arm, local RTX 5090 sweep of Muse Glimmer 30B on
`aiwf_medium_context`. The only intended model-behavior difference is the
GGUF-embedded Jinja template variable:

```json
{"chat_template_kwargs": {"reasoning_strength": "<arm>"}}
```

The valid arms are `low`, `medium`, `high`, and `xhigh`, as documented by
Meta's model card and implemented by the embedded template. There is no
supported `none`, `minimal`, or disabled mode. `audit_template.py` nevertheless
checks those spellings as negative render controls and proves that
`enable_thinking=false` and top-level `reasoning_effort=none` do not affect the
rendered prompt.

## Locked protocol

- 30 complete conversations per arm, 120 total.
- Deterministic, balanced interleaving from `make_schedule.py`.
- One server and one slot. Normal prefix caching is retained within each
  conversation, and slot 0 is explicitly erased before every conversation so
  no KV state crosses an arm or trajectory boundary.
- Public Meta K-Quant-Dynamic GGUF, Q8_0 K/V, 32,768-token context, DFlash 15.
- Embedded chat template via `--jinja`; no replacement template.
- Unrestricted reasoning budget and no request-level output-token cap.
- Temperature 1.0, top-p 0.95, top-k 64, min-p 0.0.
- The benchmark system instruction and scripted user prompts are not edited.
  `template-render-audit.json` records their SHA-256 and confirms the template
  adds its own reasoning-strength metadata after the unchanged system text.
- Claude Opus 4.5 judging, using the repository's standard judge.

Run from the repository root:

```bash
python3 ops/local-muse-glimmer-reasoning-sweep-20260811/make_schedule.py
ops/local-muse-glimmer-reasoning-sweep-20260811/worker.sh
ops/local-muse-glimmer-reasoning-sweep-20260811/judge.sh
python3 ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py
```

The collection and judging scripts are resumable. Generated campaign material
is written under `runs/muse-glimmer-reasoning-strength-n30-20260811/`.

## Result

All four arms completed 30/30 conversations and 900/900 scripted turns. `low`
is the best operating point for this benchmark: it scored 86.1%, versus 85.4%
for `medium`, 85.1% for `high`, and 84.9% for `xhigh`. The accuracy differences
are unresolved (`low - high` +1.00 points, conversation-cluster bootstrap 95%
CI -0.67 to +2.67), but `low` used 34.7% fewer mean completion tokens than
`high` and cut P95 TTFAT from 7,719ms to 3,105ms.

Redundant confirmation remains the dominant suggestion-turn failure. Across
scripted Turns 11 and 12, the exploratory rate rises from 45.0% at `low` to
60.0% at `xhigh`; the low-minus-xhigh interval still includes zero (-15.00
points, 95% CI -31.67 to +1.67). Turn 12 alone is 70.0% to 80.0% redundant
across all four arms and has no monotonic reasoning-strength pattern.

See the [full report](../../runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md),
[Meta GGUF model card](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF),
and [base chat template](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/chat_template.jinja).
