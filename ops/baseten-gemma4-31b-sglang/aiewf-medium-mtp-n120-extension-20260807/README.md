# Gemma 4 31B BaseTen N=120 extension

This is the frozen 120-conversation extension of the canonical N=30 BaseTen
SGLang v0.5.16 NEXTN/MTP campaign. It uses the identical endpoint, model,
sampling, no-filler, thinking-off, tool-index compatibility, eligibility, and
serial collection settings. The original N=30 campaign remains untouched.

Read-only preflight:

```bash
.venv/bin/python ops/aiewf-campaign-template/collect.py \
  --config ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n120-extension-20260807/configuration.json
```

Execute or resume. The wrapper returns `min_replica` to zero and waits for
`SCALED_TO_ZERO` in a `finally` block:

```bash
.venv/bin/python \
  ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n120-extension-20260807/run_campaign.py
```

Judging is resumable and may watch collection safely:

```bash
.venv/bin/python \
  ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n120-extension-20260807/judge_campaign.py \
  --execute --watch --workers 4
```
