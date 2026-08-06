# Gemma 4 26B A4B paired filler collection

This additive bundle leaves the completed parent Gemma campaign unchanged. It
collects fresh, contemporaneous no-filler and +96-dot controls against the
same dedicated BaseTen deployment.

Read-only preflight, which makes no BaseTen request:

```bash
.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/collect.py
```

Collect the frozen initial 10 pairs (10 conversations per arm):

```bash
.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/collect.py \
  --execute --stage initial
```

If the predeclared adaptive rule fires, judge and analyze the initial stage as
shown below. The analyzer writes the reviewed, hash-bound
`analysis/promotion-decision-initial.json`; use that exact generated file to
continue to 30 per arm. Do not hand-author the collector decision from the
example schema.

```bash
.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/collect.py \
  --execute --stage full \
  --decision-file ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/promotion-decision-initial.json
```

The driver is resumable and uses a nonblocking collection lock. It loads only
`BASETEN_API_KEY` from a supplied fallback dotenv file when the key is absent
from the environment. Before requests it scales the deployment to one replica;
after every live invocation it requests zero minimum replicas and confirms
scale-to-zero. Run logs, locks, and standard benchmark run directories are
ignored working state; the TSV ledgers and promotion record are durable audit
artifacts.

Judge the completed initial stage with at most two Claude workers. The first
command is a read-only preflight; the second is the only command that can send
judge requests:

```bash
.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/judge_stage.py \
  --stage initial

.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/judge_stage.py \
  --stage initial --execute --workers 2
```

Analyze the judged 10-pair stage without writing or calling a provider:

```bash
.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py \
  --stage initial
```

After reviewing that output, freeze the reports. If a prespecified promotion
trigger fired, an explicit reviewer is required and a collector-compatible
decision is written under `analysis/`; otherwise no decision file is created:

```bash
.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py \
  --stage initial --execute --reviewed-by "REVIEWER NAME"
```

That command never starts collection. Use the resulting
`analysis/promotion-decision-initial.json` in the separately reviewed
`collect.py --stage full --decision-file ...` invocation shown above. After
full collection, repeat judging and analysis with `--stage full`; the full
stage is terminal and does not accept `--reviewed-by`.

Run offline tests with:

```bash
.venv/bin/python \
  ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/test_bundle.py
```
