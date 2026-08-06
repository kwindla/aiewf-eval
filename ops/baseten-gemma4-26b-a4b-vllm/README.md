# Gemma 4 26B A4B — BaseTen AIEWF campaign

This additive bundle collects and judges the standard
`aiwf_medium_context` benchmark for `google/gemma-4-26B-A4B-it` on the
dedicated BaseTen deployment:

`https://model-qel1y223.api.baseten.co/deployment/qz4zpye/sync/v1`

The publication cohort is frozen at 30 thinking-off, no-filler conversations.
Requests are strictly sequential on the model endpoint. A model-caused short
conversation remains canonical after its first valid response; only an attempt
with no valid model response is replaced.

## Before collection

Do not start the N=30 campaign until the direct and Pipecat smoke suite has
verified all of the following:

- the exact live vLLM version is the newest deployment build;
- plain streamed text has valid OpenAI-compatible chunks and usage;
- automatic and forced tool calls stream a stable ID, function name, and
  incrementally valid arguments in the form Pipecat expects;
- a tool result can be followed by another streamed assistant response;
- `chat_template_kwargs.enable_thinking=false` suppresses reasoning cleanly;
- prefix-cache reuse is measurable and does not alter outputs;
- the one-token Gemma MTP assistant either passes the latency/correctness gate,
  or is explicitly disabled after the A/B smoke.

Then update only these fields in `configuration.json`:

- `serving.vllm_version`
- `serving.mtp.status`
- `serving.verified` to `true`

The collector deliberately refuses `--execute` while those fields are pending.

## Collection

Read-only preflight:

```bash
cd /home/khkramer/src/aiewf-eval
.venv/bin/python ops/baseten-gemma4-26b-a4b-vllm/collect.py
```

Live, resumable collection:

```bash
cd /home/khkramer/src/aiewf-eval
.venv/bin/python ops/baseten-gemma4-26b-a4b-vllm/collect.py --execute
```

The runner reads only `BASETEN_API_KEY` from
`/home/khkramer/src/gb-benchmarks/.env` when neither `VLLM_API_KEY` nor
`BASETEN_API_KEY` is already present. It does not source either repository's
environment file. It removes all `MTE_FILLER_*` variables from the child
environment and pins:

- temperature `1.0`
- top-p `0.95`
- top-k `64`
- max tokens `8192`
- `MTE_VLLM_THINKING=0`

Every CLI invocation writes the repository's standard run directory under
`runs/aiwf_medium_context/`, including `transcript.jsonl`, `runtime.json`, and
`run.log`. The campaign bundle records a separate console log and durable
attempt/canonical manifests. A filesystem lock prevents two collectors from
using the endpoint at once.

The deployment must remain live until the collector records:

```text
RUN_COLLECTION_DONE total=30 none=30
```

After that marker, no further model requests are needed by this workflow.
The completed campaign scaled the deployment back to zero replicas; the
post-teardown control-plane state is recorded in
`deployment-after-teardown.json`.

## Judging

The judge driver is read-only by default:

```bash
cd /home/khkramer/src/aiewf-eval
.venv/bin/python ops/baseten-gemma4-26b-a4b-vllm/judge_campaign.py
```

After all 30 runs are canonical:

```bash
cd /home/khkramer/src/aiewf-eval
.venv/bin/python ops/baseten-gemma4-26b-a4b-vllm/judge_campaign.py \
  --execute \
  --workers 2 \
  --max-attempts 3
```

The driver pins the actual implementation identity
`claude-opus-4-5` / `claude-agent-sdk-v4-turn-taking`, freezes transcript and
source hashes, snapshots invalid pre-existing outputs before a retry, and
validates that judging did not mutate the transcripts. It writes
`judging/COMPLETE.json` only after all 30 outputs validate.

Judging uses Anthropic and may run in parallel with work against the BaseTen
endpoint. Collection itself never overlaps two BaseTen requests.

## Analysis and README handoff

See `analysis/README.md`. Final analysis uses the fixed denominator of
30 conversations × 30 turns = 900 turns. It produces a candidate table row but
does not edit `README.md`; publication remains a separate reviewed step.
