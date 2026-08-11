# Gemma 4 31B SGLang NEXTN/MTP AIEWF campaign

This is the frozen 30-conversation, no-filler, thinking-off campaign for the
BaseTen SGLang v0.5.16 NEXTN/MTP deployment selected by the matched serving
bakeoff. Collection uses the repository's portable, sequential campaign
collector and a fixed 900-turn denominator. A model-caused short conversation
after its first valid response remains in the cohort; infrastructure attempts
with no valid response are replaced.

The request configuration is temperature 1.0, top-p 0.95, top-k 64, and an
8,192-token output cap. Thinking is explicitly disabled. The opt-in
`MTE_VLLM_NORMALIZE_TOOL_CALL_INDICES=1` compatibility behavior corrects the
known SGLang Gemma 4 streaming parser bug described in the parent directory's
README.

Preflight:

```bash
.venv/bin/python ops/aiewf-campaign-template/collect.py \
  --config ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/configuration.json
```

Execute or resume with the resource-safe wrapper. It wakes deployment
`q951m16w/q862ez8`, waits for readiness, invokes the frozen collector, and
returns the replica minimum to zero in a `finally` block:

```bash
.venv/bin/python \
  ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/run_campaign.py
```

After collection, return the deployment to `min_replica=0`, judge the frozen
canonical cohort, generate fixed-denominator aggregates, and update the
current-production README row.
