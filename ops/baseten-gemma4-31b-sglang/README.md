# Gemma 4 31B BaseTen SGLang bakeoff

These two Truss configurations isolate the effect of Gemma 4's official
NEXTN/MTP assistant while holding the serving stack and hardware constant.

- `no-mtp/`: SGLang control with RadixAttention prefix caching.
- `mtp/`: the same configuration plus the official 31B assistant checkpoint
  and the SGLang Gemma 4 NEXTN recipe.

Both configurations pin:

- SGLang `v0.5.16` (`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`) by
  immutable Linux/amd64 image digest;
- `google/gemma-4-31B-it` revision
  `842da3794eaa0b77d5f08bae87a17459d91ff475`;
- two H100 80 GB GPUs with tensor parallelism 2;
- BF16, one running request, a 32,768-token context, and 8,192-token
  chunked prefill;
- the native `gemma4` reasoning and streaming tool-call parsers;
- thinking disabled by default, with per-request overrides still accepted;
- OpenAI-compatible usage, cached-token reporting, and Prometheus metrics.

The MTP deployment additionally pins `google/gemma-4-31B-it-assistant`
revision `627c5ec1458b9086b841a91e0512fd31fd2fbbf1` and uses the official
five-step, six-draft-token, top-k-one NEXTN settings.

Before an accuracy campaign, gate each deployment with direct streaming,
forced and automatic tools, tool-result continuation, thinking-off, usage,
repeated-prefix cache, and full-conversation probes. Compare three full
conversations per arm before selecting the serving configuration for N=30.
Return every retained deployment to `min_replica=0` after testing.

## Direct TTFT probe

A 2026-08-06 direct streaming probe separates first-token serving latency
from the 30-turn campaign's realistic TTFAT distribution. Each warm phase has
30 requests. The tiny prompt contains 32–33 input tokens; the long prompt
contains about 10.7K. Both deployments were returned to zero replicas after
the probe.

| Stack | Tiny TTFT P50 | Tiny TTFT P95 | Long cold | Long warm P50 | Cached long-prefix tokens P50 |
|---|---:|---:|---:|---:|---:|
| SGLang + MTP | 222ms | 230ms | 1,436ms | 320ms | 10,662 |
| vLLM, no MTP | 229ms | 249ms | 959ms | 348ms | 10,656 |

The direct tiny-prompt result is therefore about 220–230ms, not 400ms. The
campaign's 430ms median includes real tool schemas, varying conversation
state, and benchmark/client overhead. Prefix caching is working in both
stacks and removes most long-prefix prefill latency after the cold request.
This probe compares whole serving stacks, so it does not isolate MTP; MTP is
primarily a decode optimization and should not be expected to materially
reduce time to the first token.

Raw observations are frozen in `ttft-probe-20260806.json`.

SGLang v0.5.16's Gemma 4 parser currently emits the called tool's position in
the request schema as the OpenAI streaming `tool_calls[].index`. The OpenAI
field instead identifies the call's zero-based position in the response. This
is reproducible on SGLang `main` at
`18e6c61c21ad39725522c008190d2b540dd6228d` as well as v0.5.16 and causes
Pipecat to ignore a first call to any tool other than schema entry zero. The
bakeoff therefore enables the benchmark's explicit
`MTE_VLLM_NORMALIZE_TOOL_CALL_INDICES=1` compatibility option, which maps raw
indices to response-local ordinals in first-seen order. The option is disabled
by default and the vLLM control does not require it.

## Pooled N=150 benchmark result

The selected SGLang + MTP configuration completed 150/150 valid, 30-turn
conversations. It scored 4,346/4,500 strict-correct turns, or 96.58%, with a
conversation-cluster bootstrap 95% CI of 96.13–97.02%. Pooled TTFAT was 490ms
P50 and 718ms P95. The extension also exposed intermittent serving tails: 27
of its 3,600 turns exceeded ten seconds.

The frozen campaign, audit, and reproducible analysis are in
`aiewf-medium-mtp-n150-20260807/`. The deployment was returned to
`SCALED_TO_ZERO` with zero active replicas after collection.
