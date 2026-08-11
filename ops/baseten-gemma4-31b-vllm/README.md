# Gemma 4 31B BaseTen deployment

This Truss replaces the retired Lilac route with a dedicated BaseTen endpoint
for the official `google/gemma-4-31B-it` BF16 checkpoint.

The initial serving contract is deliberately conservative:

- two H100 80 GB GPUs with tensor parallelism 2;
- a 32,768-token context limit and one concurrent request;
- automatic prefix caching and chunked prefill;
- vLLM's native Gemma 4 tool-call and reasoning parsers;
- native binary thinking control through
  `chat_template_kwargs.enable_thinking`;
- no speculative decoding until the base deployment passes the complete
  streaming, tool-continuation, usage-accounting, and benchmark probes.

The target is pinned to Hugging Face revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. The vLLM image is the same
immutable nightly previously validated by the repository's Gemma 4 26B A4B
BaseTen campaign.

Deploy with:

```bash
uvx --from truss truss push ops/baseten-gemma4-31b-vllm \
  --deployment-name gemma4-31b-vllm-20260806-apc \
  --wait --non-interactive --output json
```

The 2026-08-06 validation deployment is model `qzk215kq`, deployment
`wgvde5j`. Its OpenAI-compatible endpoint is:

```text
https://model-qzk215kq.api.baseten.co/deployment/wgvde5j/sync/v1
```

The first AIEWF medium-context freshness run completed all 30 scripted turns
on its first attempt with a valid runtime and complete usage accounting. The
30 scripted turns measured 456 ms P50, 550 ms P95, and 2,374 ms maximum TTFAT.
The run recorded 438,867 prompt tokens, including 423,072 cache-read tokens,
1,548 completion tokens, and no thinking tokens. This validates streaming,
prefix-cache accounting, thinking-off behavior, and the benchmark's tool-call
path, but it is not a replacement accuracy aggregate: the README table still
requires a newly judged multi-conversation BaseTen campaign.

Validation artifacts:

- `runs/aiwf_medium_context/20260806T094200_google_gemma-4-31B-it_8b8db844`
- `ops/model-freshness-2026-08-06/logs/infra-gemma431_baseten-attempt1.log`

After deployment, set the deployment's minimum replicas to zero. Campaign
runners may temporarily set it to one to avoid including cold-start time in
the latency measurements, but must return it to zero after the run.

After the validation run, the deployment was confirmed `SCALED_TO_ZERO` with
zero active replicas and `min_replica=0`.

Do not relabel the old Lilac aggregates or latency as BaseTen results. A new
BaseTen README row requires a newly judged BaseTen campaign.
