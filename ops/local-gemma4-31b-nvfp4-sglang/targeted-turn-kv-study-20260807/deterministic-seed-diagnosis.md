# SGLang deterministic-mode diagnosis

Date: 2026-08-07

## Finding

SGLang's `--enable-deterministic-inference` is not a request-seed switch. In
the pinned image it bundles at least four behaviors:

1. request seeds populate `SamplingBatchInfo.sampling_seed`;
2. the PyTorch position-keyed sampler replaces the ordinary sampler;
3. batch-invariant matrix/attention operations are enabled; and
4. insertion of finished requests into the radix cache is disabled.

Behavior 4 makes broad deterministic mode structurally incompatible with the
warm-cache treatment in this study. A completed prime request cannot leave its
prefix in RadixAttention for the measured request.

There is a second checkpoint-specific failure. With the NVFP4 Gemma 4 model,
Triton attention, and broad deterministic mode, a 15-token request completed,
but 14K-token target requests timed out after 180 seconds. During the stall the
scheduler process used one CPU core, GPU utilization remained near 1%, and the
request never appeared in SGLang's prefill telemetry. This localizes the stall
before model prefill, but does not establish which batch-invariant preparation
operation loops or scales pathologically. The diagnostic server log is
`server-logs/aiewf-gemma-kv-target-fp8-compact.final.log`.

## Resolution

`shims/sitecustomize.py` changes the server-argument view in exactly
`sglang.srt.sampling.sampling_batch_info`. That module now populates the
existing per-request seed tensor, while every other SGLang module continues to
see `enable_deterministic_inference=False`. The matched server also selects
SGLang's PyTorch sampler explicitly. Therefore:

- the request seed drives SGLang's position-keyed random variates;
- normal Triton model and attention kernels remain in use;
- completed requests remain eligible for radix-cache insertion; and
- the exact same read-only shim is present in BF16- and FP8-KV arms.

Initial validation:

| Arm | Cache state | Seed groups | Semantic repeat mismatches | Cache telemetry |
|---|---|---:|---:|---|
| FP8 KV | warm | 2 | 0 | 14,202 / 14,203 prompt tokens cached |
| FP8 KV | cold | 2 | 0 | zero cached tokens |
| BF16 KV | warm | 2 | 0 | 14,202 / 14,203 prompt tokens cached |
| BF16 KV | cold | 2 | 0 | zero cached tokens |

SGLang creates a random UUID-like tool-call ID after generation. That field
changed in an otherwise identical repeated tool call, so the repeatability
signature excludes only the ID while retaining assistant text, reasoning,
tool name, raw arguments, finish reason, and mechanical score.

The broad deterministic configuration remains available through
`server.sh --sampling batch-invariant` for diagnosis, but it is excluded from
the KV experiment. `--sampling seeded` is the matched experimental mode.
