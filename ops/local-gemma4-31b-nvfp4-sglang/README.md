# Local Gemma 4 31B NVFP4 on RTX 5090

This experiment measures whether NVFP4 materially changes AIEWF benchmark
quality or latency relative to the BaseTen BF16 Gemma 4 31B cohort. It uses the
public, prequantized `RedHatAI/gemma-4-31B-it-NVFP4` checkpoint at revision
`edafdf3dcaef23ff76f75b91edd6a4a975a399cf`.

The NVIDIA checkpoint was not selected: its 32.6 GB stored model leaves no
room on a 32 GB RTX 5090 for runtime allocations or KV cache. The Red Hat
checkpoint is 23.3 GB on disk, is explicitly documented for SGLang, and keeps
the vision tower, embeddings, and output head in their original precision.
`NVFP4` names the quantization format, not a promise that every tensor uses
four bits. NVIDIA's recipe also excludes all 60 language-model self-attention
blocks from NVFP4, leaving that large weight family in BF16. Red Hat applies
NVFP4 to the language transformer's linear layers, including attention, while
still excluding the vision tower and non-linear/head modules. Those additional
BF16 attention weights account for most of NVIDIA's roughly 9.3 GB size
premium.

The server is text-only, thinking-off, single-request, and prefix-cached. MTP
is intentionally omitted. SGLang's CUTLASS FP4 backend is selected explicitly
because the
pinned v0.5.15.post1 image has CUDA 13.0.1 but cuDNN 9.13; the documented
FlashInfer cuDNN backend requires cuDNN 9.15 or newer.

The completed N=30 cohort used FP8 E4M3 KV. Gemma 4 has both sliding-window
and full-attention layers, and the initial layout gave both pools enough slots
for a complete roughly 15.5K-token conversation. That equal-capacity layout
does not fit in BF16, so the first result measures NVFP4 weights plus FP8 KV,
not a weights-only NVFP4 effect. A later follow-up found a compact BF16 layout;
see below.

Decode CUDA graphs are disabled because the pinned image did not complete its
startup graph capture for this compressed-tensors FP4 checkpoint on SM120.
That limitation is recorded as a serving-latency caveat, not a model-quality
change.

The built-in multimodal warmup is skipped. SGLang v0.5.15.post1 applies the
checkpoint's compressed-tensors FP4 scheme to the unquantized vision tower
during that synthetic image warmup, despite the checkpoint exclusion list,
and fails there. This benchmark is text-only; text requests do not execute the
vision tower. The explicit protocol and full-conversation gates replace the
built-in warmup for this experiment.

The SM120 CUTLASS kernel is JIT-compiled on first use. Its generated artifacts
are persisted in `/home/khkramer/.cache/sglang-tvm-ffi-gemma4-nvfp4` so server
restarts do not repeat the several-minute compile.

Chunked prefill is limited to 2,048 tokens to keep prefill memory bounded.
The sliding-window and full-attention pools receive equal token capacity: a
single live sequence consumes a slot in each pool, so both must cover the
longest benchmark conversation.

Download the frozen checkpoint:

```bash
PYENV_VERSION=3.12.10 HF_HUB_DISABLE_TELEMETRY=1 \
  hf download RedHatAI/gemma-4-31B-it-NVFP4 \
  --revision edafdf3dcaef23ff76f75b91edd6a4a975a399cf
```

Start, inspect, and stop the local server:

```bash
ops/local-gemma4-31b-nvfp4-sglang/start_server.sh
docker logs -f aiewf-gemma4-31b-nvfp4
ops/local-gemma4-31b-nvfp4-sglang/stop_server.sh
```

The OpenAI-compatible endpoint is `http://127.0.0.1:30000/v1`. Benchmark
clients must set `MTE_VLLM_NORMALIZE_TOOL_CALL_INDICES=1` because SGLang's
Gemma 4 parser reports schema-position indices rather than response-local tool
call ordinals.

## Results

Each configuration pools its immutable N=30 cohort with a matched N=120
extension. All three pooled cohorts completed 150/150 full conversations; each
extension required exactly 120 collection attempts.

| Configuration | Strict pass | 95% CI | TTFAT P50 | TTFAT P95 |
|---|---:|---:|---:|---:|
| BaseTen BF16 weights/KV + MTP, 2xH100 | 96.58% | 96.13–97.02% | 490ms | 718ms |
| Local NVFP4 weights + FP8 KV, RTX 5090 | 95.49% | 94.96–96.00% | 105ms | 309ms |
| Local NVFP4 weights + BF16 KV, RTX 5090 | 96.16% | 95.62–96.64% | 128ms | 336ms |

Local BF16 KV minus local FP8 KV is +0.67 percentage points (independent
whole-conversation bootstrap 95% CI -0.07 to +1.40; 50,000 replicates). The
point estimate favors BF16 KV, but N=150 still does not establish a nonzero
global difference. This is the cleanest available KV-precision comparison:
weights, hardware, SGLang image, sampling, batch size, and MTP setting are held
fixed. The compact BF16 arm necessarily uses smaller, asymmetric static cache
pools.

Local BF16 KV minus BaseTen BF16 weights/KV + MTP is -0.42 point (95% CI -1.09
to +0.24). Local FP8 KV minus BaseTen is -1.09 points (95% CI -1.78 to -0.42).
These remain end-to-end deployment comparisons because weight precision, MTP,
hardware, and SGLang version differ.

Both local deployments had 100% KB grounding and normal 30-turn completion.
Errors remained concentrated in state-carrying tool use. BF16 had 28 fewer
turn-12 errors and nine fewer turn-15 errors than FP8, partly offset by three
additional errors at each of turns 17 and 24.

See `pooled-n150-analysis-20260807/REPORT.md` for the current three-way
comparison. The original N=30 reports remain frozen in
`aiewf-medium-n30-20260806/` and `aiewf-medium-bf16kv-n30-20260806/`.

## BF16 KV follow-up

Full-precision KV does fit without removing the vision modules. The useful
lever is asymmetric cache sizing: Gemma 4 has 10 full-attention layers that
need the complete conversation, but its 50 sliding-window layers only need the
local window plus in-flight prefill/decode tokens. `start_server_bf16_kv.sh`
therefore caps the BF16 full-attention pool at 16,000 tokens and the SWA pool
at 5,600 tokens (`--swa-full-tokens-ratio 0.35`). The two pools occupy 5.49 GB
and leave about 4.4 GB free after server startup. Radix prefix caching remains
enabled.

The checkpoint's vision tower and projection occupy only about 1.07 GiB, so
skipping them would not by itself make the original equal-capacity BF16 layout
fit. SGLang v0.5.15.post1 also rejects `--language-only` for Gemma 4; that flag
currently supports a limited set of encoder-disaggregation architectures.

The compact BF16 layout passed 4K- and 14K-token requests, the standard
streaming/tool/cache probe, and one complete benchmark smoke conversation
judged 30/30 strict. Tiny-prompt TTFT P50 was 43ms; the 10.7K-prefix request was
1,841ms cold and 97ms warm with 10,658 cached tokens. The pooled N=150 BF16
cohort completed every conversation and scored 4327/4500 strict (96.16%,
cluster 95% CI 95.62–96.64%). Its +0.67-point difference from local FP8 KV has
a 95% interval of -0.07 to +1.40 points.
