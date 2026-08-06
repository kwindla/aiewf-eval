# Nemotron 3 on vLLM/B200 for Voice-Agent Workloads

> Historical measurement record from February 2026. Runtime versions and
> upstream issue/PR status below are reported as observed during that campaign;
> they are not current deployment recommendations.

## Context and workload

We evaluate models for production voice-agent and multi-agent systems where latency and tool reliability both matter. Our target use cases include customer support, healthcare intake, user research, coding assistants, personal assistants, and robotics workflows that need both "thinking fast" and "thinking slow."

Our evaluation stack stresses:
- Large initial prompts
- 30-turn multi-turn context aggregation
- Time to first token (TTFT)
- Reliable tool calling and instruction following under long context

Primary benchmark in this write-up:
- `aiwf_medium_context` (AIEWF), text pipeline, strict per-turn scoring

## Serving stack

For these February 2026 measurements, we served on Modal B200 GPUs using vLLM from `main` (nightly build `0.16.0rc2.dev354+g5719a4e4e`, installed via `pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/`).

We chose the nightly over the latest stable release (v0.15.1) to pick up:
- Mamba SSM state leak fix ([PR #32118](https://github.com/vllm-project/vllm/pull/32118))
- B200 selective scan tuning
- Shared/routed expert overlap for Nemotron-H MoE
- MoE cold start optimization via `torch.compile` (`fast_moe_cold_start`)

## vLLM serve configuration

### BF16 Super (2x B200, TP=2)

```
vllm serve <model_path>
  --dtype auto
  --kv-cache-dtype fp8
  --tensor-parallel-size 2
  --pipeline-parallel-size 1
  --data-parallel-size 1
  --async-scheduling
  --enable-chunked-prefill
  --no-enable-prefix-caching
  --mamba-ssm-cache-dtype float32
  --max-num-seqs 512
  --max-model-len 65536
  --swap-space 0
  --gpu-memory-utilization 0.9
  --trust-remote-code
  --chat-template chat_template_nano.jinja
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser nano_v3
  --logits-processors ThinkingBudgetLogitsProcessor
```

### FP8 Super (1x B200, TP=1)

Same as above except:
- `--kv-cache-dtype auto` (not `fp8` -- see pitfalls below)
- `--attention-config.backend FLASH_ATTN` (not FlashInfer default -- see pitfalls below)
- `--tensor-parallel-size 1`
- `kv_cache_scheme` removed from the FP8 checkpoint's `config.json`

### Key flags and why

| Flag | Rationale |
|------|-----------|
| `--kv-cache-dtype fp8` | Saves ~40% KV cache memory on BF16 weights. Safe with BF16 weights + FlashInfer default. |
| `--no-enable-prefix-caching` | Prefix caching is unsupported for Mamba layers; enabling it causes 50-80% throughput regression. |
| `--mamba-ssm-cache-dtype float32` | NVIDIA-recommended for accuracy. Our earlier `float16` setting may have contributed to numerical drift. |
| `--async-scheduling` | Allows scheduling next batch while current batch is still running. |
| `--enable-chunked-prefill` | Enables chunked prefill for long-context inputs. |
| `--tool-call-parser qwen3_coder` | vLLM built-in parser that handles Nemotron's XML tool format. |

### Thinking budget logit processor

We use a custom vLLM V1 logit processor (`ThinkingBudgetLogitsProcessor`) adapted from NVIDIA's [Nemotron cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/vllm_cookbook.ipynb):
- Server-side default: unlimited thinking (`thinking_budget=-1`)
- Per-request override via `extra_body.extra_args.thinking_budget`
- After budget tokens, waits for a newline (up to `grace_period` extra tokens), then force-injects `</think>`

### Baseline request settings

- `temperature=0.6`
- `top_p=0.95`
- Streaming on by default
- Thinking ON by default (per-request opt-out via `chat_template_kwargs.enable_thinking=false`)

## Pitfalls encountered during bring-up

### FP8 weights + `--kv-cache-dtype fp8` produces `<unk>` tokens

With FP8-quantized model weights and `--kv-cache-dtype fp8`, approximately 40-60% of requests produce `<unk>` tokens (token ID 0) with null logprobs. This is non-deterministic and position-dependent. BF16 weights with `--kv-cache-dtype fp8` work correctly.

Root cause: interaction between ModelOpt static FP8 quantization and vLLM's FP8 KV cache. The FP8 checkpoint's `kv_cache_scheme` in `config.json` also causes `--kv-cache-dtype auto` to resolve to `fp8_e4m3`, triggering the same bug.

Fix: remove `kv_cache_scheme` from the FP8 checkpoint's `config.json` so that `--kv-cache-dtype auto` resolves to BF16.

### FP8 weights + BF16 KV cache crashes FlashInfer

After fixing the KV cache dtype, FlashInfer crashes with `AssertionError: is_strictly_contiguous(decode_query)` -- both during CUDA graph capture and during runtime inference.

Fix: `--attention-config.backend FLASH_ATTN`. The Flash Attention backend handles FP8 weights + BF16 KV cache correctly, including CUDA graph capture.

Do not use `--enforce-eager` as a workaround -- it drops throughput from ~100 tok/s to ~11 tok/s (9x slower).

### Prefix caching destroys Mamba throughput

Enabling `--enable-prefix-caching` caused 50-80% throughput regression on Nano. This feature is experimental for Mamba2 layers. Always disable explicitly with `--no-enable-prefix-caching`.

### Thinking OFF breaks tool discipline on Super

Disabling thinking entirely (`enable_thinking=false`) causes 100% spurious tool call rate on Super in multi-turn conversations. Nano is somewhat more resilient but still degrades. Low thinking budgets (`thinking_budget=20`) preserve most tool discipline while controlling TTFT.

### FP8 MoE regression in v0.15.x

During the campaign, vLLM v0.15.0 and v0.15.1 had a confirmed regression ([#34356](https://github.com/vllm-project/vllm/issues/34356)) that broke FP8 Nemotron models entirely ("No FP8 MoE backend supports the deployment configuration"). The linked fix PR ([#34404](https://github.com/vllm-project/vllm/pull/34404)) was still open at measurement time, so the campaign kept FP8 endpoints on v0.14.x.

### Cold start considerations

First cold start is slow due to empty torch.compile cache. Model loading takes ~95s (50 shards, 113 GiB for BF16 Super on 2x B200). CUDA graph capture for 51 batch sizes adds several more minutes. A persistent cache volume (`/root/.cache/vllm`) significantly speeds up subsequent cold starts.

## AIEWF results

Ground truth source: `docs/nemotron-3-super-progress.md`

| Model | Tool Use | Instruction | KB Ground | Turn Pass | Pass Rate | TTFT Med | TTFT P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemini-3-flash-preview | 300/300 | 300/300 | 300/300 | 300/300 | 100.0% | 1107ms | 1599ms |
| claude-sonnet-4-6 | 299/300 | 299/300 | 300/300 | 299/300 | 99.7% | 850ms | 4126ms |
| claude-haiku-4-5 | 298/300 | 294/300 | 300/300 | 294/300 | 98.0% | 637ms | 1615ms |
| gpt-5.1 | 294/300 | 294/300 | 300/300 | 294/300 | 98.0% | 739ms | 1492ms |
| nemotron-3-super-120b (FP8, unlimited thinking) | 299/300 | 290/300 | 300/300 | 290/300 | 96.7% | 1220ms | 1368ms |
| gpt-4.1 | 289/300 | 290/300 | 300/300 | 289/300 | 96.3% | 536ms | 1771ms |
| nemotron-3-super-120b (BF16, full thinking) | 296/300 | 289/300 | 299/300 | 289/300 | 96.3% | 922ms | 1262ms |
| gpt-4o | 291/300 | 285/300 | 299/300 | 284/300 | 94.7% | 546ms | 1369ms |
| nemotron-3-super-120b (BF16, thinking_budget=20) | 293/300 | 288/300 | 298/300 | 284/300 | 94.7% | 1005ms | 1087ms |
| nemotron-3-super-120b (FP8, thinking_budget=20) | 288/300 | 285/300 | 300/300 | 280/300 | 93.3% | 1276ms | 1398ms |
| nemotron-3-nano-30b | 287/300 | 281/300 | 295/300 | 277/300 | 92.3% | 745ms | 920ms |
| gemini-2.5-flash | 274/300 | 269/300 | 300/300 | 269/300 | 89.7% | 597ms | 1137ms |
| gpt-5.2 | 270/300 | 268/300 | 298/300 | 268/300 | 89.3% | 624ms | 1171ms |

Interpretation:
- Super is competitive with top frontier models on strict multi-turn/tool benchmarks.
- Best quality in current Super runs came from full/unlimited thinking variants.
- Budgeted variants remain usable but show a measurable reliability drop on strict turn-pass scoring.

## Thinking budget findings

### Why this matters for voice

Thinking tokens directly consume first-content latency budget. At ~200 tokens/sec, a 20-token thinking budget adds ~100ms. For sub-300ms voice targets, this is a large share of budget.

### What we observed

- Super: `thinking_budget=20` preserved much more tool discipline than strict thinking-off and was a reasonable latency/quality compromise.
- Nano: low thinking budgets degraded quality significantly on AIEWF; full thinking performed best in our sweeps.

Relevant Nano sweep results (from progress doc):
- Full thinking: `92.2%` turn pass (`83/90`)
- Best capped setting observed in that sweep: `thinking_budget=80` at `85.6%` (`77/90`)

## Preliminary external benchmark note (Gradient Bang)

From our separate mixed conversation + structured-agent benchmark effort:
- Nano 30B struggles on hardest long multi-turn tasks
- Super 120B is strong
- Super with larger thinking budgets in that benchmark can match Sonnet 4.6-level outcomes on most tasks and exceed Gemini 2.5 Flash on many tasks

## Requests for NVIDIA training + inference teams

- Investigate tool-calling reliability drop under strict thinking-off mode.
- Investigate FP8 `<unk>` behavior with `--kv-cache-dtype fp8` on FP8-quantized Nemotron Super. We suspect an interaction between ModelOpt static FP8 quantization calibration and vLLM's FP8 KV cache path.
- Investigate FlashInfer contiguity assertion failure with FP8 weights + BF16 KV cache. Currently requires falling back to `--attention-config.backend FLASH_ATTN`.
- The FP8 MoE regression in v0.15.x ([#34356](https://github.com/vllm-project/vllm/issues/34356)) blocks us from upgrading FP8 endpoints past v0.14.x.
- Provide guidance for robust `max_tokens` defaults under long-context + thinking to prevent decode runaways.
- Publish recommended vLLM/B200 config profiles for:
  - Low-latency voice-agent serving
  - Tool-heavy multi-turn agent serving
