# Local Qwen3.8 27B NVFP4 on RTX 5090 (SGLang)

Serving stack for the `qwen3.8-27b (thinking off, NVFP4)` leaderboard row: the community
Unsloth `Qwen3.8-27B-NVFP4` checkpoint served with a pinned SGLang image plus a one-file
lm_head quantization overlay on a single RTX 5090.

Locked configuration: 32,768-token context and token pool, batch one
(`--max-running-requests 1`), BF16 KV cache, explicit BF16 GDN/Mamba state with 12 Mamba
states (`extra_buffer_lazy`), chunked prefill 4,096, UnifiedRadixCache prefix caching
enabled, no speculative decoding, no request-level output-token cap. Native thinking is
disabled for this row (`enable_thinking=false`). The engine cache is flushed before every
attempt.

Campaigns:

- `aiewf-medium-none-n30-20260816/` — 30 conversations, fixed 900-turn denominator,
  thinking off. Strict pass 880/900 (97.8%). See `analysis/REPORT.md`.

Exact image, checkpoint, and overlay pins with file hashes are recorded in the campaign's
`package-manifest.json`, `inputs-manifest.json`, and `artifact-manifest.json`. The full
prelaunch/result/adversarial-review chain for this serving stack lives in the
gb-benchmarks repository under
`port-to-port/proj-2026-08-14-qwen38-27b-fp8-baseten/local-rtx5090-nvfp4/`.
