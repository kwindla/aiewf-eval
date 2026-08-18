# Qwen 3.8 27B FP8 on a dedicated Baseten H100 (SGLang)

Serving stack for the `qwen3.8-27b (thinking off, FP8)` leaderboard row: the
official `Qwen/Qwen3.8-27B-FP8` checkpoint on a dedicated single-H100 Baseten
deployment running pinned SGLang at the 262,144-token native context — BF16 KV
cache, no speculative decoding, chunked prefill 32,768, batch one. Native
thinking disabled for this row.

Campaigns:

- `aiewf-medium-none-n30-20260816/` — 30 conversations, fixed 900-turn
  denominator, thinking off. Strict pass 884/900 (98.2%). See
  `analysis/REPORT.md`.

Exact image/checkpoint pins and file hashes are in the campaign's manifests.
The full prelaunch/result/review chain lives in the gb-benchmarks repository
under `port-to-port/proj-2026-08-14-qwen38-27b-fp8-baseten/`.
