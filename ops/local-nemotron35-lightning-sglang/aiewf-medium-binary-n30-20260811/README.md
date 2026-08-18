# Nemotron 3.5 Lightning AIEWF medium-context campaign

This campaign compares the two native reasoning modes of
`nemotron-3.5-lightning` on the standard `aiwf_medium_context` benchmark:

- `off`: `enable_thinking=false`
- `on-unbounded`: `enable_thinking=true`, with no thinking budget

Each arm contains 30 conversations of 30 scripted turns (900 fixed-denominator
turns per arm). The frozen schedule alternates the order within two-run blocks,
so each arm appears first in exactly 15 blocks.

## Locked serving and request protocol

- NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 checkpoint, revision
  `e7fa1b0bdaf462c67c7f0bf638addacd89fd3054`.
- NVIDIA's dedicated SGLang image pinned at
  `lmsysorg/sglang@sha256:a04d9a1a7ffe371b05230aecab001d4ba2bfa0e5c137bc56409ecc4cbc3ac864`
  (SGLang commit `d59c1ddf70ee17fcc41c053ed38bd60bc6cc28cc`).
- One RTX 5090; one request at a time; 65,536-token server context.
- Unified RadixAttention prefix caching for full-attention and Mamba state.
  `/flush_cache` is called before each conversation, preventing state from
  crossing trajectories while retaining prefix reuse within the conversation.
- NVIDIA cookbook template controls:
  `enable_thinking=<arm>` and `force_nonempty_content=true`.
- Model-card sampling: temperature 1.0 and top-p 0.95.
- No request-level output-token cap and no thinking-budget cap.
- No filler tokens; benchmark prompts and scripted turns are unchanged.
- Logged OpenAI-compatible Pipecat service records raw first-chunk TTFT and
  content-aware TTFAT (first answer token or tool-call output).
- First-valid-response fixed-denominator eligibility: a model-caused early
  exit remains canonical and all missing future scripted turns count as errors.

The reusable collector freezes all request, benchmark, service, and schedule
sources before the first counted request. Generated campaign artifacts live in
`artifacts/`; raw conversation directories remain under
`runs/aiwf_medium_context/`.

## Admission result

Both complete 30-turn Pipecat smokes passed on 2026-08-12 UTC. Thinking off
recorded raw TTFT and TTFAT on 30/30 turns with identical 62.5ms medians.
Thinking on/unbounded recorded both metrics on 30/30 turns, with a 64ms raw
TTFT median, 1,432.5ms TTFAT median, and 7,155ms maximum TTFAT. Both runs
exercised tool calls and multi-turn continuation, called `end_session` on the
final scripted turn, and left the SGLang endpoint healthy.
