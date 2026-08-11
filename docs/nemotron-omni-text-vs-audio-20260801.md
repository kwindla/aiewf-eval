# Nemotron 3 Nano Omni on aiwf_medium_context: Text vs Audio Input (2026-08-01)

Evidence assembled for the case that the Nemotron Omni family needs more **multi-turn
conversational training data with tool calls** (in both text and audio modalities).

## Setup

- Model: local `Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`, served by **stock vLLM
  0.26.0** (PyPI, no patches) on RTX 5090, NVIDIA-recommended flags
  (`--reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser
  qwen3_coder`, fp8 KV auto-selected), `--max-num-seqs 1 --enforce-eager`.
- Benchmark: `aiwf_medium_context`, 30-turn voice-assistant conversations with a ~12K-token
  knowledge base and a tool schema (session suggestions, dietary requests, `end_session`).
- 10 conversations per input mode, identical sampling both modes: T=0.6, top_p=0.95,
  thinking off. Judge: `claude-opus-4-5` (same as leaderboard). Fixed 300-turn
  denominator per mode (missing turns after early exit count as errors, per README
  convention).
- Audio mode sends each user turn as WAV (`nemotron-audio-in` service); text mode sends
  the transcript string (`vllm-openai` service). Same conversations, same system prompt.

## Headline results

| | Audio input | Text input |
|---|---:|---:|
| Conversations reaching 30 turns | **6/10** | **0/10** |
| Conversation lengths | 1,12,16,18,30,30,30,30,30,30 | 1,1,1,1,1,1,1,11,12,16 |
| Strict pass (fixed 300-turn denom) | **56.3%** | **6.7%** |
| Tool use (fixed denom) | 57.0% | 7.0% |
| Tool use (scored turns only) | 75.3% | 45.7% |
| Instruction following (scored) | 76.7% | 43.5% |
| KB grounding (scored) | **99.1%** | **97.8%** |

Same model, same server, same conversations: **text-mode conversations median 1 turn;
audio mode mostly goes the distance.** Knowledge-base grounding is near-perfect in both
modes — the failures are concentrated entirely in conversational tool policy and
instruction following.

## The turn-0 collapse (text mode)

Measured directly against the API with the benchmark's exact first-turn payload
(n=20, T=0.6/top_p=0.95): user asks *"I'm trying to decide whether to come for workshop
day. When are the workshops?"* —

- 9/20 answer the question
- 7/20 call `end_session` (ending the conversation unprompted)
- 4/20 call an unrelated tool with **fabricated user data**
  (`submit_dietary_request {"name": "Lisa", "dietary_preference": "vegetarian"}`,
  `submit_session_suggestion {"name": "Alex Chen", ...}`)

Verified model behavior, not harness: the May-era harness (pipecat 1.1) and current
harness (pipecat 1.3) produce **byte-identical request payloads**, and the collapse
reproduces on a second, independently built server stack (patched vLLM 0.20). The
harness code comments separately record that NVIDIA's recommended instruct sampling
(T=0.2, top_k=1) collapses to `end_session` deterministically. Collapse probability is
stack-and-sampling dependent (the May 2026 runs on a DGX Spark at T=1.0 completed 47/47
conversations), which is itself evidence of knife-edge decision margins rather than
robust policy.

## Context from prior runs

- **May 2026, text mode, DGX Spark, T=1.0** (47 judged conversations, previously
  unaggregated): tool-use errors 13.8%, instruction errors 16.5%, KB errors 0.6% —
  roughly **3× the tool/instruction error rate of the base text LLM**
  (`nemotron-3-nano-30b`: 5.0% / 6.1% / 4.0% on the public leaderboard). The multimodal
  fine-tune traded away conversational tool-use capability while improving grounding.
- Family gradient (text mode, same benchmark): nemotron-3-ultra 100%, super-120b 97.0%,
  nano-30b 90.6%, **nano-omni ~84% (May, favorable stack) → 6.7% (current stack)**.
- Speech-to-speech peers on the same benchmark: ultravox-v0.7 97.7%, gpt-realtime-2
  96.0%, gemini-3.1-flash-live 91.7% — the Omni's audio-mode 56.3% is far below every
  production speech model on the board.

## Supporting evidence from the audio front-end study (May–July 2026)

See `~/src/nemotron-nano-omni/artifacts/` (audio-frontend bundles + retests + 500-sample
sweep): multi-turn tool decisions with audio in context sit on knife-edge margins —
flipped deterministically by inaudible waveform differences on a pinned stack (with
text-only controls passing), flipped stochastically by kernel-level numerics on current
stacks (~1.2–1.4% hard-failure floor at temperature 0), with degenerate failure modes
(blank text, raw-JSON-as-text tool calls, leaked `<|tool_call|>` tags anchored to
utterance content).

## The case in one paragraph

Nemotron 3 Nano Omni's knowledge grounding is excellent in both modalities, but its
multi-turn conversational tool policy is fragile everywhere: ~3× the base LLM's error
rate in its best text-mode configuration, sampling- and stack-sensitive collapse into
unprompted session termination and fabricated tool calls at the first turn of a
realistic voice-agent setup, and audio-mode performance (56.3%) far below production
speech models despite beating its own text mode. Every failure signature — degenerate
modes at decision boundaries, fabricated tool arguments, premature `end_session`, thin
margins that any numeric perturbation crosses — points at insufficient training
coverage of **long multi-turn conversations with tool calls**, in text and especially
with audio user turns in context. The knowledge is in the weights; the conversational
tool behavior is not.

## Reproduction

- Runs: `runs/aiwf_medium_context/20260801T06*_nemotron_3_nano_omni_*` (10 audio + 10
  text, each with `claude_summary.json`), plus turn-0 distribution and wire-payload
  captures in the nemotron-nano-omni session artifacts.
- Audio mode: `MTE_NEMOTRON_AUDIO_IN_BASE_URL=... MTE_NEMOTRON_AUDIO_IN_TEMPERATURE=0.6
  MTE_NEMOTRON_AUDIO_IN_TOP_P=0.95 MTE_NEMOTRON_AUDIO_IN_TOP_K= uv run multi-turn-eval
  run aiwf_medium_context --model nemotron_3_nano_omni --service nemotron-audio-in`
- Text mode: `OPENAI_API_KEY=EMPTY VLLM_BASE_URL=... uv run multi-turn-eval run
  aiwf_medium_context --model nemotron_3_nano_omni --service vllm-openai --pipeline text`
