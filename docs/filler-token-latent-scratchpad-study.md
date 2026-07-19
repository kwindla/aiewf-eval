# Filler-Token Latent Scratchpad — Cross-Model Study

**Question:** Does appending content-free "filler" tokens (a run of dots) to a
prompt recover the accuracy that reasoning models lose when run with thinking
OFF — without paying the latency cost of visible reasoning?

**Origin:** arxiv 2607.03502 ("filler tokens as a latent scratchpad"), building on
Pfau et al. 2024 ("Let's Think Dot by Dot"). The paper tested only open-weights
models on toy/arithmetic tasks and could NOT verify on closed API models. This
study extends it to **17 model configs across 4 providers on a real 30-turn
conversational tool-use benchmark** (`aiwf_medium_context`).

## Method

- `MTE_FILLER_DOTS=<n>` appends `n` space-separated dots to the **final user
  turn** of each outgoing request (shared `services/filler.py`). Applied on a
  copy at request-build time, so the persisted history / in-context examples stay
  filler-free by design (don't teach the model the filler pattern across turns).
- Default probe: **96 dots**, thinking-OFF, vs a no-filler baseline, ~6–10 runs
  each (30 turns/run = 300 judged turns at n=10), interleaved, judged by the
  claude-agent-sdk judge.
- Wired into every open request path: OpenAI Responses (gpt-4.1/5.4/5.6), Lilac
  (gemma), BaseTen (inkling/glm/kimi/nemotron/oss), OpenRouter (Qwen3/DeepSeek),
  and Anthropic (Haiku — injected at `_create_message_stream`, skipping
  tool_result blocks so it lands on the question).
- TTFAT is measured content-aware (first *answer* token, thinking excluded), so a
  latency-neutral result is meaningful.

## Results — +96 dots vs nofiller, thinking-off

| model | provider | arch | nofiller | +96 dots | Δ | verdict |
|---|---|---|---|---|---|---|
| **gpt-5.4** | OpenAI | (undisclosed) | 90.3 | 96.3 | **+6.0** | ✅ works (n=10) |
| **gpt-5.6-sol** | OpenAI | (undisclosed) | 96.7 | 100.0 | **+3.3** | ✅ works (n≈6→10) |
| **gpt-5.6-terra** | OpenAI | (undisclosed) | 92.8 | 95.8¹ | +3.1¹ | ⚠️ helps survivors BUT ~70% turn-0 `end_session` abort |
| gpt-5.6-luna | OpenAI | (undisclosed) | 89.2 | 88.3 | −0.9 | ⬜ no |
| **gemma-4-31b** | Lilac | dense | 97.0 | 99.0 | **+2.0** | 🟡 mild+ (n=10) |
| gpt-4.1 | OpenAI | (non-reasoning) | 96.6 | 96.7 | +0.0 | ⬜ no-op |
| gpt-oss-120b | BaseTen | MoE | 84.9 | 85.3 | +0.3 | ⬜ no-op |
| kimi-k2.6 | BaseTen | MoE | 94.4 | 94.4 | +0.0 | ⬜ no-op |
| nemotron-3-ultra (550B) | BaseTen | MoE | 98.9 | 98.9 | +0.0 | ⬜ no-op (ceiling) |
| **Qwen3-8B** | OpenRouter | dense (36L) | 77.2 | 77.2 | +0.0 | ⬜ no-op |
| **Qwen3-14B** | OpenRouter | dense (40L) | 91.3 | 91.3 | +0.0 | ⬜ no-op |
| **Qwen3-32B** | OpenRouter | dense (64L) | 90.7 | 93.3 | **+2.7** | 🟡 mild+ (depth signal; deepening) |
| claude-haiku-4-5 | Anthropic | (undisclosed) | 99.4 | 98.3 | −1.1 | ⬜ no (deepening) |
| inkling | BaseTen | (undisclosed) | 97.1 | 95.0 | −2.1 | 🔻 negative |
| DeepSeek-chat-v3.1 | OpenRouter | MoE | 94.0 | 90.7 | −3.3 | 🔻 negative (deepening) |
| **glm-5.2** | BaseTen | MoE | 99.7 | 95.3 | **−4.3** | 🔻 negative (n=10) |
| **nemotron-super (120B)** | BaseTen | MoE | 84.9 | *breaks* | ❌ | filler → 100% turn-0 `end_session` |

¹terra_filler96 aborts on ~70% of attempts (turn-0 `end_session`); the +3.1 is on
the survivor subset and carries survivorship bias. See "Downside risk" below.

**TTFAT stayed flat everywhere** — dots are input-only, so where filler helps it's
free at the median (gpt-5.4 P50 ≈ 640–680ms across all doses).

## gpt-5.4 dose-response (thinking-off)

| dots | pass% | Δ | TTFAT P50 |
|---|---|---|---|
| 0 | 90.3 | — | 677ms |
| 24 | 95.4 | +5.1 | 657ms |
| 48 | 91.2 | +0.9 (noise) | 649ms |
| 96 | 96.3 | +6.0 | 658ms |
| 192 | 97.5 | +7.2 | 641ms |

Roughly monotonic (dots48 dip is noise). Median latency dead flat at every dose;
only the P95/max tail creeps up at 192 (more prefill). 96 = safe sweet spot; 192
= ~1pt more at some tail-latency risk. Approaching the thinking-on ceiling
(gpt-5.4 low = 97.0%).

## Patterns

The models sort into **three regimes**: **exploit** (filler → latent compute →
gain), **ignore** (robust, no change), **derail** (OOD dots hijack behavior → loss
or breakage). Candidate explanatory variables:

1. **Baseline headroom — RULED OUT.** Lowest baseline (oss-120b 84.9) gains
   nothing; highest (glm 99.7) is hurt; the winners sit mid-range.
2. **Depth (serial layers) — supported.** Filler adds *parallel* compute at
   *fixed depth* (Pfau); it can't add the serial depth that CoT provides. In the
   dense Qwen3 sweep, benefit appears **only at the deepest model** (32B/64L
   +2.7) and is absent at 8B/36L and 14B/40L. The cleanest contrast is 14B vs
   32B: *same baseline (91%), same arch, same training/tokenizer — only depth
   differs, and only the deeper one benefits.*
3. **Training idiosyncrasy dominates architecture.** Within one vendor+generation,
   gpt-5.6 **sol/terra benefit but luna doesn't** — same family, opposite result.
   So the exploit ability is a per-checkpoint post-training property, not an
   architecture class. (Also: gemma dense benefits, but DeepSeek/glm MoE are hurt
   while the paper's MoE DeepSeek-V3 was its top gainer — arch family alone
   doesn't predict.)
4. **Task-dependence, not just model-dependence.** DeepSeek-V3 was the paper's
   *strongest* positive on toy arithmetic; DeepSeek-chat-v3.1 here is **negative**
   (−3.3). A model's toy-task filler gain does not transfer to conversational
   tool-use. (Caveat: V3.1-chat ≠ the paper's V3; deepening in progress.)

## Downside risk — filler is not a free "always try it" trick

- **Net-negative** on glm-5.2 (−4.3, n=10) and inkling (−2.1).
- **Catastrophic** on **nemotron-super**: 96 trailing dots read as
  "conversation over" → spurious `end_session` on **turn 0, 100% of 24 attempts.**
- **Partial breakage** on **gpt-5.6-terra**: the *same* `end_session` failure on
  ~70% of attempts. So the "filler derails tool discipline" hazard recurs across
  models.
- This connects to a prior, independent finding: `docs/nvidia-nemotron-vllm-b200-writeup.md`
  documented that **thinking-off itself** (`enable_thinking=false`) causes a 100%
  spurious-tool-call rate on Nemotron-Super. Filler-triggered `end_session` is the
  same failure family.

**Implication:** per-model validation is mandatory before deploying filler — it
can help (+6), do nothing, silently cost accuracy (−4), or hard-break tool use.

## Caveats

- n≈6–10; several rows still deepening (marked above).
- "Thinking off" is not one thing: BaseTen `reasoning_effort=none`, vLLM
  `enable_thinking=false`, and Responses `reasoning.effort=none` are different
  switches with different tool-discipline consequences (see the Nemotron
  June-vs-July reconciliation: same model scored 63% thinking-off via vLLM
  `enable_thinking=false` on an FP8 stack vs 98.9% via BaseTen `reasoning_effort=none`).
- Serving stack matters (quant, prefix caching, batching) and is confounded across
  providers.
- We cannot observe the latent computation directly; "exploit" is inferred from
  the accuracy gain at flat TTFAT, not measured.

## Actionable

- **gpt-5.4 (none, +96 dots): 96.3% @ ~660ms** — thinking-on accuracy at
  thinking-off latency. A legitimate leaderboard config (already footnoted).
- gpt-5.6-sol similar (+3.3 → 100%). terra unusable+filler (end_session).
- Everything else: no-op or worse. Do not apply blindly.
