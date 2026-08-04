# Kimi K2.6 on Cerebras — eval notes (2026-05-25)

## Model

Kimi K2.6 (Moonshot AI), served by Cerebras Inference. Frontier open-weight MoE
model: 1T total params, ~32B active per token. Text-only at launch on Cerebras
(vision targeted later).

| Field | Value |
|---|---|
| Cerebras model ID | `moonshotai-kimi-k2.6` |
| Endpoint | `https://api.cerebras.ai/v1` (OpenAI-compatible) |
| Context | 131K tokens |
| Throughput | ~1,000 TPS (claimed) |
| Reasoning modes | Thinking (default) / Instant (`reasoning_effort="none"`) |
| Sampling (guide) | Thinking T=1.0/top_p=0.95; Instant T=0.6/top_p=0.95 |
| Reasoning field | `reasoning` (Cerebras renames OpenAI's `reasoning_content`) |

Access is gated. Our default `CEREBRAS_API_KEY` does **not** have Kimi entitled;
a separate EAP key ("Pipecat org created by Cerebras for Kimi testing") does. The
EAP key is the active `CEREBRAS_API_KEY` in `.env`; the old key is commented out
just above it.

## Harness changes

Kimi runs through the existing `cerebras` service alias
(`pipecat.services.cerebras.llm.CerebrasLLMService`, OpenAI-compatible), so no new
service was needed. Two changes in `src/multi_turn_eval/pipelines/base.py` (the
`"Cerebras" in class_name` branch):

1. **Thinking/Instant mode + sampling**, driven by env vars:
   - `MTE_CEREBRAS_REASONING_EFFORT=none` → Instant mode (thinking disabled).
     Unset/anything else → default Thinking mode.
   - `MTE_CEREBRAS_TEMPERATURE` / `MTE_CEREBRAS_TOP_P` override sampling.
   - Defaults follow the Cerebras guide: Thinking T=1.0, Instant T=0.6, top_p=0.95.

2. **Bug fix — use `settings=`, not `params=`.** `CerebrasLLMService.__init__`
   always builds its own `settings` object and passes it to the parent
   `BaseOpenAILLMService`. The parent only applies a deprecated `params=` *when no
   `settings` is given* (base_llm.py:191), so any `params=InputParams(...)` we
   passed for Cerebras was silently dropped — temperature, top_p, and the
   `reasoning_effort` we put in `extra` never reached the API. The first instant
   run looked wrong (still ~220 reasoning tokens/turn) because of this. Fix: build
   `service_class.Settings(temperature=..., top_p=..., extra={"reasoning_effort":
   ...})`. `reasoning_effort` lands as a top-level chat param because
   `CerebrasLLMService.build_chat_completion_params` does `params.update(extra)`.

   Verified after the fix: Instant mode → 0 reasoning tokens; Thinking → 210–1463
   tokens/turn. (Other OpenAI-compatible aliases — vllm, lilac, modal — also pass
   `params=`, but they construct the base `OpenAILLMService` directly, where the
   deprecated path still works. The dropped-`params` issue is specific to
   subclasses that pre-build `settings`, like Cerebras.)

3. **Content-aware TTFB** (`services/cerebras_logged.py`, `LoggedCerebrasLLMService`;
   the `cerebras` alias now points here). The base service stops the TTFB metric on
   the first non-empty `choices` chunk, which in Thinking mode is the first
   *reasoning* token — understating true user-visible TTFT by ~2-3x. This subclass
   overrides `_process_context` to stop TTFB only on the first content / tool-call /
   transcript delta (reasoning-only chunks don't count), mirroring the existing
   Nemotron fix. Token accounting and tool-call handling are otherwise identical to
   upstream. Without this, the Thinking-mode latency numbers are meaningless.

## Performance — `aiwf_medium_context`

### Thinking vs Instant (n=30 each, 900 turns each)

| Model | Pass Rate | Turn Pass | Tool Use | Instruction | KB Ground | TTFT Med | TTFT P95 | TTFT Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| moonshotai-kimi-k2.6 (thinking) | 98.0% | 882/900 | 889/900 | 882/900 | 900/900 | 452ms | 1350ms | 3483ms |
| moonshotai-kimi-k2.6 (instant) | 94.0% | 846/900 | 847/900 | 855/900 | 900/900 | 256ms | 480ms | 1639ms |

Thinking **98.00%, 95% Wilson CI [96.86%, 98.73%]** vs Instant **94.00%,
[92.25%, 95.37%]** — the CIs do not overlap, so thinking is significantly better.
Thinking TTFT is content-aware (see fix below); instant TTFT was always valid (no
reasoning). KB grounding is perfect (900/900) in both modes.

### Thinking, n=50 (1,500 turns) — statistical confirmation

| Model | Pass Rate | Turn Pass | Tool Use | Instruction | KB Ground |
|---|---:|---:|---:|---:|---:|
| moonshotai-kimi-k2.6 (thinking) | 97.6% | 1464/1500 | 1471/1500 | 1467/1500 | 1500/1500 |

Turn-pass **97.60%, 95% Wilson CI [96.70%, 98.26%]**. The n=10 estimate held. KB
grounding was perfect across all 1,500 turns. Thinking cost is modest: median ~160
reasoning tokens/turn.

(TTFT omitted from this n=50 row — it was collected before the content-aware-TTFB
fix and measured time-to-first-*reasoning*-token, not first visible content. See
the corrected TTFT below.)

### Corrected TTFT (content-aware, thinking, n=30 / 900 turns)

The original TTFT numbers were wrong: the base Cerebras/OpenAI service stops the
TTFB metric on the first non-empty `choices` chunk, which in Thinking mode is the
first *reasoning* token — not the first user-visible *content* token. After adding
a content-aware TTFB override (`LoggedCerebrasLLMService`, see below) and re-running
30 full thinking runs:

| Metric | Old (first reasoning token) | Corrected (first content token) |
|---|---:|---:|
| TTFT median | ~241ms | **452ms** |
| TTFT P95 | ~472ms | **1350ms** |
| TTFT max | 1616ms | **3483ms** |

Corrected-run pass rate: **98.0% (882/900), 95% Wilson CI [96.86%, 98.73%]** —
consistent with the n=50 figure, now on honest TTFT.

The median is still fast (~450ms) because reasoning length is highly variable —
many turns reason lightly and stream content in ~300–450ms. But the **P95 of ~1.5s
is the number that matters for voice**: it exceeds the ~700ms "too slow for voice"
guideline, so Thinking-mode Kimi is borderline on the tail — quick on typical
turns, with a meaningful slow tail when it reasons hard. Instant mode is unaffected
(no reasoning; ~155–250ms TTFT) but carries the quality regression documented above.

## Failure analysis

### Thinking (36 failures / 1,500 turns)

| Turn | Failures | % of runs | Pattern |
|---|---:|---:|---|
| 16 | 26 | 52% | Premature `request_tech_support` — files the ticket on "I'm having trouble with the mobile app" instead of first asking what the issue is |
| 19 | 7 | 14% | Deflects an answerable venue-navigation question ("can't access maps, how do I get to Salon 2?") as out-of-scope |
| 29 | 3 | 6% | Occasionally misses `end_session` at the goodbye |

The turn-16 over-eager tool call is the dominant, systematic failure (72% of all
failures). It's the same premature-tool-call sensitivity that trips other models on
the equivalent turn; Kimi just hits it more consistently. Fixing this one behavior
would push the model toward ~99%.

### Instant (54 failures / 900 turns, n=30)

Instant mode fails in a qualitatively different way — **context amnesia**. In 44 of
54 failures the model re-asks for the user's name (already given as "Jennifer
Smith" at turn 10) instead of submitting the expected tool call:
- turn 15 (×21/30 = 70%): re-asks name instead of `submit_dietary_request`
- turn 17 (×11), turn 24 (×11): same, blocking `request_tech_support` / `vote_for_session`
- turn 29 (×10/30 = 33%): misses `end_session` at the goodbye

Without reasoning the model loses track of established conversational state. This is
the clear cost of disabling thinking, and why thinking mode is the right config for
agentic multi-turn voice work — **+4.0 points (98.0% vs 94.0%, non-overlapping 95%
CIs)** at no first-token-latency penalty on typical turns.

## Operational notes

The EAP endpoint does not tolerate high concurrency. **40 concurrent runs reliably
hung** — every run idle-timed-out around turn 8–9 (where prompts grow), with a
couple of server-side 503s and no clean 429s. Throttling to **≤8 concurrent** ran
clean (one transient wave-level hiccup). Use a concurrency cap for any large sweeps
on this key.

Run sources for the n=50 thinking aggregate: 10 from the thinking/instant sweep + 32
from the throttled 40-run batch (8 of that batch's runs died in one bad wave) + 8
low-concurrency top-up runs.
