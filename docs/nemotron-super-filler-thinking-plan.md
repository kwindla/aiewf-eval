# Nemotron-3-Super: Filler Tokens × Thinking Budget — Pareto Plan

**Goal:** Nemotron-Super (120B-A12B) is a good mid-sized model, but on this
benchmark it needs substantial thinking to hit top accuracy (self-hosted:
97.0% at a 512-token thinking budget), and with reasoning disabled its tool
discipline degrades. Our benchmark objective is excellent accuracy at very low
TTFAT. **Hypothesis:** cheap *input-side* latent compute (filler tokens) can
substitute for some *output-side* serial compute (thinking tokens), moving the
accuracy/TTFAT Pareto frontier — e.g. (small thinking + filler) dominating
(large thinking, no filler).

This is theory-consistent with the depth analysis in
`docs/filler-token-latent-scratchpad-study.md`: filler adds parallel compute at
fixed depth, thinking adds serial depth — they are complementary, so partial
substitution is plausible where the task mix is breadth-heavy.

## Endpoint choice: BaseTen (accepted trade-off)

BaseTen's production endpoint is preferred because our self-hosted vLLM stack
had real inference bugs (`docs/nvidia-nemotron-vllm-b200-writeup.md`: FP8
`<unk>` corruption on 40–60% of requests; `enable_thinking=false` → 100%
spurious tool calls). BaseTen thinking-off already scores 84.9% — the
"pathological" behavior was substantially a stack artifact.

**Probed control surface on BaseTen (2026-07-19, direct API):**

| control | result |
|---|---|
| `reasoning_effort=none` | **400 rejected** |
| `reasoning_effort=low/medium/high` | accepted but **inert** (reasoning length ≈ unchanged) |
| `chat_template_kwargs.enable_thinking` | **the real switch**: true → separate pre-answer `reasoning_content` (~350 tok default) + terse answer; false → no thinking phase, verbose reasoning inline in `content` |
| `chat_template_kwargs.thinking_budget` | **soft nudge only**: budget 32/128/512 → ~256/322/391 actual reasoning tokens. Guides, doesn't cap |
| `extra_args.thinking_budget` | ignored |

**Consequence:** BaseTen cannot reach truly small thinking budgets (the range
compresses to ~256–390 tokens). The fine-budget axis of the original idea needs
the self-hosted vLLM `ThinkingBudgetLogitsProcessor` (hard cap). Decision: run
Phase 0 on BaseTen; revisit self-hosted (BF16, not FP8) only if Phase 0 shows
promise AND the soft-budget spread is too narrow to matter.

## Known risk (the gate)

With thinking OFF, 96 trailing dots reliably read as end-of-conversation:
**100% turn-0 spurious `end_session`** (24/24). Same failure on gpt-5.6-terra
(~70%). Whether a *thinking phase* inoculates against this is untested and
decides the whole line. Filler pattern is therefore a first-class variable: a
pattern that doesn't read as "conversation over" (e.g. dashes) may dodge the
trigger even where dots fail. `MTE_FILLER_TOKEN` (new knob, default `.`) sets
the repeated token.

## Phase 0 — gate + baselines (BaseTen, running)

Driver: `run_nemsuper_phase0.sh` (scratchpad); manifest
`nemsuper_phase0_manifest.tsv`; model `nvidia/Nemotron-120B-A12B`.

- **(a) GATE — does thinking-ON survive filler?**
  4 attempts `enable_thinking=1` + 96 dots, 2 attempts + 96 dashes
  (`MTE_FILLER_TOKEN=-`). Each attempt classified SUCCESS / ABORT
  (turn-0 `end_session`) / FAIL — no blind retries, so abort *rate* is measured.
  Successes are judged and banked as real thinking-on+filler cells.
  - Dots abort rate ≈ 0 → idea lives; proceed to Phase 1 with dots.
  - Dots abort, dashes survive → proceed with dashes (pattern matters).
  - Both abort → line is dead on BaseTen for trailing-filler patterns.
- **(b) CONTROLS (refresh, n=6 each):**
  - `nemsuper_think_nofiller` — `enable_thinking=1`, no filler. The honest
    BaseTen high-accuracy anchor (the 97%@512 figure is self-hosted, not
    comparable). Expect accuracy near ceiling; TTFAT elevated by the ~350-token
    thinking phase.
  - `nemsuper_nothink_nofiller` — `enable_thinking=0`, no filler. Low-latency
    anchor; prior n=4 gave 84.9% (noisy) — refreshed at n=6.
  Both give the (accuracy, TTFAT) corners of the Pareto plane.

## Phase 0 RESULTS (2026-07-19)

**Gate (0a): PASSED mechanically — thinking inoculates against filler-abort.**
6/6 thinking-ON+filler attempts completed clean (4× dots96, 2× dash96, zero
`end_session` aborts) vs 100% abort at thinking-off. The abort hazard is a
thinking-off-only phenomenon.

**But filler HURTS thinking-on accuracy** — the substitution hypothesis fails:

| cell | n | pass% | TTFAT P50 |
|---|---|---|---|
| nothink_nofiller (survivors) | 5 | 73.3% (range 23–93!) | 517ms |
| **think_nofiller** | 6 | **99.4%** | 967ms |
| think_dots96 | 4 | 91.7% (−7.7) | 1021ms |
| think_dash96 | 2 | 95.0% (−4.4) | 1111ms |

Both filler patterns degrade the 99.4% thinking-on baseline, and filler buys no
latency (thinking dominates TTFAT; the filler cells are ~50–150ms *slower*).

**Major control finding — thinking-off Super is pathological on BaseTen too,
and the earlier 84.9% was survivorship-biased.** Fresh controls: 13 attempts →
5 clean / 6 turn-0 aborts / 2 other; survivors averaged 73.3% with wild
variance (one 23% run). Re-examining the broaden-phase logs confirms its
"truncations" were the same spurious `end_session` (turn-1 aborts + premature
end_sessions at turns ~19–22), not BaseTen 529s. So the spontaneous-abort
propensity is model-level (present without any filler, ~40–50% of attempts);
filler amplified it to 100%. The self-hosted writeup's "thinking OFF breaks
tool discipline" holds on BaseTen's production stack as well.

**Verdict: Phase 1 not warranted on BaseTen.** No path to Pareto improvement:
(a) filler degrades accuracy when thinking is on, (b) the soft thinking_budget
can't create a low-latency thinking tier, (c) thinking-off is unusable as a
filler base (aborts nofiller). The clean Super config on BaseTen is
**enable_thinking=1, no filler: 99.4% @ 967ms TTFAT**. Remaining option if the
low-TTFAT goal stays priority: the self-hosted hard-budget fallback below.

## Phase 1 — Pareto sweep (NOT RUN — gated off by Phase 0 verdict)

Grid on BaseTen, ~6 runs/cell, using the best-surviving filler pattern:

- thinking axis: `enable_thinking=0` | `=1 + thinking_budget=128` (soft) |
  `=1` unbudgeted — (the soft-budget cell may collapse into unbudgeted; keep it
  only if 0b shows the soft budget moves TTFAT at all)
- filler axis: 0 | 48 | 96

Per cell: pass%, TTFAT P50/P95, reasoning-token count. Deliverable: the
accuracy-vs-TTFAT scatter; success = a (thinking-light + filler) cell that
dominates the thinking-on nofiller anchor (≥ its accuracy, materially lower
TTFAT), or a (thinking-off + filler) cell well above 84.9% without breakage.

## Phase 2 — pattern optimization (only if Phase 1 finds a live cell)

At the best cell: dots vs dashes vs newline-ish vs a neutral repeated word;
possibly count sweep {24..192}. Also the arxiv paper's observation that
pattern choice is model-specific applies here.

## Fallback

If BaseTen's soft budget makes Phase 1 degenerate (only off vs ~350-tok on),
port the sweep to self-hosted vLLM with the hard `ThinkingBudgetLogitsProcessor`
at BF16 (avoids the FP8 `<unk>` bug), budgets {32, 64, 128, 256, 512} × filler
{0, 48, 96} — accepting stack risk in exchange for the fine-budget axis.

## MODAL HARD-BUDGET SWEEP RESULTS (2026-07-19/20)

The fallback ran on the Modal budget app. History: the v1 app (vLLM 0.14.0 +
custom `ThinkingBudgetLogitsProcessor`) banked unlim/512/256 but its forced-close
path **wedges generation when natural thinking overshoots the cap** (budgets
≤128: ~100% of turns hang; 73 consecutive failures). Fixed by `serve_b200_budget_v2.py`:
**vLLM 0.25.1 + native `thinking_token_budget`** (upstream since v0.21, exact
enforcement: 32→34, 64→67, 128→143 observed), super_v3 reasoning parser, same
BF16 ckpt/TP2/fp8-KV/nano template. New URL `daily--nemotron-super-b200-budget-serve.modal.run`
(old kwindla-- v1 app left to idle). Gotcha: 0.25 renames the API field
`reasoning_content` → `reasoning`. Client: `MTE_VLLM_NATIVE_BUDGET=1`.

**Dose-response (n=6/cell; v1-custom cells marked †; Modal network latency
included — subtract the known Modal-vs-VPC delta for absolute comparisons):**

| budget | pass% | TTFAT P50 | TTFAT P95 |
|---|---|---|---|
| unlimited† | 98.9 | 2670ms | 8171ms |
| 512† | 97.8 | 2531ms | 5027ms |
| 512 (native) | 96.1 | 2083ms | 3435ms |
| 256† | 97.8 | 2676ms | 3256ms |
| **128** | **98.3** | **1532ms** | 3417ms |
| **64** | **97.2** | **1218ms** | **1520ms** |
| **32** | **96.7** | **1071ms** | **1228ms** |

**Findings:**
1. **Accuracy is FLAT (96–98%) from unlimited down to budget 32, while median
   TTFAT halves (2670→1071ms) and P95 collapses (8171→1228ms).** There is no
   accuracy knee down to 32. Small hard budgets strictly dominate large ones on
   this benchmark: **budget 128 = 98.3% @ 1532ms** matches unlimited accuracy
   within noise at ~1.1s lower median; budget 32 holds 96.7% at a 1.2s P95.
2. **Tool discipline survives scarcity**: 0 `end_session` aborts in 74 attempts
   at budgets 32–512 with thinking on (2 stray network FAILs). The June-era
   "low budgets break tool discipline" does not reproduce on this stack.
3. **Filler-under-scarcity: DEAD.** b64+48dashes = b64 exactly (97.2 @ same
   latency); b32+48dashes = 93.3 vs 96.7 (−3.4). Filler does not substitute for
   serial thinking even when thinking is rationed; at the scarcest it hurts.
   Combined with Phase 0 (filler hurts at abundant thinking), there is NO regime
   in which filler helps Nemotron-Super.
4. v1-custom vs v2-native at 512: 97.8 vs 96.1 (ranges overlap, consistent);
   v2 ~450ms faster at the median (vLLM 0.25 perf).

**Recommended config: `thinking_token_budget=128` (98.3% @ 1532ms Modal-P50,
sub-1s after VPC network adjustment) or 64 for the tightest tail.**
