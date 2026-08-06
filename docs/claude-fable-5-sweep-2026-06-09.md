# Claude Fable 5 — aiwf_medium_context sweep (2026-06-09)

## Results (10-run aggregates, 300 turns each)

| Config | Pass Rate | Tool Use | Instruction | KB Ground | TTFT Med | TTFT P95 | TTFT Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| low (`effort: low`) | 100.0% | 300/300 | 300/300 | 300/300 | 3535ms | 5148ms | 8815ms |
| default (no effort param ≈ `high`) | 100.0% | 300/300 | 300/300 | 300/300 | 3956ms | 6496ms | 13602ms |

Perfect scores on every judged dimension in both configs — 600/600 turns. Ties
`nemotron-3-ultra (128)` and `claude-sonnet-4-6` as the only 100% rows in the
README table, but with the slowest TTFT of the perfect scorers by a wide margin.

The planned low → medium → high → xhigh effort sweep was gated on TTFT P50 <
1500ms after each level. `low` came in at 3535ms, so the sweep stopped there
(the `default` row, ≈ effort `high`, was run as the out-of-the-box reference).
Even the single fastest turn observed at `low` (1909ms) misses the gate; the
model pays roughly 2s of prefill on this workload before any thinking starts.

## Configuration

- Service: `anthropic` (`LoggedAnthropicLLMService`), pipecat 1.1.0,
  `anthropic` SDK pinned to 0.108.0 (pipecat's floor of 0.49.0 predates
  adaptive thinking and `output_config`).
- Effort: `MTE_ANTHROPIC_EFFORT=low` → `thinking: {"type": "adaptive",
  "display": "summarized"}` + `output_config: {"effort": "low"}`. Unset →
  same thinking config, no `output_config` (model default, effort `high`).
- `max_tokens` 16384 (thinking counts against it; pipecat's 4096 default can
  truncate mid-thought at higher efforts).
- Prompt caching enabled (default).

## Findings along the way

1. **Fable 5 cannot run without thinking.** Adaptive thinking is always on;
   `thinking: {"type": "disabled"}` returns a 400. Confirmed empirically and
   in the adaptive-thinking docs ("thinking is always enabled and cannot be
   disabled"). There is no "no-thinking" row for this model — the closest
   thing is `effort: low`, where the model skips thinking on simple turns.

2. **`display: "summarized"` is load-bearing.** Fable 5 defaults
   `thinking.display` to `"omitted"`: thinking blocks stream with an empty
   `thinking` field plus a signature. Pipecat 1.1.0's `AnthropicLLMAdapter`
   can only rebuild a context thought message into a thinking block when both
   text and signature are truthy — an empty-text thought falls through to a
   role-less dict and the *next* turn crashes with `KeyError: 'role'`. Two
   early runs were lost to this before the harness pinned
   `display: "summarized"`.

3. **TTFT instrumentation had to change.** Upstream `AnthropicLLMService`
   stops the TTFB metric at stream creation — before any tokens.
   `LoggedAnthropicLLMService` now stops it at the first user-visible event
   (text delta / tool_use block) and emits `raw_ttfb_ms` on the first event of
   any kind, matching the `LoggedCerebrasLLMService` / `LoggedGoogleLLMService`
   convention. On thinking turns the visible TTFT ran 1.4–1.9s later than the
   raw value — the old instrumentation would have under-reported latency by
   that much and made effort levels indistinguishable.

4. **Latency shape.** low → default costs only ~420ms at the median but
   doubles the tail (max 8.8s → 13.6s): at high effort the adaptive thinker
   commits to long thinking on hard turns.

## Run tracking

- Allowlists: `docs/ten-run-allowlists/claude-fable-5-{low,default}-2026-06-09.txt`
- Aggregates: `docs/ten-run-aggregates/claude-fable-5-{low,default}-2026-06-09.{txt,json}`
- Driver: `scripts/run_fable5_sweep.sh` (2 excluded pre-fix baseline runs were
  replaced by top-ups 11/12)

## Voice-optimized probe (follow-up, same day)

Both follow-ups were implemented and tested with a single 30-turn probe run
(`runs/aiwf_medium_context/20260609T125622_claude-fable-5_fca26467`):

- `PatchedAnthropicLLMAdapter` (in `services/anthropic_logged.py`) round-trips
  empty-text+signature thinking blocks — docs-blessed ("pass each thinking
  block back exactly as received, including blocks whose thinking field is
  empty") — enabling `MTE_ANTHROPIC_THINKING_DISPLAY=omitted`. An adversarial
  Codex review (via /cx-delegate) produced 8 findings; 3 were fixed
  (signature-less thoughts now filtered before conversion instead of becoming
  phantom "(empty)" assistant turns; `omitted` requires the patched service;
  TTFB gate cleared per measurement cycle) and 5 accepted as pre-existing
  upstream behavior or intentional.
- `MTE_ANTHROPIC_VOICE_STEERING=1` (fable-only, env-gated) appends to the
  system prompt: "This is a conversational voice application. Fast responses
  are important to the user experience. We need to prioritize low latency.
  Always answer directly without deliberating."

**Probe result (effort low + display omitted + steering, 1 run, 30 turns):**

| Metric | voice-optimized | plain low (10-run) |
|---|---:|---:|
| Pass rate | 100.0% (30/30) | 100.0% (300/300) |
| TTFT med | 2980ms | 3535ms |
| TTFT p95 | 5907ms | 5148ms |
| Fastest turn | 2144ms | 1909ms |

Steering worked exactly as the docs promise: median thinking delay fell from
1.4–1.9s to **14ms** (6/30 turns above 100ms), with no measurable quality
cost. But raw time-to-first-byte was 2880ms median, so visible TTFT is now
~97% serving-side prefill. **Conclusion: Fable 5's voice latency on this
workload is bound by serving latency, not reasoning configuration** — no
combination of effort, display mode, or prompt steering can reach the ~1500ms
voice bar when the first byte takes 2–3s.
