# Blog post notes — GLM-5.1 and Gemma 4 31b on Lilac

Notes from running `aiwf_medium_context` against `zai-org/glm-5.1` and
`google/gemma-4-31b-it` via Lilac's OpenAI-compatible chat completions endpoint
(`https://api.getlilac.com/v1`). Date: 2026-04-30.

## Headline (10 runs each, no thinking)

| Model | Pass Rate | Turn Pass | Tool Use | Instruction | KB Ground | TTFT Med | TTFT P95 | TTFT Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zai-org/glm-5.1 | 95.7% | 287/300 | 288/300 | 292/300 | 300/300 | 845ms | 14520ms | 43878ms |
| google/gemma-4-31b-it | 95.0% | 285/300 | 285/300 | 285/300 | 300/300 | 358ms | 1111ms | 2129ms |

GLM-5.1's TTFT P95/Max are dominated by the cold-start first run.
Excluding run 1 (warm path only, 9 runs): **822 ms median / 1524 ms P95 / 4229 ms max** —
that's the steady-state GLM-5.1 latency on Lilac.

## Per-tool-turn pass rate

| Turn | Tool | GLM-5.1 | Gemma 4 |
|---|---|---:|---:|
| 11 | `submit_session_suggestion` (1st) | **10/10** | **10/10** |
| 12 | `submit_session_suggestion` (2nd) | **10/10** | **2/10** |
| 15 | `submit_dietary_request` | **10/10** | 6/10 |
| 17 | `request_tech_support` | 9/10 | 9/10 |
| 24 | `vote_for_session` | **10/10** | 9/10 |
| 29 | `end_session` | **0/10** | **10/10** |

## Mirror-image failure modes

**GLM-5.1** nails 5 of 6 tool calls perfectly across the live conversation
(50/50 across turns 11/12/15/24, plus 9/10 on tech support). The single failure
mode is `end_session`: when the user says *"I just wanted to say the conference
was great. I don't have anything else,"* the model produces a friendly closing
text but does not call `end_session()`. **10 of 10 runs missed it.** The closing
text is well-formed; the tool simply isn't invoked.

This is a regression vs. the older `glm-5-fp8` (Modal-hosted sglang) row in our
README, which scored 99.7% — meaning it caught this turn almost perfectly. We
can't yet say whether the regression is in the model itself or in Lilac's
serving config (precision, batching, tool-call template), without testing
GLM-5.1 against another provider.

**Gemma 4** is the mirror image. It nails the *first* `submit_session_suggestion`
(turn 11, 10/10), then immediately afterward when asked for a *second* similar
suggestion at turn 12, it produces an affirmation like *"I've submitted your
second suggestion"* without actually calling the tool. **8 of 10 runs missed
it.** This is the same "words without action" failure mode we documented for
`gpt-5.4-mini` at no-reasoning: the model batches a verbal acknowledgment but
not a tool invocation. Gemma 4 also drifts on the very next tool-call turn
(`submit_dietary_request`, turn 15: 6/10) before recovering for the rest of the
conversation, including a clean 10/10 on `end_session`.

## Why "both at 95%" is misleading

Headline pass rates are nearly identical (95.7% vs 95.0%), but the qualitative
picture is opposite:

- **GLM-5.1** is a strong tool-using assistant that never hangs up.
- **Gemma 4** hangs up cleanly but flakes on back-to-back tool calls.

System-prompt nudges would likely fix both — for GLM-5.1, an explicit reminder
that user-completion language means call `end_session`; for Gemma 4, a reminder
to issue the tool call alongside any *"I've submitted X"* affirmation. But the
benchmark's value is precisely measuring robustness without that hand-holding,
so these are real signals about each model's default behavior.

## TTFT to flag back to Lilac

For GLM-5.1, the very first run (after a quiet period) had **median TTFT 15.6 s
and max 43.9 s**. The next request to the same model dropped to **1020 ms
median** — a 15× improvement from a warm container.

Generation throughput is fine: a turn that produced ~117 output tokens spent
~37 s in TTFT and well under a second in actual generation. Whatever is slow
on the cold path is server-side request-routing or model-loading, not TPS.

Gemma 4 showed no comparable cold-start: 358 ms median TTFT held steady across
all 10 runs, max 2129 ms. The infrastructure clearly *can* be fast on this
account; GLM-5.1's autoscaling appears to be the bottleneck.

## Thinking-on results (10 runs each)

| Model | Pass Rate | Turn Pass | Tool Use | Instruction | KB Ground | TTFT Med | TTFT P95 | TTFT Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zai-org/glm-5.1 (thinking) | 95.9% | 281/293 | 281/293 | 285/293 | 293/293 | 916ms | 1508ms | 3282ms |
| google/gemma-4-31b-it (thinking) | 95.2% | 239/251 | 244/251 | 239/251 | 251/251 | 1019ms | 1746ms | 15209ms |

Denominators below 300 because of pipeline idle-timeout truncation:
- GLM-5.1 thinking: 1 of 10 runs truncated at turn 23.
- Gemma 4 thinking: 2 of 10 runs truncated at turns 4 and 7.

Headline pass rates are nearly identical to the no-thinking results (95.7% / 95.0%).
But the per-turn breakdown tells a much more interesting story.

## Per-tool-turn comparison: thinking helps Gemma, doesn't fix GLM

| Turn | Tool | GLM no-think | GLM think | Gemma no-think | Gemma think |
|---|---|---:|---:|---:|---:|
| 11 | `submit_session_suggestion` (1st) | 10/10 | 10/10 | 10/10 | **8/8** |
| 12 | `submit_session_suggestion` (2nd) | 10/10 | 10/10 | **2/10** | **8/8** ← fixed |
| 15 | `submit_dietary_request` | 10/10 | 10/10 | 6/10 | **8/8** ← fixed |
| 17 | `request_tech_support` | 9/10 | 9/10 | 9/10 | **8/8** |
| 24 | `vote_for_session` | 10/10 | 9/9 | 9/10 | **8/8** |
| 29 | `end_session` | **0/10** | **0/9** ← unchanged | 10/10 | **8/8** |

**Two clean findings:**

**1. Thinking does not fix GLM-5.1's `end_session` blind spot.** Across 19
attempts that reached turn 29 (10 no-thinking + 9 thinking), the model called
`end_session` exactly **0 times**. The closing text it generates is graceful and
appropriate — it just never invokes the tool. Reasoning effort doesn't change
this. It's a regression vs the older `glm-5-fp8` (~99.7% pass rate, presumably
caught this turn correctly).

**2. Thinking *completely* fixes Gemma 4's back-to-back tool-call problem.**
The "words without action" failures at turns 12 and 15 (no-think: 2/10 and 6/10)
become **8/8 and 8/8** under thinking. Across the 8 valid Gemma 4 thinking runs,
every required tool call is issued correctly. The remaining quality gap is
small (instruction-following slips on a few non-tool turns) and the model
resembles a near-perfect tool user when given reasoning budget.

## Reliability tradeoff for Gemma 4 thinking

The catch: **2 of 10 Gemma 4 thinking runs were truncated** by our pipeline's
45-second idle timeout. Looking at the TTFT distribution: median 1019 ms, P95
1746 ms, but max **15209 ms** on the runs that completed — and presumably worse
on the runs that timed out. Some single turns under thinking exceed 45s
end-to-end response time, which kills the benchmark run.

GLM-5.1 thinking is much tighter (max TTFT 3282 ms) and only truncated 1 of 10
runs. Gemma 4 thinking, when it works, is excellent. When it doesn't, it
produces no usable transcript past the turn that hung.

For voice agent use cases, this matters more than the headline pass rate:
**a single >45s turn is a session killer.** A user will hang up. So Gemma 4
thinking's quality improvement comes with a reliability tax that the no-think
mode does not have.

## Updated mental model of the four configurations

| | Quality | Latency floor | Latency tail |
|---|---|---|---|
| GLM-5.1 no-think | 95.7% (loses end_session) | 822ms warm | 4229ms warm |
| GLM-5.1 thinking | 95.9% (still loses end_session) | 916ms | 3282ms |
| Gemma 4 no-think | 95.0% (loses tool-call follow-through) | 358ms | 2129ms |
| Gemma 4 thinking | 95.2% (near-perfect tool calls) | 1019ms | **15209ms+, truncates runs** |

For voice agents:
- **Gemma 4 no-think** is the fastest option but flakes on rapid tool calls.
- **GLM-5.1 (either mode)** has a stable latency tail and never hangs the
  pipeline, but won't end sessions cleanly.
- **Gemma 4 thinking** has the best per-turn quality but unacceptable tail
  latency.

There isn't a single "right" answer in this matrix — the choice depends on
which failure mode is cheapest to mitigate at the application layer.

## Things to flag back to Lilac

1. **GLM-5.1 cold-start TTFT.** First request after a quiet period took
   **15.6 s median, 43.9 s max** — 15× the warm path. Once warm, it's tight.
   Looks like autoscaling or container-cold-load.

2. **Gemma 4 thinking long-tail TTFT.** Median 1019 ms is fine, but max 15.2 s
   on completed runs (and presumably higher on the runs we lost). Some single
   turns appear to take much longer than the others, with no obvious correlation
   to which turn. Worth Lilac investigating whether thinking-mode reasoning is
   hitting per-request limits or queue contention.

3. **Thinking tokens not separately reported.** Lilac's chat completion
   responses don't break out reasoning tokens — `thinking_tokens` is null
   even when `chat_template_kwargs.enable_thinking` is set. The thinking
   appears to happen (response sizes ~3× longer; quality improves) but it's
   bundled into `completion_tokens`. Useful for billing but obscures latency
   debugging.

## Caveats

- 10 runs is a small sample; per-run variance on this benchmark is ~5pp.
- Lilac infra was tested on a single afternoon; cold-start and tail-latency
  behavior may differ off-peak.
- KB grounding was perfect across all 4 configurations — these models know the
  facts. Differences are entirely in tool-call discipline.
- The "GLM-5 was better" comparison vs the README's `glm-5-fp8` row is
  confounded by infrastructure: that row ran on Modal-hosted sglang at FP8;
  this run is on Lilac at whatever precision they serve. Until we test
  GLM-5.1 against a second provider, we can't fully separate model regression
  from serving-config differences.
