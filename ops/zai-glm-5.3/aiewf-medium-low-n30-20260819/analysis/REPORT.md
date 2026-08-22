# GLM-5.3 (Z.AI, low reasoning) — aiwf_medium_context campaign, 2026-08-19

## Configuration

| | |
|---|---|
| Model | `glm-5.3` |
| Endpoint | `https://api.z.ai/api/paas/v4` (Z.AI first-party, OpenAI-compatible) |
| Service | `--service zai` → `LoggedZaiLLMService` |
| Reasoning | `thinking={"type":"enabled"}`, `reasoning_effort=low` |
| Sampling | temperature 1.0, provider-default top-p, `max_tokens=16384` |
| Filler | none |
| Cohort | 30 conversations, fixed 900-turn denominator |
| Judge | `claude-opus-4-5`, `claude-agent-sdk-v4-turn-taking` |

**There is no thinking-off arm.** `reasoning_effort` accepts only `low`, `high`,
and `max`; `none`, `minimal`, `medium`, and `xhigh` are all rejected with HTTP
400 code 1210 ("This model always engages in thinking and cannot be disabled").
This row is the model's lowest available reasoning setting and is **not**
comparable to the thinking-off rows on the leaderboard, which have reasoning
genuinely disabled. GLM-5.3 still emitted ~13 thinking tokens per turn here.

## Results

| metric | value |
|---|---|
| Strict turn pass | **888/900 (98.7%)** |
| Bootstrap 95% CI | 97.9% – 99.3% (whole-conversation resampling, 100,000 samples) |
| Conversations completing all 30 scripted turns | 30/30 |
| Tool-use errors | 12/900 (1.3%) |
| Instruction-following errors | 12/900 (1.3%) |
| KB-grounding errors | 0/900 (0.0%) |
| Turn-taking errors | 0/900 (0.0%) |
| TTFAT P50 / P95 / max | 2376ms / 4761ms / 8007ms |
| Run outcomes | 30/30 clean, 30/30 judged, zero replacements |

Latency follows the leaderboard convention: the first scripted response of each
conversation is excluded.

## Latency caveat: measure-window dependence

This endpoint's response time drifted downward substantially during the session
in which these cohorts were collected. On the companion GB port-to-port
benchmark, no-reasoning TTFB P50 fell from 4,554ms (17:32–17:49Z) to 1,755ms
(18:40–19:49Z), a 41% change for an *identical* configuration, with cached-input
share flat at ~97% — so it is provider-side load, not prefix-cache warming.

Consequently this row's TTFAT is a point-in-time measurement, not a stable model
property. Do not compare it against rows collected on other days. Streaming is
genuinely incremental (a `low` completion spread its chunks over 0.72s, a `max`
completion over 5.24s), so the numbers are not buffered-response artifacts.

## Instrumentation: what had to be fixed, and how it was validated

`LoggedZaiLLMService` subclasses `VLLMOpenAILLMService`. Three defects were found
by inspecting live measurements — none of them surfaced as an error:

1. **Raw TTFB was never emitted.** Only `LoggedLilacLLMService` and its
   subclasses emit `RawTTFBMetricsData`; the vLLM base does not. `raw_ttfb_ms`
   was simply absent from the first smoke's transcript.
2. **The raw-TTFB clock started after the awaited request**, which is where the
   HTTP request is issued and the stream opens, so every turn measured ~0ms.
3. **Thought frames are gated on a Qwen model-name match** that `glm-5.3` fails.
   The reasoning aggregator was installed but starved, silently dropping
   `reasoning_content` from tool-call history — exactly what Z.AI's
   preserved-thinking contract warns against.

The Lilac line has both metrics but pushes no thought frames, and the reasoning
aggregator is driven entirely by those frames, so the fix keeps the vLLM base and
adds the missing metric locally. All three are covered by regression tests in
`tests/test_zai_routing.py`, including one asserting that the clock statement
precedes the awaited request — a mocked test passes either way.

**Validation that the metric separates, rather than merely runs:** a matched
single-conversation probe at `max` effort measured raw TTFB P50 2,621ms against
TTFAT P50 3,805ms — a 715ms median thinking delay (P95 2,965ms) over 4,416
thinking tokens. At `low` the delay is ~0ms, which is genuine rather than broken:
~13 reasoning tokens per turn arrive in the same network flush as the first
answer token.

## Artifacts

- Run dirs: `../run-dirs.txt` (30 entries)
- Per-conversation ledger: `../results.tsv`
- Per-conversation logs: `../logs/` (local only — `*.log` is gitignored)
- Campaign script: `scripts/run_zai_glm53_campaign.sh low 30 n30`
- Companion GB port-to-port campaign and runbook:
  `gb-benchmarks/port-to-port/docs/glm53-zai-runbook.md`
