# Inkling on BaseTen — integration reference (for gb-benchmarks)

Everything needed to run `thinkingmachines/inkling` on BaseTen from another repo.
Validated 2026-07-15 against the live endpoint.

## Endpoint & auth

- **OpenAI Chat Completions compatible** — use any OpenAI client, no BaseTen SDK.
- **base_url:** `https://inference.baseten.co/v1`
- **model:** `thinkingmachines/inkling`
- **auth:** `BASETEN_API_KEY` (already in `gb-benchmarks/.env`) as the OpenAI `api_key` / `Authorization: Bearer`.

```python
from openai import OpenAI
client = OpenAI(api_key=os.environ["BASETEN_API_KEY"],
                base_url="https://inference.baseten.co/v1")
```

## The model

Thinking Machines **Inkling** — 975B-param MoE (~41B active), 1M context, reasons
natively over text/image/audio. **Serverless on BaseTen** (pay-per-token shared
endpoint). Note for comparison: it is **NOT** serverless on Together (that 400s,
demanding a dedicated endpoint); Fireworks/Modal/Databricks also host it.

## Reasoning ("thinking") effort — the key control

- Parameter: **`reasoning_effort`**, a top-level OpenAI-standard field BaseTen honors.
  Levels: **`none` · `minimal` · `low` · `medium` · `high` · `xhigh` · `max`**.
  (`extra_body={"chat_template_kwargs":{"reasoning_effort": ...}}` also works. There
  is also an underlying continuous float 0.0–1.0; TM report evals at `effort=0.99`.)
- **Reasoning output is a separate field:** `message.reasoning_content` (streaming:
  `delta.reasoning_content`), distinct from `message.content` / `delta.content`.
  Length is in `usage.completion_tokens_details.reasoning_tokens`.

## Gotchas (read before benchmarking)

1. **`reasoning_effort="none"` folds the chain-of-thought into `content`** — the
   visible answer becomes verbose ("shows its work"), and `reasoning_tokens=0`.
   Use **`low` or higher** to keep reasoning in `reasoning_content` with a clean
   answer.
2. **With tools in the request, BaseTen batches reasoning server-side.** It emits
   the entire chain-of-thought as a *single* `reasoning_content` delta that arrives
   *simultaneously* with the first `content` token. Consequence: you **cannot**
   separately time "first thinking token" vs "first answer token" when tools are
   present — the whole thinking latency is baked into the one time-to-first-token.
   (Without tools it streams reasoning incrementally, first reasoning ~270ms →
   first content ~540ms.) A benchmark that always sends tools will see
   raw-TTFB ≈ TTFAT.
3. **`max_tokens` must have headroom** — reasoning counts against it. We use
   **16384**; a low cap truncates the answer mid-thought at higher effort.
4. **temperature:** Thinking Machines' reference example uses **temperature=1**.

## Recommended config for gb-benchmarks

```python
resp = client.chat.completions.create(
    model="thinkingmachines/inkling",
    messages=messages,
    tools=tools,                      # benchmark tools
    reasoning_effort="low",           # sweep none/low/medium/high/max as desired
    max_tokens=16384,
    temperature=1.0,
    stream=True,
    stream_options={"include_usage": True},
)
# Per chunk: answer  = delta.content
#            thinking = delta.reasoning_content   (ignore for the transcript / TTFAT)
#            length   = usage.completion_tokens_details.reasoning_tokens
# TTFAT (voice-relevant latency) = time to first chunk whose delta.content is non-empty,
#   NOT the first chunk overall (that would be the reasoning delta with no tools).
```

## Findings vs effort (aiwf_medium_context)

Thinking effort here is a **pure latency cost with no accuracy gain**. `none`/`low`/`max`
are 10-run aggregates; the intermediate levels are 2 runs each (median trend only).

| effort | runs | pass% (mean, range) | TTFT P50 | TTFT P95 | TTFT max | reasoning tok (med) |
|---|---|---|---|---|---|---|
| none | 10 | 94.7% (87–100) | 917ms | 1674ms | 3453ms | 0 (folded into answer) |
| minimal | 2 | 93.3% | 1688ms | — | — | 76 |
| low | 10 | 96.3% (93–100) | 1559ms | 3464ms | 4717ms | 66 |
| medium | 2 | 93.3% | 2026ms | — | — | 118 |
| high | 2 | 95.0% | 2231ms | — | — | 143 |
| xhigh | 2 | 93.3% | 2326ms | — | — | 176 |
| max | 10 | 93.7% (90–97) | 2486ms | 5954ms | 12779ms | 162 |

- Median TTFT climbs ~920ms → ~2490ms none→max; reasoning tokens 0 → ~160.
  **Pass rate stays flat ~94–96%** — no effort trend (`low` marginally best and
  most consistent). Failures are tool-use/instruction; KB grounding is perfect.
- **The tail is the real killer:** TTFT P95 goes 1.7s (none) → 3.5s (low) → 6.0s
  (max), worst-case up to **12.8s at max**. Higher effort = a much fatter latency
  tail (an occasional turn thinks a lot).
- **Not voice-latency-viable at any effort** (even `low` P50 ~1.5s / P95 ~3.5s vs
  the ~700ms bar) — strong model, wrong latency envelope, like Fable 5.
- **`low` is the best pick** if you run one: highest mean pass (96.3%), tightest
  floor, and a far better tail than `max` (P95 3.5s vs 6.0s; max 4.7s vs 12.8s).
