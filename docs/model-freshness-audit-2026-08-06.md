# Text-model production freshness audit — 2026-08-06

## Scope

This audit treats the README text-model table as a current-production view, not
as a permanent historical leaderboard. Historical transcripts and dedicated
study reports remain available even when a row leaves the summary table.

The retained table has 33 configurations. The executable manifest is
`ops/model-freshness-2026-08-06/run_campaign.py`; it reproduces the published
thinking, filler, sampling, endpoint, and provider settings and records one
complete 30-turn conversation per retained row. It also retains the removed
GLM-5, Lilac Gemma 4 31B, Inkling Small, and GPT-5.4-mini freshness probes for
auditability. OpenAI Pro models are excluded by both the manifest and the
repository model policy.

## Rows removed from the current-production table

### Provider route retired or no longer available

- `kimi-k2.6 Cerebras (thinking)` and `(instant)`: the exact Cerebras model ID
  now returns 404 and is absent from Cerebras's current production catalog.
- `zai-org/glm-5.1` on Lilac: the exact model ID now returns 404 and is absent
  from Lilac's live model list.
- `nova-2-pro-preview`: the exact Bedrock inference profile now returns 404 and
  is absent from the current account catalog.
- `gemini-3.1-flash-lite-preview`: Google lists this preview as shut down on
  2026-05-25 with `gemini-3.1-flash-lite` as its replacement. The old alias
  still answered a direct probe for this account, but a shut-down preview is
  not an appropriate production recommendation. The table already contains
  newer stable Flash-Lite coverage.
- `glm-5 (thinking)` on Modal: the deployment and exact model ID still exist,
  but three independent full-conversation attempts all stopped producing output
  after turn 2 and hit the benchmark's 45-second idle timeout on turn 3. None
  produced a valid `runtime.json`, so there is no complete current cost sample.
- `inkling-small (none)` and `(low)`: `none` completed its current probe, but
  `low` completed 0/10 attempts (eight ended at turn 16 and two stalled). The
  model family is retained in the historical study artifacts, not the current
  production summary.
- `gpt-5.4-mini (medium)` and `(none)`: `medium` completed only 1/6 attempts;
  four attempts ended at turn 13 and one at turn 28. `none` completed 0/4, with
  every attempt calling `end_session` incorrectly on turn 13.
- `lilac/gemma-4-31b-it (thinking off)`: five probe attempts ran to their idle
  timeout after only a few responses and produced no valid full-conversation
  runtime. The last two raised the idle allowance from 45 to 120 seconds, so
  the failure is not explained by the historical row's 43-second latency tail.

### Original self-hosted deployment no longer exists

The following rows depended on exact Modal deployments that are not in the live
workspace app list. Most are also superseded by a newer family member already
in the table. Recreating them on a new serving stack would be a new experiment,
not a freshness rerun of the published row.

- `nemotron-3-ultra (128)` and `(96)`
- `qwen3.5-27b (thinking)` and thinking off
- `qwen3.5-9b (thinking)` and thinking off
- `qwen3.5-4b (thinking)` and thinking off
- `nemotron-3-super-120b (512)`
- `nemotron-3-nano-30b (512)`
- `glm-4.7-flash`

The exact Nemotron Super 512 aggregate came from the removed
`nemotron-super-b200-bf16-v3` endpoint. A different Nemotron Super app exists
today, but it uses a newer vLLM/native-budget stack and is not interchangeable
with the old latency row.

## Replacement-route status

- BaseTen's hosted Model API exposes `moonshotai/Kimi-K2.6`. Its prior Cerebras
  accuracy and latency are not transferred to this route. The replacement row
  uses a new 30-conversation, fixed-900-turn, thinking-off BaseTen campaign:
  845/900 strict turns passed (93.9%; conversation-cluster bootstrap 95% CI
  92.3–95.3%) at 480 ms median TTFAT. Collection produced 30 canonical
  conversations from 41 conversation attempts; the table's score and latency
  exclude recovery rows, while the cost sample includes their billed tokens.
- A dedicated two-H100 BaseTen deployment of the official
  `google/gemma-4-31B-it` checkpoint completed its first 30-turn freshness run
  on the first attempt. The run had complete token and prefix-cache accounting
  and measured 456 ms median TTFAT. This validates the replacement serving
  path, but one run is not an accuracy aggregate, so Gemma 4 31B remains out of
  the summary table pending a new judged multi-conversation campaign. The
  deployment was returned to `SCALED_TO_ZERO` with zero active replicas.

## Retention decisions

- Older OpenAI aliases (`gpt-4o`, `gpt-4.1`, and their mini variants) remain
  because the exact IDs are still in the live API catalog and are useful
  production price/latency baselines.
- Claude Sonnet 4.6 and Haiku 4.5 remain because the exact IDs are supported and
  provide lower-latency or lower-cost comparisons to Claude 5 models.
- Gemini 2.5 Flash remains because the exact stable ID is live and explicit
  `thinking_budget=0` makes it a useful low-latency control.
- Dedicated BaseTen deployments remain only where their exact deployment IDs
  still exist at scale-to-zero and can reproduce the published serving stack.

## Completion and cost-estimation protocol

A freshness run is accepted only when `runtime.json` reports `completed` and
valid, and transcript turns 0 through 29 are all present. Infrastructure or
early-exit attempts are retained in the campaign log but are not used as the
one-row cost sample. Recovery turns beyond 29 are included in token totals when
they were actually billed.

Token coverage is checked on every recorded API call. The cost analysis uses
the accepted conversation only, current public prices, and separate cached
input rates where the provider exposes cached-token counts. Conversation
minutes are estimated from the actual user and assistant text at a documented
speech-rate assumption; they are not the benchmark process's accelerated wall
time. Dedicated GPU rows require an additional utilization assumption and are
reported separately from token-billed APIs.
