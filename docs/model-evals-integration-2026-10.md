# Model evaluation integration assessment — October 16, 2026

The `integration/model-evals-2026-10` branch combines the completed model
campaigns, their supporting harness changes, and the independent judge parser
fix. Its starting point is `agent/pipecat-1.10-speech-latency` at `49ff7a2`,
which contains eleven commits beyond `origin/main` at `ba75da6`.

## Included work

| Source | Included changes |
|---|---|
| `agent/pipecat-1.10-speech-latency` | Pipecat 1.10, GPT-Live and Gemini 3.8 Live support, consolidated September speech results and prompts, capture adapters and offline audits, text routing, OpenRouter controls, latency corrections, filler-study follow-ups, and documentation cleanup. |
| `step3a-judge-parser` at `e21dcfb` | Structural selection of the unique judge JSON result, including responses surrounded by prose or Markdown; ambiguous results fail explicitly. The current judge model and rubric are preserved. |
| Uncommitted work in `codex/gemini-3.7-flash-evals` | Gemini 3.7 Flash low and Grok 4.6 low frozen N=30 campaigns, model routing, stateful xAI Responses continuation, raw latency metrics, leaderboard entries, and chart updates. |

The combined starting point already incorporates `openrouter-controls`,
`speech-support`, `speech-results`, `text-model-routing`, `filler-study-aug`,
`ttfat-first-response`, `ttfat-reanalysis`, and `docs-cleanup`. These are alternate
source branches for the same work, rather than additional PR requirements.
It also supersedes the older `agent/pipecat-1.10-live-speech` integration.

Integration retains the newer historical latency corrections and shared
first-response filtering helper. The imported text-model work does not replace
them with the older values or duplicate filtering implementation. Speech
publication data and prompt identities remain unchanged. Frozen campaign
source hashes describe the original captures, not this integrated checkout.

## Deferred work

| Source | Assessment |
|---|---|
| `ttfat-pilot-instrumentation` at `349ebf4` | A separate research/instrumentation project: 37 changed files and approximately 13,000 added lines, including experimental post-tool lifecycle handling. Its runner pins local checkout paths, a server on port 30190, and a shaping commit. The pilot and its locally retained analysis/review artifacts require a separate portability and publication review; the completed model campaigns do not depend on this code. |
| `ops/xai-grok46-medium/` in the Gemini worktree | Incomplete N=30 campaign: the retained report covers seven conversations and slot eight has an unfinished attempt record. It is not published as a completed leaderboard cohort. |
| Uncommitted changes in the original speech checkout | Independent LiveKit/Meta provider routes and PhoneLLM reasoning-history experiments. The LiveKit adapter imports private `livekit.agents` helpers although the project does not declare that dependency; the Meta route has no corresponding routing tests. These need a separate integration pass. |
| Other untracked captures, ablations, serving studies, and scratch projects | Preserved at their original locations. They are not required inputs to the published campaigns; raw captures and scratch outputs are not added to this branch. |

All source branch refs and original worktrees, including their uncommitted
changes, were preserved. This assessment does not claim that the deferred
instrumentation or provider experiments are invalid; they are independent work
with additional integration requirements.

## Verification

- The starting combined branch passed all 207 existing tests.
- A fresh integration environment was installed with `uv sync --locked --group dev`;
  `uv lock --check` passed.
- `uv run --locked python -m pytest -q` passed all 224 tests. Coverage includes
  speech publication counts and prompt identities, latency filtering, provider
  routing, judge parsing, and stateful Responses continuation for normal replies
  and single/parallel tool outputs. Existing stateless requests retain history.
- Each new text campaign has canonical slots 1–30 and all original transcript,
  judged-turn, and summary inputs available in its source worktree. Both analysis
  scripts were replayed against those inputs into temporary directories; their
  result JSON matched the frozen aggregates exactly. No model calls or judging
  requests were made.
- The README and standalone leaderboard retain the same 49 text-model rows; the
  SVG chart was regenerated from the integrated README.
- `git diff --check` passed.

The new text results are Gemini 3.7 Flash low at 900/900 strict turns and Grok 4.6
low at 740/900. GPT-Live and Gemini 3.8 Live results remain in the
[September speech report](speech-results-2026-09/REPORT.md), whose accuracy is
conditional on completed conversations and whose failure counts are separate.
