# Gemini 3.8 Flash low — standard AIEWF campaign

Date: October 17, 2026. Base: `origin/main` at `5dfcd96`.

Use the existing `aiwf_medium_context` text benchmark, its complete knowledge
base and 30 scripted turns, and the shared Claude Opus 5 judge. The fixed
schedule contains 30 conversations (900 scheduled turns) at concurrency one.
Smoke conversations are excluded from this cohort.

## Request configuration

- Exact model: `gemini-3.8-flash`, Google Developer API `generateContent`.
- `thinking_level=low`, `include_thoughts=true`; low is the model's supported
  floor. Medium is the provider default; minimal and disabled are unsupported.
- Provider-default sampling; no filler tokens; full conversation history.
- Explicitly omit `max_output_tokens` using `MTE_GOOGLE_MAX_TOKENS=""`.
  Pipecat otherwise supplies 4096, including thinking. The model's own output
  limit still applies. Older campaigns and unset settings are unchanged.
- Pin `pipecat-ai==1.12.0` and `google-genai==2.31.0`, the latest stable Google
  SDK verified from PyPI on the campaign date. Freeze `pyproject.toml`, the
  lockfile, and the runtime sources in the campaign hash manifest.
- Retain the standard recovery and tool deduplication settings, with
  `MTE_TOOL_RESULT_RUN_LLM=0` and a 45-second text idle timeout.

## Eligibility, judging, and reporting

Use the existing portable collector and its `first_valid_response` policy.
Freeze the configuration, schedule, benchmark, and implementation before
collection. Retain all attempts. Infrastructure-invalid attempts without a
valid first response may be replaced; model-caused short conversations after
a valid response remain canonical. Missing future scheduled turns count as
failures in the fixed 900-turn denominator. Do not top up based on scores.

Judge each canonical conversation with the unchanged shared rubric. Require
the judged turns to match its observed scheduled prefix. Strict pass requires
tool use, instruction following, and knowledge-base grounding all to pass.
Report completion counts, component errors, and a deterministic 20,000-resample
conversation-cluster bootstrap interval alongside accuracy.

Report content-aware TTFAT and separate raw TTFT. Exclude scripted turn zero
from latency; retain later scripted and recovery responses. Freeze canonical
run paths, source hashes, aggregate JSON, analysis code, and the report.
Original transcripts, logs, and judge responses remain local under `runs/`.

The direct smoke checks provider-reported model identity and low reasoning;
the complete Pipecat smoke checks 30-turn continuation, tool delivery through
`end_session`, provenance, and the uncapped configuration before serving is
marked verified. The first attempted smoke encountered the Pipecat 1.12 adapter
signature change before receiving a model response and is retained as a
harness diagnostic, separate from all canonical results.

A subsequent complete smoke verified conversation flow but missed a Pipecat
1.12 timing change: `start_ttfb_metrics` moved from the streaming helper into
the upstream context processor. The custom processor now explicitly starts it
on every request. The initial partial collection is retained under
`smoke/missing-ttfat-diagnostic/` and excluded from the final cohort. A fresh
full smoke must have positive TTFAT and raw TTFT on every recorded turn before
restarting collection with new source hashes.

Sources: [model and supported thinking levels](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash),
[Google SDK release](https://pypi.org/project/google-genai/2.31.0/),
[Pipecat release](https://github.com/pipecat-ai/pipecat/releases/tag/v1.12.0),
and [published introductory pricing](https://ai.google.dev/gemini-api/docs/pricing).
Cost estimates use $0.75/M uncached input, $0.075/M cached input, and $3.75/M
output including thinking through December 31, 2026; these are estimates rather
than invoice charges.

[Evidence storage and local audit instructions](../README.md).
