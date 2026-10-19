# Fresh Baseten Ultra comparison

User authorized October 18, 2026: first test thinking off and on through Baseten,
judge the results, and then decide next steps. No historical captures are reused.

Use the released `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B` through Baseten's
shared Model API at `https://inference.baseten.co/v1`. Request `reasoning_effort`
`none` for off and `high` for on, with no additional template toggle or numeric
thinking budget. Both arms use temperature 1.0, top-p 0.95, max_tokens 16384,
no filler, identical benchmark instructions, knowledge base, tools, and 30
scripted user turns. The completion cap includes reasoning and answer tokens.

Freeze 30 temporal pairs, each containing one off and one on conversation, with
pair order randomized using seed 20261018. Collect the 60 slots sequentially,
one conversation/request stream at a time, using the existing portable collector.
Keep normal recovery nudges, tool deduplication, and tool-result response behavior.
Use the same 180-second idle timeout for both arms. Space request starts by
at least three seconds in each conversation, before starting API latency metrics.
After HTTP 429 before streaming begins, retry up to three times with 10/20/40
second waits; these waits remain included in the affected request's latency.
Log each pacing delay and retry. Never retry a response after streaming begins. Freeze implementation and
configuration hashes before collection. Do not tune prompts or settings based on
model performance. Direct streaming and full-harness smokes are diagnostics and
do not enter the registered cohorts.

Each slot retains its first infrastructure-valid model response. A subsequent
model-caused incomplete conversation stays in the cohort; missing future turns
are failures in the fixed 900-turn denominator per arm. Infrastructure-invalid
attempts retain their logs and are replaced without score-based selection.
Report attempt counts, completion counts, and completion-conditional quality
separately. Do not top up or silently drop incomplete model conversations.

After collection, freeze every selected transcript's hash. Judge every observed
scripted turn, including tool-only turns, with `claude-opus-5-5`, medium effort,
adaptive thinking, no tools/project settings, and the same frozen text rubric as
the README refresh. Hide generating-model identity, arm, paths, and old scores.
Exclude explicit reasoning sections with the existing `think-sections-v1`
normalization and preserve raw captures. Accept the first structurally valid
grade per input. Preserve all API errors and usage. On Anthropic credit/balance
failure, wait at least 90 seconds and retry; ordinary transient failures use
bounded backoff. Never manufacture a grade for a provider-rejected input.

Recalculate content-aware TTFAT and raw first-chunk timings from per-turn source
values; exclude each conversation's first scripted turn and include recoveries.
Publish positive finite P50, linear P95, maximum, observation/missing counts,
strict quality and component errors, plus 20,000 whole-conversation bootstrap
intervals. Record request usage and judge costs. Keep this as a separate study;
do not replace the existing README Ultra rows before reviewing these results.

Documentation checked before execution:

- [Baseten Ultra model](https://www.baseten.co/library/nvidia-nemotron-ultra/)
- [Baseten reasoning controls](https://docs.baseten.co/inference/model-apis/reasoning)

The docs describe `none`/`high` as reasoning off/on, and do not document a numeric
Ultra reasoning cap. Preflight verifies the observed response and streaming path.

## Operational pilot and prospective restart

Before any judging, the initial unpaced batch hit explicit HTTP 429 throttling.
The entire batch (successful and incomplete cases alike) was stopped and excluded;
see [pilot archive](../../README.md). The effective workspace
limits are 120 RPM and 500,000 combined TPM; see smoke/effective-limits.json and
[Baseten limits](https://docs.baseten.co/inference/model-apis/pricing-and-limits).
This restart freezes the same schedule, prompts, tools, model and sampler with
transport-only pacing and bounded retries. No pilot grades were requested.
Remaining infrastructure failures after retries are retained as incomplete
registered conversations, visibly identified in the report; no selective top-ups.

[Evidence storage and local audit instructions](../../README.md).
