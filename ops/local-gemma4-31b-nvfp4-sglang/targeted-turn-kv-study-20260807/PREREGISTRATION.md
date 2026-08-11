# Targeted Gemma KV study preregistration

Frozen before the first targeted GPU replay on 2026-08-07.

The governing design is
[`docs/gemma4-kv-precision-targeted-turn-plan-2026-08-07.md`](../../../docs/gemma4-kv-precision-targeted-turn-plan-2026-08-07.md),
including its incorporated Fable review. This file records implementation-level
choices that were frozen after inspecting the actual historical wire path.

## Primary comparison

- Checkpoint and revision: `RedHatAI/gemma-4-31B-it-NVFP4` at
  `edafdf3dcaef23ff76f75b91edd6a4a975a399cf`.
- Local hardware: one RTX 5090; one request at a time; arms run serially.
- SGLang image:
  `lmsysorg/sglang@sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2`.
- Arms differ only in `--kv-cache-dtype bfloat16` versus `fp8_e4m3`.
- Both arms use `--max-total-tokens 16000`,
  `--swa-full-tokens-ratio 0.35`, no CUDA graphs, no MTP, batch/concurrency
  one, and identical remaining server flags.
- Turns 12 and 15 are separate co-primary outcomes. The outcome is exactly one
  correct target tool call with correct arguments.
- Primary inference is conditional on the 12 frozen real prefixes per turn,
  weighted equally. Each turn's golden prefix is a separately reported
  mechanism probe.
- The same seed and snapshot are paired across KV arms.
- Confirmatory turn tests use Holm adjustment once at the final sample size.

## Wire-level sampler correction

The historical campaign JSON and startup log say `top_k=64`. Inspection of
Pipecat 1.3.0's `BaseOpenAILLMService` and a live construction probe showed
that `InputParams.top_k` is documented as ignored and is not copied into the
HTTP request. Therefore the reproduction body freezes the actual wire behavior:

- temperature 1.0;
- top-p 0.95;
- no top-k field (SGLang's OpenAI default is therefore used);
- max tokens 8,192;
- thinking disabled.

Adding top-k 64 would be a new sampler experiment, not a reproduction.

## Frozen corpus and seeds

- `snapshot-manifest.json` freezes 26 canonical request snapshots and their
  hashes: 12 real prefixes plus one golden prefix for each target turn.
- The real bank contains four prerequisite-valid prefixes from each of the
  BaseTen BF16, local FP8-KV, and local BF16-KV historical cohorts. Selection is
  the lowest four deterministic SHA-256 values and never reads the target-turn
  outcome.
- Turn 15 selection excludes histories that already called the dietary tool on
  turn 14 or did not establish the expected dietary-request state.
- `seed-manifest.json` freezes repeatability, cap-parity, cache-pilot, and
  primary allocations.
- `macro-schedule.tsv` freezes the serial FP8/BF16/BF16/FP8 primary order.

## Gates

1. Prompt-token hashes must match across arms for every snapshot.
2. The compact FP8 server must complete one normal 30-turn benchmark.
3. A 128-case comparison must show outcome parity and no truncation for
   `max_tokens=512` versus 8,192 before the shorter cap is used.
4. Within each arm, 100 snapshot/seed cases must reproduce exactly in warm/warm
   and cold/cold repeats, including cold repeats separated by a restart.
5. Warm requests must achieve at least 90% prompt-token cache hits after an
   explicit prime; salted cold requests must report at most one cached token.
6. The frozen scorer must agree with at least 99.5% of the 900 historical
   target-turn judgments.
7. Any HTTP, parser, integrity, or within-state repeatability failure blocks the
   large campaign until explained.

The scorer's raw agreement is 897/900 (99.667%), so gate 6 passed. All three
disagreements are manually resolved early-call realignment cases: the dietary
tool ran prematurely on turn 14 and the historical judge credited turn 15.
Those histories are excluded from the turn-15 replay bank. The replay scorer
properly requires the target action on the target request.

The historical-geometry FP8 bridge uses 64 golden-prefix seeds and 16 seeds
for each of the 12 real prefixes per turn (256 cases per turn), warm, with the
same seed ranges as the larger compact-geometry pilot. This allocation was
added before either geometry's pilot results were collected and does not alter
any existing seed allocation.

The continuous mechanism probe teacher-forces a pure canonical Gemma 4 tool
call for each snapshot. `build_teacher_tokens.py` uses the checkpoint's pinned
chat tokenizer, first proving that its generation-prompt tokens exactly equal
the frozen server token IDs, then appending the 34-token turn-12 or 27-token
turn-15 canonical call syntax. `/generate` returns prompt log probabilities
starting one token before that suffix, which is required to score the first
`<|tool_call>` decision token. The probe records the canonical sequence mean
log probability, the first-token best alternative and margin, exact token
hashes, and cache counts under both KV/cache states. It is mechanistic and
secondary; it does not replace the preregistered binary primary outcome.

The cache pilot reuses one of the already gated, semantically identical
warm/warm or cold/cold repetitions for golden-prefix seeds 0–49. The pilot
runner therefore collects golden seeds 50–127 plus all preregistered bank
seeds. Analysis combines the repeatability artifact and pilot-extension
artifact, selecting one outcome per `(arm, cache, snapshot, seed)`. This saves
400 redundant GPU requests across the four arm/cache cells without changing
the frozen allocation or looking at outcomes.

## Collection and continuation

- Cache pilot: per turn and arm, 128 golden-prefix seeds plus 32 seeds on each
  of 12 real prefixes, under both warm and cold cache modes (512 seed cases per
  turn/arm/cache mode).
- Main warm campaign: per turn and arm, 512 golden-prefix seeds plus 128 seeds
  on each of 12 real prefixes (2,048 seed cases per turn/arm).
- Continue each snapshot stratum proportionally to 4,096 and then 8,192 only if
  either turn's simultaneous paired interval remains wider than ±2 points.
- The cache difference-in-differences equivalence margin is ±3 points. Cold
  sampling may be expanded based on pilot discordance, without outcome-based
  selection.

Raw request/response events, errors, timing, usage, cache counts, hashes, and
mechanical scores are retained. Request/server failures count as primary
failures under intention-to-test and are also reported separately.

## Gate-driven amendment: deterministic server mode

Amended after the first four FP8 warm-cache smoke requests and before any
BF16 replay, cold-cache replay, pilot, or primary collection. SGLang accepts a
per-request `seed`, but its sampler ignores that seed unless the server starts
with `--enable-deterministic-inference`. In the initial smoke, both repeated
seed groups produced different exact outputs and one changed the binary tool
outcome. The repeatability gate therefore failed as designed; those four rows
are retained in `results/fp8-smoke-warm.jsonl` and excluded from inference.

All causal replay arms will restart with `--enable-deterministic-inference` and
an explicit `--attention-backend triton`. The explicit backend preserves the
historical deployment's Triton attention choice across both KV dtypes; SGLang
otherwise changes the Blackwell deterministic default to FlashInfer. SGLang
will use its PyTorch position-keyed sampler in deterministic mode. This changes
the targeted experiment from the production sampler, so results estimate the
KV effect under the matched deterministic replay configuration. The existing
full-conversation campaigns remain the production-configuration evidence.

The same warm/warm and cold/cold repeatability gates are rerun from scratch.

## Gate-driven amendment: seed-only sampler shim

Amended before any BF16 targeted replay or pilot collection; this supersedes
the preceding broad-deterministic-mode amendment. The broad
deterministic mode could not satisfy this study's cache-state intervention:

- SGLang sets `disable_finished_insert=True` on its radix cache when
  deterministic inference is enabled, so an explicitly primed prompt cannot
  remain warm after its request finishes.
- On this Gemma 4 NVFP4 checkpoint, a 14K-token chat request also stalled for
  more than 180 seconds before prefill when batch-invariant mode was enabled.
  A 15-token control request completed. The stalled requests and server logs
  are diagnostic only and excluded from inference.

The matched arms therefore use the narrow, recorded shim in
`shims/sitecustomize.py`. It makes only
`SamplingBatchInfo.from_schedule_batch()` see deterministic sampling as
enabled, causing SGLang to populate the existing per-request
`sampling_seed`. The server explicitly uses SGLang's PyTorch position-keyed
sampler, but does **not** enable batch-invariant model kernels or disable radix
cache insertion. Attention remains Triton, batch/concurrency remains one, and
all other matched geometry is unchanged.

This is an instrumentation change to make the public OpenAI `seed` field do
what the replay protocol requires; it is not part of either KV treatment. The
same shim is mounted read-only into both KV arms, its source is retained, and
the broad deterministic mode remains available as a diagnostic server option.
The first four FP8 warm-cache seed-only requests reproduced semantically for
both repeated seeds. One pair differed only in SGLang's generated tool-call
UUID, which the repeatability signature now correctly excludes as transport
metadata. Four cold requests also reproduced semantically and reported zero
cached prompt tokens.

### Repeatability signature

The OpenAI streaming route does not expose output token IDs unless logprobs
are requested. Historical benchmark requests did not request logprobs, and
adding that field would change the measured request and serving work. The
within-state gate therefore compares exact decoded assistant text, exact
reasoning text, exact tool names and raw argument strings, finish reasons, and
the mechanical outcome. It excludes only SGLang's post-generation random
tool-call UUID. Exact agreement on this signature is the wire-faithful
substitute for the initially planned output-token-ID assertion; raw SSE events
remain retained for audit.

### Gate-driven amendment: malformed model tool calls

The first hardened BF16 warm-repeatability collection stopped after seed 45
returned a complete, non-truncated `tool_calls` response whose sole call named
`submit_session_suggestion` but contained an empty argument string. The raw SSE
events, finish reason, usage, cache accounting, and hashes were all present.
That is a failed model outcome, not a response-parser integrity failure.

Before observing any BF16-versus-FP8 contrast from the primary experiment, the
scorer was amended to classify a structurally delivered call with unparseable or
missing arguments as `malformed_tool_call`. Collection retains that outcome.
Only a missing/unparseable response message (`response_parser_failure`), a
transport/server error, truncation, or cache-gate failure aborts a stage. The
aborted 91-row collection remains preserved but is excluded from inference;
repeatability restarts under a newly hashed collection plan.

The restarted gate then stopped cleanly at the boundary between the two golden
snapshots because SGLang's full `/get_server_info` document changed after
generation. Inspection showed that the selected configuration was unchanged;
the full document also contains mutable runtime telemetry, including
`internal_states.last_gen_throughput`. Before any primary contrast, the binding
was narrowed to the already enumerated serving configuration plus exact Docker
identity, command, environment, mounts, GPU, network, IPC, and shared-memory
state. The full server-info hash remains in each capture as diagnostic evidence
but is not a configuration invariant. The second partial collection (100 rows)
is preserved and excluded; the complete gate restarts under another new plan.

## Pre-primary independent-audit amendments

Amended after the one-arm FP8 warm cache-pilot extension completed, but before
any BF16 cache-pilot result, any cross-arm pilot contrast, or any primary
collection. An independent GPT-5.6-sol/xhigh code audit found fail-open
collection safeguards and a dependence structure that the first analysis
implementation did not preserve. The amendments are driven by code and design
inspection, not by a BF16-versus-FP8 effect estimate. The completed FP8 warm
extension is admissible only if its retrospective provenance and exact-allocation
audit passes; otherwise it will be rerun.

### Collection and replay integrity

- Every future stage must bind its requested arm to the live server's checkpoint,
  revision, image, KV dtype, attention and sampling backends, pool geometry,
  concurrency, seed shim, container ID, and start time. A stage-specific plan
  freezes the expected request cells and request hashes.
- Rows carry stage, plan, server-configuration, and server-instance identifiers.
  Resume accepts an existing row only after all immutable fields and its request
  hash match; a request ID alone is insufficient.
- A stage audit must prove exact allocation completeness, no extra or duplicate
  logical cells, recomputed request/base/input/event hashes, fresh mechanical
  rescoring, no HTTP/parser/length failures, and the appropriate per-request
  cache criterion.
- Server logs and inspect records are stage-qualified so later restarts cannot
  overwrite provenance.
- The BF16 restart-separated cold repeat remains a blocking gate. For both arms,
  `repeat=0` is the sole inferentially reusable golden row; all later repetitions
  are gate-only.

### Scorer semantics

The successful tool arguments are now frozen to the semantics actually observed
in the 150-conversation historical corpus. Both turns require exactly the two
schema keys and the normalized name `jennifer smith`. Turn 12 accepts exactly
the normalized canonical suggestion, with or without the leading `a session
on`; turn 15 accepts exactly `vegan`. Negations, substring collisions, extra or
missing fields, malformed JSON, wrong tools, and multiple calls are failures.
Adversarial unit tests cover those branches. Historical agreement remains
897/900 (99.667%), and the completed FP8 warm rows will be rescored from raw
completions under this frozen version.

### Seed-cluster inference and sequential precision

The same numeric seed is reused across all 12 prefixes, and SGLang's sampler
keys its Gumbel stream by seed and absolute token position. Prefixes therefore
do not supply 12 independent realizations of a seed. The point estimand remains
the equal-prefix mean, but primary uncertainty jointly resamples numeric seeds:
for each seed, first average the 12 paired prefix differences, then bootstrap
those seed clusters. Independent within-prefix resampling is retained only as a
sensitivity analysis. Final Holm-adjusted primary p-values use a
seed-cluster-robust score test rather than pooled exact McNemar; pooled McNemar
is descriptive only.

The three permitted cumulative looks are frozen at 2,048, 4,096, and 8,192
cases per turn and arm. Incremental allocations are explicit in
`seed-manifest.json`, and every level follows the FP8/BF16/BF16/FP8 half-block
schedule in `macro-schedule.tsv`. The precision interval spends the family-wise
5% error budget conservatively across two turns and three possible looks
(Bonferroni tail probability `0.05 / (2 × 2 × 3)`). Continue proportionally if
either turn's simultaneous interval half-width exceeds two percentage points.
Confirmatory Holm tests are computed once, at the final stopped sample size.

## Post-collection, pre-aggregate analysis amendment

Amended on 2026-08-07 after all four 2,048-case primary blocks had completed
and passed their plan-bound and combined integrity audits, but before any
cross-arm primary effect, confidence interval, or stopping decision was
computed.  The eight raw result/plan files were sealed in
`results/v2-primary-2048-postcollection.sha256` before this amendment.
The predecessor preregistration bound into all four collection plans has
SHA-256
`e7fd4f6457c646ac4bfde1257aa6933297ac37fd76df577e0d084aa4a2b4963d`.

An independent code audit found that `analyze_replays.py` emitted ordinary
fixed-look cluster p-values and Holm adjustments whenever two warm bank cells
were present.  It had therefore already written such values into the cache
pilot analysis, contrary to the instruction above to compute confirmatory Holm
tests only once at the final stopped sample.  A fixed-look p-value followed by
an outcome-dependent precision stopping rule is not automatically
sequentially valid.  Those pilot p-values are a protocol deviation, are
exploratory only, and are not used for a primary claim or decision.

The primary analysis is amended as follows before the first aggregate is run:

- the analyzer must require an exact permitted cumulative look: all 12 frozen
  real prefixes per turn, the exact cumulative seed ranges, warm cache,
  repeat zero, and the frozen FP8/BF16/BF16/FP8 stage mapping;
- the continuation metric is the conventional simultaneous-interval
  half-width, `(upper - lower) / 2`, and continuation is required if either
  bank turn exceeds two percentage points;
- ordinary primary McNemar, cluster-Wald/score, and Holm p-values are
  suppressed rather than treated as confirmatory;
- confirmatory primary inference uses the already preregistered seed-cluster
  bootstrap interval whose tail budget is simultaneous across both turns and
  all three permitted looks.  Whether that interval excludes zero is the
  family-wise, sequentially protected significance statement.

This replaces the planned final Holm-test workflow; it does not alter the
estimand, frozen observations, pairing, weighting, bootstrap allocation,
permitted looks, or precision threshold.

This amendment specifically supersedes only
`seed-manifest.json`'s `continuation_rule.confirmatory_testing` value.  The
seed manifest and four frozen collection plans remain byte-unchanged for
provenance; their conventional half-width rule and every allocation remain in
force.
