# Gemma 4 KV-precision targeted-turn experiment

Date: 2026-08-07

Adversarial review: [Fable review and required changes](gemma4-kv-precision-targeted-turn-plan-2026-08-07-fable-review.md).
The blocking corrections from that review are incorporated below.

## Decision summary

Build a direct, serial OpenAI-compatible replay harness for scripted turns 12
and 15. It should send frozen, token-identical conversation snapshots to the
same local NVFP4 checkpoint under two otherwise matched SGLang configurations:
BF16 KV and FP8 E4M3 KV. Use the same sampling seeds in both arms, score the
tool decision mechanically, and treat each `(snapshot, seed)` as a paired
experimental unit.

The primary campaign should use SGLang's normal RadixAttention prefix reuse,
because that is both the production configuration and the efficient way to run
thousands of 14K-token prompts. A smaller, same-seed cold-prefix control should
make cache reuse an explicit experimental factor. Replaying every request cold
is not a purer test of KV precision: even without cross-request prefix reuse,
autoregressive inference still writes and reads every earlier token's K and V
at the configured cache dtype.

There is also a potentially important diagnostic arm. The existing FP8 server uses
unit KV scales because the checkpoint declares `kv_cache_scheme: null`, the
launch command supplies no `--quantization-param-path`, and pinned SGLang
v0.5.15.post1 explicitly falls back to K/V scales of 1.0. SGLang warns that
this can reduce accuracy when values saturate or fall into the subnormal range.
Before interpreting the result as an inherent FP8 limitation, measure those
failure modes. Only if they are material should we implement support for
calibrated per-layer K/V scales and compare:

1. NVFP4 weights + BF16 KV.
2. NVFP4 weights + FP8 E4M3 KV with unit scales, reproducing the current arm.
3. NVFP4 weights + FP8 E4M3 KV with calibrated scales, if a validated SGLang
   path can be implemented.

The BaseTen BF16-weights/BF16-KV/MTP deployment can replay the same corpus as
an external reference, but it is not part of the local KV causal contrast.
Weights, hardware, MTP, and SGLang version all differ.

## Why cache reuse is not the same as KV precision

At each transformer layer, the current hidden state is projected into Q, K,
and V. Q is used for the current attention operation. K and V are written to a
GPU token-to-KV pool so subsequent tokens can attend to them without rerunning
the entire prefix. With BF16 KV, those tensors remain BF16. With FP8 KV, K and
V are quantized when stored and scaled/dequantized, often inside the attention
kernel, when read. The generated token's own K/V is then appended, so the same
mechanism continues during decode.

Gemma 4 is a hybrid-attention model: 10 layers use full attention and 50 use
sliding-window attention. SGLang therefore maintains separate effective
capacity constraints for the full-attention and SWA pools. The long-range
state needed by these turns is available to full-attention layers, while the
SWA layers retain only their local window.

RadixAttention adds reuse *between requests*. It indexes tokenized prefixes in
a radix tree and retains completed requests' KV slots as evictable cache
entries. A later byte/token-identical prefix can reuse those stored K/V values
instead of recomputing them. `--disable-radix-cache` disables that
cross-request reuse; it does not eliminate the within-request KV cache.

Consequently:

- A warm replay uses one previously computed copy of the prefix K/V many times.
- A cold replay recomputes the same prefix K/V for each request and then stores
  it in the selected BF16 or FP8 representation.
- Both paths still measure FP8-versus-BF16 KV. Warm/cold differences can come
  from ordinary chunk-boundary and reduction-order numerics as well as from
  cache-manager, prefix-match, page-boundary, eviction, or corruption effects;
  output disagreement alone does not establish a cache bug.
- SGLang already provides the requested “create a KV cache and replay it”
  behavior automatically. We do not need to serialize an external cache or use
  a session handle for the main experiment.

The pinned server also supports `cache_salt` as a cache-key-only field. A fixed
salt produces warm hits; a unique salt per request should force cold prefixes
without changing model-visible tokens. This is a useful control if, and only
if, returned cache-token counts verify the intended hit/miss state.

References:

- [SGLang server arguments](https://docs.sglang.ai/advanced_features/server_arguments.html)
- [SGLang serving benchmark and cache-flush controls](https://docs.sglang.ai/developer_guide/bench_serving.html)
- [RadixAttention design](https://lmsys.org/blog/2024-01-17-sglang/)

## Existing evidence to preserve

The targeted turns must be analyzed separately. In the independent N=120
extension, the local BF16-minus-FP8 difference was approximately +20.0 points
on turn 12 but only +2.5 points on turn 15. BaseTen showed the opposite shape:
it was weaker on turn 12 and much stronger on turn 15. Pooling the two turns
hides this interaction and could incorrectly suggest a generic BF16 effect.
Because these turns and effect sizes were selected after looking at the
historical data, treat their magnitudes as discovery estimates subject to
winner's curse. The new replay cohort is the test; it must not be powered on
the assumption that the +20-point turn-12 estimate will repeat.

Both turns have deterministic behavioral endpoints:

- Turn 12 must call `submit_session_suggestion` with Jennifer Smith and the
  state-machine/complex-workflows suggestion. Typical failures claim the
  suggestion was submitted without calling the tool.
- Turn 15 must call `submit_dietary_request` with Jennifer Smith and `vegan`.
  Typical failures ask for redundant confirmation and require the benchmark's
  recovery turn.

No LLM judge is required for the primary outcome.

## Hypotheses and estimands

Preregister these before collection:

- **H1, primary:** Conditional on the frozen real-prefix bank for turn 12,
  exact tool-decision success differs between local BF16 KV and current
  unit-scaled FP8 E4M3 KV.
- **H2, primary:** Conditional on the frozen real-prefix bank for turn 15,
  exact tool-decision success differs between the same two arms.
- **H3, mechanism:** The KV percentage-point effect differs by turn
  (`KV × turn` interaction on the risk-difference scale).
- **H4, cache control:** Warm-prefix reuse does not materially change the
  BF16-minus-FP8 effect relative to cold-prefix replay. Predefine a practical
  equivalence margin rather than treating “not significant” as equivalence.
- **H5, scale mechanism:** If calibrated FP8 is feasible, calibrated FP8
  reduces or eliminates any deficit of unit-scaled FP8.

The primary estimand for each turn is the equal-prefix-weighted paired
percentage-point difference in exact tool-decision success over the frozen
real-prefix bank. The single constructed golden prefix is a separately
reported mechanism probe, not a population sample and not part of the primary
bank average. Report error categories and output-token distributions as
secondary endpoints.

## Phase 0: remove avoidable confounds

1. Create one server template shared by the two matched local arms. Hold every
   launch option constant except `--kv-cache-dtype`.
2. Explicitly use `bfloat16` versus `fp8_e4m3`; do not rely on an implicit
   default for the BF16 arm.
3. Give the matched arms identical pool geometry: `--max-total-tokens 16000` and
   `--swa-full-tokens-ratio 0.35`. The earlier FP8 and BF16 campaigns used
   different pool sizes because FP8 had extra capacity. The compact geometry
   has passed the full benchmark only with BF16 KV, so require an FP8-compact
   full-conversation smoke before using it. Also run a small, same-seed FP8
   bridge cell with the historical uncapped, ratio-1.0 geometry; otherwise a
   null matched-geometry result would not explain the historical comparison.
4. Record the actual prefill and decode attention backends from server logs.
   Force the same backend if both dtypes support it. If FP8 necessarily selects
   a different kernel, define the result as a deployment-level FP8-KV effect
   and use the tensor probe below to isolate numerical storage loss.
5. Confirm and record the FP8 scale behavior. The pinned image's
   `model_runner.py` warns that absent scales default to 1.0, while the frozen
   Red Hat checkpoint contains no KV-cache scheme. Pinned Gemma 4 currently
   exposes no `load_kv_cache_scales` method, so merely adding a scale-file
   argument is expected to fail. Do not silently upgrade SGLang inside the
   primary reproduction. First measure pre-quantization range, saturation, and
   underflow. Do not build a scale-loader arm unless that gate finds a plausible
   range problem; derive any scales from independent calibration text, not the
   evaluation snapshots.
6. Keep batch size and concurrency at one, CUDA graphs disabled, no MTP,
   **temperature 1.0, top-p 0.95, no top-k field**, and the original tool
   schema/order. Although campaign configuration and logs said top-k 64, a
   post-review wire audit showed Pipecat 1.3.0 documented `InputParams.top_k`
   as ignored and never copied it into the HTTP request. Omitting top-k
   reproduces what the model actually received; adding 64 would be a new
   sampler experiment. The historical
   `max_tokens` convention is 8,192. Because every historical target response
   was at most 156 tokens and a 14K prompt leaves little room in a 16K pool,
   the replay may use a preregistered 512-token cap only after a 128-seed parity
   probe against 8,192 shows identical outcomes and no truncation; otherwise
   retain 8,192 and enlarge both matched pools equally.
7. Test seeded repeatability under the production flags. The pinned OpenAI
   schema accepts `seed`, and the sampler has a position-keyed per-request
   seeded path. For at least 100 `(snapshot, seed)` cases per arm, require
   identical parsed output and token IDs across warm/warm repeats and across
   cold/cold repeats, including cold repeats separated by a restart. Failure
   within the same cache state blocks the campaign. Warm-versus-cold
   disagreement is an H4 outcome, not a repeatability gate, because the two
   paths can legitimately differ in low-order floating-point numerics.

## Phase 0A: cheap go/no-go analyses

Do these before spending thousands of GPU requests:

1. Run greedy replays and a teacher-forced margin probe on the candidate
   snapshots under both KV dtypes. Record top log probabilities at the first
   tool-versus-text decision token and the likelihood of the canonical tool-call
   sequence. A large, repeatable margin shift supports the sampling campaign; a
   negligible shift argues for a smaller behavioral run and more emphasis on
   upstream mediation.
2. Analyze the existing 300 local full-conversation transcripts for mediation
   by earlier assistant text. Quantify arm differences in turns 9–11 and 13–14,
   and test whether prefix features predict the later failure. A null frozen-
   prefix replay then means “no direct effect at this frozen state”; it does not
   rule out an FP8 effect mediated through earlier generated history.
3. Treat “long-range retrieval” as a hypothesis, not an assumption. Jennifer's
   name and the preceding tool state are only a few hundred conversation tokens
   behind the target even though the fixed system/KB prefix makes the total
   prompt roughly 14K tokens.

## Phase 1: build frozen request snapshots

Write a direct HTTP harness rather than starting Pipecat once per repetition.
Use one persistent connection and one outstanding request at a time.

1. Instrument one ordinary benchmark run, or the request builder itself, to
   capture the exact OpenAI request immediately before turns 12 and 15:
   messages, tool calls/results, tool schema, sampling fields, and chat-template
   options.
2. Replace generated IDs with fixed valid IDs and canonicalize JSON key order.
   Freeze each request as JSON and record its SHA-256.
3. Ask SGLang to return prompt token IDs. Assert that the token-ID SHA-256 is
   identical across both local arms. Prefer passing the frozen `input_ids`
   back through SGLang's supported request extension so tokenizer or template
   behavior cannot drift; retain messages so stop-token and tool-call behavior
   are unchanged.
4. Construct one mechanism snapshot from benchmark-golden assistant messages
   and exact successful tool results. This isolates the target decision from
   stochastic differences in earlier model replies, but it is an idealized,
   snapshot-specific probe and is reported separately.
5. Construct the primary bank from real, valid prefixes selected without
   reference to the target-turn outcome—for example four each from historical
   BaseTen, local FP8, and local BF16 cohorts. Validate that every prefix
   contains the required name and preceding tool state. Treat prefix as a
   blocking factor and weight the 12 frozen prefixes equally; never attribute a
   source-prefix effect to KV precision.
6. Freeze the seed list before either arm runs. Every `(prefix, turn, seed)` is
   sent to both KV arms.

Freezing history is essential. The full-conversation campaigns allowed earlier
assistant wording to differ across arms, so their target-turn inputs were not
byte-identical. That measures end-to-end conversational behavior but is not a
clean KV ablation.

## Phase 2: validate the mechanical scorer

Implement a scorer that parses the raw tool call and emits:

- correct tool and arguments;
- correct tool, wrong or missing argument;
- wrong tool;
- no tool, false claim of completion;
- no tool, redundant confirmation/question;
- malformed tool stream/parser failure;
- request/server failure.
- duplicate or multiple tool calls, with an explicit rule for whether the
  historical pipeline's deduplication would have accepted them.

Normalize only harmless case, whitespace, and article differences in the
suggestion text. Validate the scorer against all 900 historical target-turn
judgments (two turns across 450 conversations). Require at least 99.5%
agreement and manually resolve every
disagreement before freezing the scorer. Preserve raw responses so alternative
scoring remains possible.

Choose streaming or non-streaming before validation. If streaming, reproduce
the benchmark's response-local tool-index normalization and call deduplication.
If non-streaming, separately define how TTFAT will be measured and verify that
parsed outcomes match streaming on a sentinel set. For turn 15, compare the
single-shot label with the historical judge's recovery-aware label so a
redundant confirmation is consistently a target-turn failure even if recovery
later succeeded.

## Phase 3: cache experiment and main campaign

### Cache gate

Begin with 512 paired seeds per turn and KV arm, comparing:

- **warm:** one fixed `cache_salt`, with the prefix primed before measurement;
- **cold:** a unique `cache_salt` per request, or a separate
  `--disable-radix-cache` server if salt behavior does not produce verified
  zero-hit requests.

Record prompt length, cached-token count, request hash, seed, output-token IDs,
tool parse, TTFAT, and total latency. For the same seed, compare warm and cold
outputs exactly. This mismatch rate is an experimental result, not proof of a
cache bug. Estimate the paired warm/cold discordance rate in the pilot, then
increase the cold cells until the difference-in-differences interval can test
the preregistered ±3-point equivalence margin, or explicitly widen that margin
before the main campaign.

### Main warm-prefix campaign

Start with 2,048 paired seeds per turn per KV arm: 8,192 direct completions.
Allocate 512 seed pairs to the golden mechanism snapshot and 1,536 evenly over
the 12-prefix real bank (128 per prefix). Report the golden and bank estimates
separately; the primary bank estimate gives every real prefix equal weight.
When continuing, double every prefix stratum rather than adding observations to
whichever prefix appears interesting.

The 16K pool holds roughly one 14K full-attention prefix, so warm requests must
be scheduled in contiguous per-prefix blocks. After every prefix switch or
server restart, prime that prefix and require each measured request's
`cached_tokens` to meet a frozen per-snapshot minimum. A cache miss is a failed
warm-cell artifact, not a usable observation. Run the single-GPU campaign in
preregistered FP8/BF16/BF16/FP8 macro-blocks, with half of every prefix's seed
range assigned to each appearance; each macro-block groups its requests by
prefix and re-primes at every switch. Both local servers reserve 90% of the
same RTX 5090 and therefore run serially, never concurrently.

At the observed roughly one-second target-turn completion latency, 8,192
requests should take approximately 2.5–3 hours of pure inference. A 4,096-seed
campaign per cell would take roughly five to six hours. These are estimates;
the pilot should replace them with measured throughput.

Use a precision-based continuation rule that does not depend on whether the
effect is significant:

1. Analyze at 2,048 paired seeds per turn.
2. Continue to 4,096 and then 8,192 if the simultaneous paired confidence
   interval for either primary turn is wider than ±2.0 percentage points. The
   interval uses a conservative Bonferroni allocation across both turns and all
   three permitted looks; exact incremental seed ranges and ABBA blocks are
   frozen before the first cross-arm pilot contrast.
3. Stop at 8,192 and report achieved precision even if the width target is not
   reached.

Interim analyses are used only to decide whether the preregistered CI-width
target has been reached. Run the Holm-adjusted confirmatory tests once, at the
final sample size; do not repeatedly test significance at each look.

Independent-arm worst-case calculations need about 4,800 observations per arm
and turn for a ±2-point ordinary 95% interval on a difference. Pairing the
same seeds should usually be more efficient, but the plan must use observed
discordance—not assumed correlation—to decide whether 2,048 is sufficient.

## Statistical analysis

- Use paired outcomes because the same snapshot and seed appear in both arms.
- Report each turn separately. Because the same numeric seed drives overlapping
  position-keyed random streams in all 12 prefixes, compute one equal-prefix
  mean paired difference per seed and jointly resample those seed clusters.
  Report independent within-prefix resampling only as a sensitivity analysis.
  Use a seed-cluster-robust score test for the primary weak-null inference;
  pooled exact McNemar is descriptive. Do not resample the single golden prefix
  as if it represented a population of histories.
- Control the two primary turn tests with Holm's procedure or simultaneous
  bootstrap intervals. Do not add the optional scale/cache hypotheses to that
  primary family; label them secondary and report their own families.
- Fit a secondary mixed-effects logistic model with KV dtype, turn, cache mode,
  source-prefix class, and interactions; include a random intercept for prefix.
- For cache equivalence, report the difference-in-differences
  `(BF16−FP8)_warm − (BF16−FP8)_cold` against a preregistered margin, initially
  ±3 points. A nonsignificant interaction alone is not evidence of equivalence.
- Report HTTP/parser failures separately and also count them as failures in an
  intention-to-test analysis.
- Do not pool turn 12 and turn 15 as the primary result. Their historical
  effects are qualitatively different.
- A confirmed targeted-snapshot effect does not replace the global benchmark
  estimate. It is a different estimand from overall deployment quality.

## Mechanistic follow-ups

The behavioral replay answers whether an FP8 deployment changes decisions. It
does not by itself explain why. Add these probes in order of value:

1. **KV scale viability.** Capture pre-quantization K/V distributions by layer
   on independent calibration text. Measure range, E4M3 saturation and
   subnormal occupancy, MSE, and cosine error with unit scale. Only if range
   failure is material should we derive held-out scales and implement the
   smallest isolated Gemma 4 loader patch or test a separately pinned SGLang
   upgrade; never mix that change into the reproduction arm.
2. **Layerwise divergence.** For a handful of deterministic requests, compare
   BF16 attention outputs with quantize/dequantize FP8 outputs after each layer.
   Separate the 10 full-attention layers from the 50 SWA layers and identify
   where final-token hidden-state or logit divergence grows.
3. **Context-distance curve.** Create controlled versions of each task at
   approximately 2K, 4K, 8K, 12K, and 14K tokens. Hold the final request and
   relevant facts fixed while varying irrelevant prefix material. This tests
   whether FP8 loss grows with attention distance or simply with total context.
4. **State-explicit ablation.** Restate the missing information in the final
   turn: “Submit another suggestion for Jennifer Smith…” and “Yes, submit a
   vegan dietary request for Jennifer Smith.” If the KV gap disappears, the
   failure is memory/state retrieval rather than general tool syntax.
5. **Relevant-fact placement.** Move Jennifer's name and the authorization
   turn nearer to or farther from the final query while keeping length stable.
   This distinguishes distance from load.
6. **Format comparison.** If supported by a common attention backend, compare
   E4M3, E5M2, and BF16 KV. E4M3-versus-E5M2 behavior can indicate whether
   mantissa precision or dynamic range/clipping dominates.
7. **Weight/KV factorial replication.** On hardware large enough for BF16
   weights, run BF16 weights × {BF16, FP8 KV}. Alongside the local NVFP4
   weights × {BF16, FP8 KV}, this separates weight quantization, KV
   quantization, and their interaction. MTP must be off for this factorial.

## Implementation and audit artifacts

Create a new, self-contained directory under
`ops/local-gemma4-31b-nvfp4-sglang/targeted-turn-kv-study-20260807/` containing:

- frozen canonical request JSON and token-ID hashes;
- one shared server template plus explicit per-arm overrides;
- a serial replay client with resume support and atomic JSONL writes;
- a frozen seed manifest and per-prefix FP8/BF16/BF16/FP8 macro-block schedule;
- the validated deterministic scorer and its historical agreement report;
- raw outputs, server metadata/logs, cached-token observations, and failure
  records;
- preregistration, analysis script, machine-readable aggregates, and report;
- a teardown/audit record proving the container and GPU workload stopped.

Before the full run, require these gates:

1. Identical prompt-token hashes across arms.
2. Correct cache-hit/miss behavior in the warm/cold cells.
3. Seeded repeatability established within each cache state; warm/cold
   disagreement retained as an outcome rather than a gate.
4. Same model/checkpoint, sampling, tool schema, attention backend where
   possible, pool geometry, and concurrency.
5. Mechanical scorer validated against historical judgments.
6. FP8-compact completes one full benchmark conversation, and a 32-seed target
   smoke produces no HTTP, parser, or artifact-integrity failures.
7. The cheap greedy/margin and historical mediation probes are complete and
   their implications are recorded before the seed manifest is unblinded.

This design turns the current suggestive full-conversation result into a
focused causal test while retaining a clean route to diagnose whether the
effect comes from FP8 storage itself, unit scaling, long-context retrieval,
the attention kernel, or prefix-cache reuse.
