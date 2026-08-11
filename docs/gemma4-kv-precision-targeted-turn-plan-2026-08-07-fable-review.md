# Adversarial review: Gemma 4 KV-precision targeted-turn plan (2026-08-07)

Reviewer: Claude Fable 5. Scope: `docs/gemma4-kv-precision-targeted-turn-plan-2026-08-07.md`,
checked against the frozen artifacts in `ops/local-gemma4-31b-nvfp4-sglang/`
(server scripts, campaign `configuration.json` files, pooled N=150 report), the
cached checkpoint config, and the pinned SGLang image
(`v0.5.15.post1`, digest `00c53fe4…`), whose source I inspected directly.

## Overall verdict

Post-review correction: a subsequent construction probe against Pipecat 1.3.0
showed that `InputParams.top_k` is explicitly ignored and was not present on
the historical HTTP wire. Accordingly, the implemented reproduction uses
temperature 1.0 and top-p 0.95 but omits top-k. This supersedes Finding 1's
recommendation to send top-k 64 while preserving the review as an audit of the
evidence available at the time.

The core design — frozen token-identical snapshots, paired seeds, mechanical
scoring, cache mode as an explicit factor, per-turn analysis — is sound and a
large improvement over the full-conversation comparison. Most of the plan's
SGLang/checkpoint claims are **correct and verified** (see below). But the plan
contains one outright factual error in the sampling specification, several
places where the experiment as written cannot deliver what it promises (warm
scheduling under the 16K pool; the Phase 0.7 determinism gate; the cache-gate
equivalence power), and one interpretive gap: a frozen-prefix replay may
correctly fail to reproduce the historical turn-12 gap if that gap is mediated
by earlier-turn drift, and the plan has no cheap test for that before spending
~8K GPU completions.

## Claims verified against source (no action needed)

- Checkpoint `config.json` has `kv_cache_scheme: null`; 10 full-attention +
  50 sliding-window layers, `sliding_window: 1024`. Confirmed.
- Pinned `model_runner.py`: with `--kv-cache-dtype fp8_e4m3` and no
  `--quantization-param-path`, it logs a warning and uses unit scales; **with**
  a param path it `raise`s `RuntimeError` for models lacking
  `load_kv_cache_scales`, and none of the `gemma4_*.py` model classes implement
  it. So the plan's "expected to fail" is right, and the failure is loud, not
  silent. Confirmed.
- OpenAI-compat protocol accepts `seed` (mapped to `sampling_seed`),
  `cache_salt` (single or list), and `input_ids` (which skips server-side
  template tokenization while messages still drive tool/stop behavior).
  The sampler has an explicit per-request, **position-keyed** seeded path.
  Confirmed. The position-keyed RNG is good news for the paired design: same
  seed in both arms yields common random numbers per decode position, so pairs
  diverge only where logits differ — a legitimate variance-reduction mechanism.
- `--enable-cache-report` is set in both server scripts, so per-request
  cached-token verification is available. Confirmed.
- "Cold replay is not a purer test of KV precision" — correct. With chunked
  prefill (2048) every chunk after the first reads earlier K/V from the pool at
  the configured dtype, so FP8 round-tripping is exercised warm or cold.

## Finding 1 (blocking): the plan's sampling parameters are wrong

Phase 0.6 specifies "temperature 0.6, top-p 0.95". Every frozen Gemma 4
campaign — local FP8 N=30 and N=120, local BF16 N=30 and N=120, and both
BaseTen cohorts — ran with `MTE_VLLM_TEMPERATURE=1.0`, `MTE_VLLM_TOP_P=0.95`,
`MTE_VLLM_TOP_K=64`, `MTE_VLLM_MAX_TOKENS=8192` (see each
`configuration.json`, `common_environment` and provenance needles
`T=1.0`, `top_k=64`). Running the replay at 0.6 without top-k both changes the
behavioral distribution being studied and disconnects the campaign from the
historical evidence it is trying to explain. Turn 12 sits near a decision
boundary (~47–65% error in every arm), exactly where temperature changes move
outcomes most. The plan must specify T=1.0, top-p 0.95, top-k 64 and assert
them in provenance, as the campaign template already does.

## Finding 2: causal identification and the geometry confound

The historical turn-level evidence confounds KV dtype with pool geometry: the
FP8 campaign ran `--swa-full-tokens-ratio 1.0` with no `--max-total-tokens`;
the BF16 campaign ran `--max-total-tokens 16000 --swa-full-tokens-ratio 0.35`.
The plan correctly equalizes geometry going forward (Phase 0.3), but this cuts
both ways and the plan doesn't confront it:

- The new "FP8 reproduction" arm is *not* the configuration that produced the
  observed deficit. If the deficit was partly a geometry/pool-layout effect
  (page layout, SWA pool sizing, eviction behavior), the matched-geometry
  contrast will correctly shrink toward null and the study will be read as
  "FP8 exonerated" when the historical arm was the thing that mattered.
  A small FP8 cell in the **original** geometry (same seeds, a few hundred
  pairs) is cheap and links the new experiment to the old evidence.
- The FP8 + compact-geometry configuration has never run the full benchmark
  ("the compact geometry has already passed the full benchmark" was true only
  of BF16). Add a smoke gate for FP8-compact.

Also note both servers set `--mem-fraction-static 0.90`; they cannot co-run on
one 32 GB RTX 5090, so A/B/B/A means full server restarts and re-priming per
block. That's workable but should be stated, because it interacts with the
warm-cache scheduling problem below.

## Finding 3: warm-prefix operation is not compatible with naive scheduling

`--max-total-tokens 16000` means the full-attention pool holds roughly **one**
~14K-token prefix. The design has 13 prefixes (golden + 12-prefix bank) × 2
turns. Any interleaving across prefixes evicts the radix entry and silently
turns "warm" requests cold. The plan records cached-token counts, which would
detect this after the fact, but the schedule must be designed for it up front:
group all requests for one prefix contiguously, prime after every prefix
switch and server restart, and make per-request `cached_tokens ≥ expected`
a hard assertion rather than a logged observation. One helpful fact: if both
turn snapshots for a prefix are built from the same golden history, the
turn-12 prompt is a strict token prefix of the turn-15 prompt, so ordering
turn-15 priming first covers both. Conversely, the A/B/B/A block structure and
the half-golden/half-bank allocation need to be reconciled with this grouping
in the preregistered schedule, not improvised.

## Finding 4: the Phase 0.7 repeatability gate conflates two different things

Phase 0.7 requires identical parsed output and token IDs for the same
`(snapshot, seed)` "after a cache hit and after a server restart". A restart
empties the radix cache, so this is a warm-versus-cold comparison — and warm
vs cold legitimately changes the floating-point path: with a prefix hit,
prefill covers only the suffix, chunk boundaries shift, and split-KV reduction
order differs, so last-position logits can differ in low-order bits and a
seeded sampler near a probability threshold can flip tokens. That is expected
numerics, not nondeterminism, and it is precisely what H4 is supposed to
measure. As written, the gate will likely fail for benign reasons and stall
the campaign ("investigate before launching"). Redefine:

- **Gate (validity):** identical outputs across repeats *within the same cache
  state* — warm/warm repeats and cold/cold repeats, including across restarts
  for cold. Failure here is real nondeterminism and does block the campaign.
- **H4 (hypothesis):** warm-vs-cold same-seed disagreement is an experimental
  outcome with its own preregistered analysis, never a launch gate.

Relatedly, the plan's interpretation of warm/cold mismatches as
"cache-manager, prefix-match, page-boundary, eviction, or stale/corrupt reuse
effects" over-claims: chunk-boundary numerics produce mismatches with a fully
correct cache. The cache-bug interpretation needs corroboration (e.g., token
divergence *before* the suffix, or hash mismatches in cached spans), not just
any output difference.

## Finding 5: estimand, pseudoreplication, and the dominant golden prefix

Half the budget goes to one constructed golden prefix. 1,024 seeds on one
prompt estimate P(success | this exact snapshot) very precisely — but that is
a *snapshot-specific* estimand, and no amount of seeds makes it generalize to
"turn 12". The stated H1/H2 ("on turn 12, success differs between arms") reads
as a population claim; the design supports it only as a mechanism probe plus a
12-prefix robustness check (~85 seeds/prefix/turn). Consequences:

- A "paired bootstrap over `(prefix, seed)` blocks" that resamples prefix
  blocks is unstable when one block holds 50% of the data. Resample seeds
  *within* prefix strata (inference conditional on the prefix set), report the
  golden prefix and the bank separately, and never quote a pooled CI in which
  the golden prefix dominates as if it were the turn-level effect.
- The golden prefix is an idealized history (golden assistant text + exact
  tool results) that the model may never produce for itself; effects measured
  on it may not transfer to self-generated histories. The bank partially
  covers this — keep prefix as a blocking factor, as planned, and preregister
  that a golden-only effect with no bank effect is interpreted as
  prefix-specific.
- Note the bank's BaseTen-sourced prefixes were generated by different weights
  (BF16); source class is properly a blocking factor, as the plan says.

## Finding 6: winner's curse and what the historical numbers actually support

Turns 12 and 15 were selected *from the same data* that produced the effect
sizes. The +20.0-point turn-12 extension effect is therefore an upper-biased
estimate; power and the continuation rule should assume smaller true effects.
The internal evidence already shows fragility: pooled N=150 turn-15 shows a
9-error gap while the N=120 extension shows only ~+2.5 points — implying the
N=30 cohorts alone contributed a ~20-point turn-15 swing that did not
replicate. Preregistering H1/H2 on new data handles validity; the review point
is calibration — do not treat ±2-point precision as generous relative to a
"20-point" effect that is partly selection artifact.

Also specify H3's scale (percentage-point difference vs. log-odds): with both
turns near 50% error the choice matters less than usual, but interaction
claims are scale-dependent and this should be pinned before data collection.

## Finding 7: sequential looks and the cache-gate's equivalence power

- The 2,048 → 4,096 → 8,192 continuation rule is precision-targeted (good — it
  does not condition on effect size), but the plan should state explicitly
  that confirmatory Holm-adjusted tests are computed **once, at the final N**,
  with interim looks used only for CI width. Testing at each look with no
  alpha spending inflates the primary family.
- The cache gate is 512 paired seeds/cell, but the preregistered warm/cold
  equivalence margin is ±3 points on a difference-in-differences across four
  cells. Its CI width is governed by the observed warm/cold discordance rate
  q: if same-seed warm/cold outputs disagree rarely (q ≈ 1–5%), 512 is
  adequate; if the chunk-boundary effects in Finding 4 make q large (10–20%+),
  the DiD interval blows past ±3 and the gate can neither establish
  equivalence nor its absence. Preregister an adaptive rule: measure q in the
  pilot, then size the cold cells to reach the margin, or widen the margin
  with justification.

## Finding 8: the FP8-scaling arm is over-weighted a priori; calibration leakage

Unlike INT8, FP8 E4M3 is a floating-point format: relative precision is
constant across its normal exponent range, so a global scale of 1.0 costs
accuracy only where values **saturate** (|x| > 448) or fall into the subnormal
region. Per-layer calibrated scales are the remedy for range mismatch, not a
general accuracy upgrade. Attention-sink K outliers make saturation plausible
for some heads, so the question is empirical — which means the distribution
probe (mechanistic follow-up #1: range/saturation/underflow by layer) should
be a cheap **precondition** for building the calibrated arm and the loader
patch, not a parallel effort. If measured saturation is negligible, H5 is
near-dead on arrival and the patch effort should be dropped.

Separately: "derive held-out calibration scales" from distributions captured
"on the two frozen prefixes" calibrates on the evaluation inputs. It's a mild
leak (scales are 60–120 scalars), but for a study this careful, derive scales
from independent text (e.g., other benchmark conversations) and only evaluate
on the frozen prefixes.

## Finding 9: scorer I/O parity with production

The scorer validation plan (900 historical judgments, ≥99.5% agreement,
adjudicate all disagreements, preserve raw outputs) is good. Two gaps:

- Production ran with `MTE_VLLM_NORMALIZE_TOOL_CALL_INDICES=1` (SGLang's
  Gemma 4 parser emits schema-position indices in streamed deltas) and
  `MTE_DEDUPE_TOOL_CALLS=1`. The replay harness must either consume
  non-streamed responses (sidestepping index reassembly, but then TTFAT
  streaming metrics need a separate decision) or replicate the normalization;
  and the taxonomy needs a category/policy for duplicate and multiple tool
  calls, which the pipeline previously deduped before judging.
- The turn-15 category "no tool, redundant confirmation" must be checked
  against how the historical judge scored conversations that *recovered* on
  the recovery turn (`MTE_ENABLE_RECOVERY=1`), so replay single-shot scoring
  and historical labels mean the same thing during validation.

## Finding 10: the missing cheap experiments (do these first)

Three near-zero-cost analyses would materially de-risk or re-scope the 8K-run
campaign, and the plan defers or omits them:

1. **Teacher-forced margin probe (promote follow-up #3 to Phase 0).** A
   handful of deterministic requests per arm capturing top logprobs at the
   first tool-vs-text decision token gives a continuous, low-variance measure
   of the KV effect on the exact decision the campaign will sample thousands
   of times. If the logit margin barely moves between dtypes on both
   snapshots, a large sampled effect is unlikely and the campaign can shrink;
   if it moves a lot, you have a mechanism readout before spending GPU-days.
   Add greedy (temperature-0) replays of each snapshot per arm as the modal
   decision at essentially zero cost.
2. **Mediation analysis on the existing 300 local transcripts (zero GPU).**
   The scripted user turns are fixed, so the only cross-arm difference
   entering turn 12 in the historical data is each arm's own earlier
   assistant text. Quantify whether turn-11/turn-9–10 phrasing differs
   systematically between FP8 and BF16 transcripts and whether turn-12
   failure is predicted by identifiable prefix features. This directly
   estimates how much of the historical gap is mediated by earlier-turn
   drift — the component the frozen-prefix design deliberately removes. The
   plan should preregister the interpretation either way; without this, a
   null replay result is ambiguous between "no KV effect" and "effect exists
   but is mediated upstream".
3. **Retrieval-distance sanity check on the framing.** The facts turn 12
   needs (name at turn 10, an identical successful tool call at turn 11) sit
   a few hundred tokens back; the 14K context is dominated by the system
   prompt and knowledge base. With a 1,024-token sliding window, the relevant
   turns are likely outside the SWA window but trivially within the 10
   full-attention layers' reach. The "long-range state retrieval" framing is
   an assumption, not a fact; follow-ups #5/#6 test it, but the write-up
   should not presuppose it.

## Minor points

- `/flush_cache` (referenced via the bench-serving doc) may be a simpler,
  directly verifiable cold control than unique `cache_salt` values; the plan
  already conditions salt use on verified cache counts, which is the right
  discipline either way.
- Token-ID hash equality across arms (gate 1) is near-tautological with one
  image and tokenizer, but harmless as a tripwire.
- The warm design reuses one physical stored-KV realization of each prefix
  for all seeds; since FP8 rounding is deterministic this is fine, and the
  cold cell is the correct check that it isn't a corrupt/unlucky copy.
- Guard the eventual write-up: even a confirmed snapshot-level FP8 deficit on
  turn 12 does not overturn the global result (BF16−FP8 = +0.67 pts, CI
  −0.07 to +1.40). Targeted-turn effects and deployment-quality claims are
  different estimands.
- 14K prompt + priming + up to 8,192 completion tokens against a 16,000-token
  pool leaves little headroom; cap `max_tokens` for replay (responses are
  short) or verify the longest bank prefix + max decode fits.

## Required changes (blocking, in priority order)

1. **Fix the sampling spec**: temperature 1.0, top-p 0.95, top-k 64 (and the
   8192 max-token convention), matching every frozen campaign; assert via
   provenance needles. (Finding 1)
2. **Split the Phase 0.7 gate**: repeatability within identical cache state is
   the launch gate; warm-vs-cold agreement is H4, never a gate. (Finding 4)
3. **Design the warm schedule around the 16K pool**: contiguous per-prefix
   blocks, priming after every switch/restart, hard per-request
   `cached_tokens` assertions; reconcile with A/B/B/A and the golden/bank
   split in the preregistered schedule. (Finding 3)
4. **Fix the inference plan for the dominant golden prefix**: stratified
   (within-prefix) seed resampling, golden and bank reported separately,
   H1/H2 restated as snapshot/mechanism estimands rather than turn-population
   claims. (Finding 5)
5. **Run the cheap probes before the campaign**: teacher-forced margins +
   greedy replays, and the zero-GPU mediation analysis of existing
   transcripts, with preregistered interpretation of a mediated-upstream
   outcome. (Finding 10)
6. **Link new arms to the historical evidence**: a small FP8 cell in the
   original (ratio-1.0, uncapped) geometry, and a full-benchmark smoke of the
   never-run FP8-compact configuration. (Finding 2)
7. **One confirmatory analysis at final N** (or explicit alpha spending) for
   the primary family across the 2,048/4,096/8,192 looks. (Finding 7)
8. **Size the cold cells adaptively** from pilot warm/cold discordance, or
   widen the ±3-point equivalence margin with justification. (Finding 7)
9. **Scorer I/O parity**: decide streamed vs non-streamed consumption, handle
   tool-call index normalization and duplicate/multiple calls, and align
   turn-15 labels with recovery-turn judge semantics before validation.
   (Finding 9)

## Optional improvements

- Gate the calibrated-scale arm on measured saturation/underflow, and derive
  scales from held-out text rather than the evaluation prefixes. (Finding 8)
- Prefer `/flush_cache` (if verified in the pinned image) over per-request
  unique salts for the cold cells.
- Specify H3's interaction scale (pp vs. log-odds) in the preregistration.
- State the one-GPU serial-restart constraint and its interaction with block
  structure explicitly in the plan.
- Temper effect-size expectations for winner's curse (turn-15's N=30 vs N=120
  inconsistency is a concrete internal warning). (Finding 6)
- Add the deployment-vs-snapshot estimand guardrail language to the report
  template. (Minor points)
- Cap replay `max_tokens` well below the pool headroom.
