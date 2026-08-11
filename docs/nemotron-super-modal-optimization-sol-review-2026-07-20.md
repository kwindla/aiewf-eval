# Adversarial review: Nemotron-Super Modal optimization

Verdict: keep the no-promotion decision, but revise the causal, statistical,
APC, and restart claims before treating the report as decision-grade. The raw
counts support avoiding APC+MTP and retaining BF16 021826 v2 as the incumbent;
they do not establish that an APC×MTP interaction caused the failure.

## Required corrections

1. **Describe nonadditivity, not proven causality.** The per-conversation counts
   of the six required tools are APC+MTP `[1,4,3,1,0,1]`, MTP-only
   `[6,5,6,3,6,6]`, APC-only `[6,6,6,5,6,5]`, and v2
   `[6,6,6,5,5,6]`. The descriptive factorial interaction is therefore
   -3.67 tools/conversation: APC had a 0.00 effect without MTP and a -3.67
   effect with MTP. That is strong evidence that APC alone was not sufficient
   to reproduce the collapse in this sample. It is not a causal interaction
   test: the cells were sequential, non-randomized batches, each came from one
   deployment, and the report presents only pairwise tests rather than an
   interaction contrast. Replace “rules out ... APC alone and instead
   implicates” with “did not reproduce under APC alone and is consistent with
   a configuration interaction.” The two combined-cell exact p-values are
   reproducible under exhaustive whole-conversation label enumeration, but the
   claimed MTP-only-v2 `p=0.697` is not: the stated run-level vectors give
   `812/924 = 0.8788` for the same two-sided mean-difference test. Correct it or
   name and preserve the different statistic. Treat all p-values as unadjusted
   screening summaries whose exchangeability assumption is not guaranteed by
   this run order.

2. **Make the conversation the unit for latency as well as quality.** The table
   pools 182--207 turn latencies per cell, which overweights conversations with
   recovery interactions and is not cluster-level analysis. The medians of the
   six conversation-level TTFAT medians are approximately 1521 ms (v2), 1317
   ms (MTP-only), 1326 ms (APC-only), and 1094 ms (APC+MTP), rather than
   1532/1322/1328/1098 ms. NVFP4 is 1296 ms over all six attempts or 1307 ms
   over the five completed survivors, rather than a pooled 1317 ms. Report the
   six run-level values (and an interval if making an inferential claim).
   Pooled strict-turn accuracy is arithmetically acceptable only where every
   judged conversation has the fixed 30-turn denominator. Label APC+MTP
   `122/150` as **judge-available** accuracy: its five judged runs include the
   incomplete attempt and omit a completed attempt whose judge output failed.
   Keep NVFP4's conditional survivor accuracy (`139/150`, n=5), all-attempt
   completion (`5/6`), and all-attempt required-tool opportunities (`30/36`)
   as three distinct endpoints. At n=6, one abort is the entire 5/6-versus-6/6
   difference; neither 5/6 nor 6/6 is a reliability estimate precise enough
   for equivalence or promotion. The runner should likewise record
   `run_rc`, strict completion, and judge status in separate manifest columns;
   currently `judge_failed` overwrites the fact that a transcript completed.

3. **Withdraw the “clean APC evidence” sentence.** The 12,638-token APC arm
   falls from 1106 ms to a 584 ms warm median, but the alleged no-APC NVFP4 arm
   falls from 1254 ms to 881 ms under the same fixed-order probe. Thus at least
   about 373 ms of a first/repeat delta can occur without APC; the roughly
   150 ms difference between those two deltas is itself confounded by snapshot,
   quantization, and one ordered sequence. The 25,238-token APC observation is
   additionally pre-warmed by the 12,638-token prefix. Lines 73--77 and
   108--113 of the report currently contradict each other. State that these
   probes show repeat/warmup behavior but do not identify APC benefit. A clean
   estimate needs a matched checkpoint and randomized/interleaved repeat order
   or independent cold instances. Also make APC-off explicit with
   `--no-enable-prefix-caching` in both MTP-only definitions and archive the
   resolved vLLM startup configuration; omission of the positive flag relies
   on model/version defaults and the supplied artifacts contain no server log
   proving the resolved state.

4. **The missing BF16 030326 arm blocks both quality and performance
   attribution.** The report is correct that the NVFP4 result is unmatched;
   extend that caveat to the apparent decode-throughput gain, not only quality.
   Call the future BF16 arm a *matched control*, not a paired experiment: the
   conversations are independent and use no paired seed. The current BF16 and
   NVFP4 source definitions differ only in checkpoint, app/cache names, and
   comments, but path names do not prove that the checkpoints share identical
   pre-quantization weights, tokenizer, and generation/config metadata. Verify
   immutable revisions or hashes before attributing a difference to
   quantization. The supplied data directory also has no BF16 startup log,
   deployed-definition snapshot, or app-list record, so the detailed 25-minute
   failure timeline and zero-task assertions are not independently auditable.

5. **Make the restart runbook executable and fail-safe.** The definition called
   “checked-in” is currently untracked. Preserve the exact reviewed revision
   before deployment. Use the full endpoint and `curl --fail-with-body`, create
   a new absolute output directory before the probe, and require an absent/empty
   manifest. This matters because `probe_modal_super_endpoint.py` does not
   create its output parent, while `run_modal_quality_cell.sh` creates relative
   paths before changing to `/home/khkramer/src/aiewf-eval` and then appends to
   `manifest.tsv`; the report's relative `<results>/...` form can fail or mix
   attempts. The runbook must spell out the four-turn command, use an absolute
   path such as
   `/home/khkramer/src/aiewf-eval/docs/nemotron-super-modal-optimization-data/bf16-030326-mtp-only`,
   and add cleanup on success, command failure, or interruption. Stop the exact
   app name/record and poll its exact app-list row until `State=stopped` and
   `Tasks=0`; do not rely only on the happy-path step after attempt 6.

## Production recommendation

The direction is appropriately conservative: do not promote APC+MTP,
APC-only, or NVFP4, and leave the incumbent v2 unchanged. Soften “small
measured-quality risk” for MTP-only to “an observed 3.3-point decrement with
wide uncertainty at n=6.” A staged MTP-only candidate should require prespecified
completion and required-tool gates, randomized short run blocks, and independent
deployment/batch replication before promotion. Describe NVFP4's 305--318 tok/s
as an absolute endpoint observation, not a quantization uplift, until the
matched BF16 030326 control starts and completes.
