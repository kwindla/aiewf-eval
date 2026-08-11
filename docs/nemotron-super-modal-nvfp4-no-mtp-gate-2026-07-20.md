# Nemotron-Super NVFP4 no-MTP gate — 2026-07-20

Status: prospectively locked before deployment or outcome collection.

## Question

Did native one-token MTP contribute to the incomplete conversation and tool
misses in the NVFP4 030326 cell, or do those failures persist when MTP is
disabled?

## Frozen cell

- Checkpoint: NVFP4 030326.
- Runtime: vLLM 0.25.1 on 2x Modal B200, TP2.
- MTP: disabled. This is the only intended inference change from the prior
  NVFP4 030326 MTP-only cell.
- APC: explicitly disabled with `--no-enable-prefix-caching`.
- Other controls: FP8 KV cache, Mamba SSM cache float16, chunked prefill,
  FlashInfer autotune disabled, Super-v3 reasoning parser, Qwen3-Coder tool
  parser, and the same Nano chat template.
- Workload: `aiwf_medium_context`, native thinking budget 128.
- Exactly six conversations, with no replacement attempts.
- A strict completion requires an actual `end_session` call on scripted turn
  29. Judge only strict completions, using the existing fixed judge version.

Before the six conversations, run `/v1/models`, the existing budget/decode/
prefill probes, and the four-turn budget-64 smoke. These checks do not count
toward the six-conversation cell.

## Decision gates

This is a diagnostic gate, not a powered equivalence test.

- Advance only if the cell has 6/6 strict completions, at least 34/36 required
  scripted tool calls, and at least 171/180 judged turn passes (95.0%).
- If any gate fails, stop the NVFP4 optimization track rather than launching
  broad, budget, concurrency, or TP1 sweeps.
- If all gates pass, run a matching BF16 030326 no-MTP/APC-off cell before
  attributing differences to weight format or testing TP1.
- Do not test APC+MTP again. APC remains off by default.

Report whole-conversation distributions and operational failures separately
from survivor-only judge accuracy. The six runs are sequential and small, so
any p-values are descriptive screening summaries rather than confirmation.

## Matched BF16 control, locked after the NVFP4 gate passed

The NVFP4 no-MTP cell passed all three gates. The follow-up control therefore
uses BF16 030326, vLLM 0.25.1, TP2, no MTP, explicit APC-off, and every other
serving and workload control above. Run the same probes, four-turn smoke, and
exactly six budget-128 conversations with no replacements.

NVFP4 may advance to a TP1 diagnostic only if its already-observed no-MTP cell
retains 6/6 completion, at least 34/36 required calls, and at least 171/180
turn passes, and is no more than six judged turn passes below this BF16
control. This is an operational screen, not an equivalence margin justified
for confirmatory inference. A failed margin stops the NVFP4 track.
