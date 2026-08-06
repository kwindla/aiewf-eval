# Fixed-denominator analysis workflow

This directory contains the final analysis workflow for the Qwen3.6-27B
BaseTen AIEWF campaign. It never discovers runs by scanning `runs/`; inclusion
is determined exclusively by the parent campaign's `canonical.tsv`.

## Before judging or while collection is active

Run the read-only preflight:

```bash
.venv/bin/python ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py preflight
```

Preflight accepts a contiguous in-progress prefix of the frozen 60 slots. It
checks the campaign configuration, canonical/frozen-order agreement, unique run
directories, transcript shape, model identity, and thinking-mode evidence. It
reports—but does not require—judge artifacts.

## After all 60 canonical runs are judged

Run:

```bash
.venv/bin/python ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py final
```

Final mode refuses to run unless there are exactly 30 `high` and 30 `none`
conversations, every pair contains one of each arm, all 60 runs have complete
judge artifacts, and judgment coverage exactly matches observed scheduled
transcript rows.

It writes:

- `aggregates.json`: complete machine-readable protocol, arm summaries,
  effects, input hashes, and per-run audit records
- `aggregates.tsv`: compact arm-level table
- `effects.tsv`: high-minus-none point estimates and paired-bootstrap intervals
- `included-runs.tsv`: exact cohort, per-run results, judge identity, and SHA-256
  hashes for transcript/judgment/summary inputs
- `REPORT.md`: concise human-readable result report

## Scoring policy

Each arm has a fixed denominator of 30 conversations × 30 turns = 900 turns.
If a conversation stops early, every unobserved future turn fails tool use,
instruction following, KB grounding, turn-taking, and strict pass.

Strict pass matches the public benchmark definition: tool use, instruction
following, and KB grounding must all pass. Turn-taking is supplementary and
does not enter strict pass.

Two completion measures are kept separate:

- full scheduled coverage: all turns 0–29 were observed
- strict protocol completion: full coverage plus `end_session` exactly on
  scheduled turn 29

TTFAT is content-aware time to first assistant text or tool-call token. Latency
summaries are conditional on observed responses; a missing turn remains an
accuracy failure but is not assigned an invented latency.

Turn-rate intervals resample whole conversations. High-minus-none intervals
resample the 30 frozen high/none temporal pairs, preserving the blocking used
by the collection campaign. Completion rates additionally include Wilson
intervals because ordinary bootstrap intervals can collapse at 0% or 100%.
