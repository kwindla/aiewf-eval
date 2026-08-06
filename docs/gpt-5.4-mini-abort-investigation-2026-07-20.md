# gpt-5.4-mini premature-end_session investigation (2026-07-20, rev 3)

**Revision history.** Rev 1 of this note (and an earlier working note claiming "76%
turn-0 aborts / 6× alias drift") contained material errors found by adversarial review
(gpt-5.6-sol via Codex, full review in session records): a classifier bug counted
turn-28 aborts as completions, hiding a third trigger site; the April comparison used
a wrong cohort (85 mixed-effort dirs instead of the 10-run `none` cohort); and the
mechanism language overreached. This revision rebuilds the numbers from per-run
transcripts with completion defined strictly as `end_session` at the scripted turn 29.
Rev 3 adds the completed effort-medium follow-up and its adaptive-stopping and
survivor-selection caveats.

## Setup

gpt-5.4-mini, `reasoning.effort=none` (Responses API), aiwf_medium_context, July 20
2026. 48 conversation attempts: 17 no-filler, 17 with 96 trailing dots, 14 with 96
trailing dashes. Every transcript read; abort = `end_session` before turn 29.

## Strict outcome table

| config | attempts | premature exits | true completions (end_session @ t29) |
|---|---:|---:|---:|
| no filler | 17 | **17 (100%)** | **0** |
| + 96 dots | 17 | 16 (94%) | 1 |
| + 96 dashes | 14 | 11 (79%) | 3 |

Exit-site distribution (exact turns):

- **no filler:** t9 ×1, **t13 ×11**, post-t17 ×1, **t28 ×4**
- **dots:** t7 ×1, **t8 ×8**, t9 ×1, t13 ×5, t28 ×1
- **dashes:** t8 ×1, **t13 ×8**, post-t17 ×1, t28 ×1

Strict completions remain sparse (0/17, 1/17, and 3/14). The small screen does not
establish an overall completion-rate difference (no-filler vs dots: Fisher `p=1.0`;
no-filler vs dashes: `p=0.081`). Its clearest exploratory signal is **relocation**:
the turn-8 site is essentially dots-specific — exact-turn Fisher tests:
dots 8/17 vs no-filler 0/17, p≈0.003; vs dashes 1/14, p≈0.021 (exploratory,
site selected post hoc).

## Three trigger sites, all closing-adjacent pragmatics

1. **Turn 13** — *"Thanks for submitting both session suggestions. Is there food…?"*
   The dominant site in every config; identical to the April-documented failure.
2. **Turn 28** — *"One last detail…"* — the newly recognized site (masked in rev 1 by
   the classifier bug). The judge marks these exits early/inappropriate; they forfeit
   the scripted close.
3. **Turn 8** — *"Actually, I made a mistake. I will only be at the conference on
   June 5th."* — appears almost exclusively with dots (8/17 vs 0/17 no-filler).

Failure shape: nearly all exits are a bare `end_session({})` with empty text; one
dots exit (t7) batches `end_session` with a spurious `submit_session_suggestion`.
Token telemetry on exit responses is null in 43 of 44 cases (one explicit
`thinking_tokens: 0`) — consistent with, but not proof of, no reasoning on those
responses.

## Interpretation (hedged)

All three trigger utterances are closing-adjacent ("thanks", "one last", a
walking-back correction). Dots are *associated with* a large, turn-specific
redistribution of exits toward the mildest such cue. An ellipsis-pragmatics mechanism
(trailing dots read as trailing off) is one candidate explanation; equal-token dashes
not producing the shift is consistent with it, but token identity, frequency, and
decoding effects are not excluded. No causal mechanism is demonstrated here.

## April comparison: no evidence of drift

The April `none` cohort (10 runs; `docs/gpt-5.4-mini-analysis-2026-04-02.md`) under
the same strict classifier is 10/10 premature (8–9 at t13, one t6, one t28); the
one run previously labeled a "completion" is itself the t28 exit. July no-filler:
17/17 premature. **Both
cohorts are ~100% premature; there is no evidence the alias changed.** The rev-1
drift claim compared July against an 85-run mixed-effort aggregate and is void.
April data also show that nonzero effort does not eliminate the defect
(premature t13 exits: low 8/10, medium 6/10, high 4/10 — declining, never gone).

## Effort-medium follow-up: adaptively stopped screening result

A follow-up cell ran `reasoning.effort=medium`, comparing no filler with 96 trailing
dashes. The driver alternated arms until the dash arm reached eight strict completions,
then spent the remaining attempts on no filler, with a 30-attempt cap. Completion still
means `end_session` at scripted turn 29; a turn-28 call is premature.

| config | attempts | premature exits | strict completions | accuracy among strict completers | median TTFAT among strict completers |
|---|---:|---:|---:|---:|---:|
| no filler | 20 | 16 | **4/20 (20%)** | 86.7% (`n=4`) | 1117 ms |
| + 96 dashes | 10 | 2 | **8/10 (80%)** | 93.3% (`n=8`) | 1152 ms |

The strict-completion difference is large. The fixed 2×2 table gives a two-sided
Fisher exact value of `p=0.004111`, but that value is **not calibrated for this
outcome-dependent stopping and allocation rule** and is descriptive only. Exit sites
also shift: no filler exits at t13 ×15 and t28 ×1;
dashes exit at t13 ×1 and t15 ×1. Among strict completers, dashes score +6.7 points
(86.7% → 93.3%; selected-subset conversation-cluster permutation `p=0.0054`) with
a 35 ms higher sample median TTFAT. This selected-subset p-value has no
unselected-population or causal interpretation.

The accuracy and latency comparison is explicitly **survivor-selected**. Treatment
changes which conversations reach turn 29, so these conditional numbers do not estimate
the effect on an unselected population and must not be promoted into the cross-model
master accuracy table. The adaptive, capped allocation also makes the completion test a
screening result that should be confirmed with a fixed sample size. The clearest
description is therefore the observed 8/10 versus 4/20 strict-completion split under
the adaptive screen, not a calibrated treatment-effect or causal accuracy claim.

## Consequences

1. **gpt-5.4-mini never legitimately completes this benchmark at effort=none**
   (0/17 no-filler). Its per-turn pass rates (~83% in April) coexist with ~100%
   premature termination — the leaderboard row conceals the completion failure.
2. Its filler cells stay out of the cross-model master table; survivor-accuracy
   comparisons from rev 1 are void (the "survivors" included t28 aborts).
3. The relocation finding (dots → turn-8 exits, p≈0.003 exploratory) is the clearest
   effort-none signal, and it is *consistent with* the study's glyph-dependent hazard
   observations on terra and nemotron-super — with denominators now stated per
   failure mode rather than pooled.
4. The medium+dashes adaptive screen observed higher strict completion (8/10 vs 4/20), but its
   survivor accuracy is not a population estimand. Follow-ups, in priority order:
   (a) fixed-size replication of medium no-filler vs dashes with completion as the
   preregistered primary outcome; (b) frozen-history trigger factorial at t8/t13/t28
   × {dots, dashes, neutral word, none} × paraphrases removing "Actually/only",
   "Thanks", "One last detail"; (c) contemporaneous none/low/high re-run with
   preregistered n before any drift claim; (d) same probe on gpt-4.1-mini.

## Provenance

Run dirs: `runs/aiwf_medium_context/20260720T1*_gpt-5.4-mini_*`. The files
`docs/filler-study-data/gpt54mini_manifest.tsv` and
`docs/filler-study-data/gpt54mini_attempts.txt` preserve the original loose-classifier
screen and are **legacy provenance, not authoritative strict classifications**; their
`SUCCESS` labels include turn-28 exits. The strict table above was rebuilt directly
from every transcript using turn 29 as the only completion. April baseline:
`docs/gpt-5.4-mini-analysis-2026-04-02.md`. Turn script:
`benchmarks/_shared/turns.py` (answer at t28, end_session at t29). Medium follow-up:
`docs/filler-study-data/mini_medium_manifest.tsv` and
`docs/filler-study-data/mini_medium_attempts.txt`; the all-attempt run-directory map is
`docs/filler-study-data/mini_medium_all_attempts.tsv`.
