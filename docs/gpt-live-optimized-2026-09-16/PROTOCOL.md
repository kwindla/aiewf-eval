# GPT-Live prompt-optimized experiment

Status: cohort launcher stopped September 16, 2026 after a coordinator defect surfaced. Started captures are diagnostic; see cohort-paused.json. Approved to proceed
by the user. Pilot attempts are diagnostic and excluded from a future 25-attempt
cohort. Freeze the validated implementation before that cohort; concurrency five,
no replacement of failed attempts, denominator 750 planned turns.

## Configuration

- Frontend: gpt-live-1, Marin, compact voice/delegation prompt.
- Backend: gpt-5.6-terra medium, full unchanged KB and task instructions.
- Existing benchmark questions, audio, tools and simulated results.
- Named row: **GPT-Live prompt-optimized**.
- Method: **live-exchange-v1**, explicitly different from the standard timing-only
  protocol. Scores are experimental, not a clean causal estimate of prompt effects.
- Coordinator: gpt-5.6-luna low classifies whether speech is answer/clarification/
  refusal/social or progress-only. No KB, expected answers, tool expectations or
  scores. Incorrect substantive answers must still advance. Cache decisions on
  exact spoken content; unchanged progress-only speech waits until timeout.
- Keep the corrected input-end quiet interval, pending-delegation/tool holds and
  playback drain. Recheck all conditions after coordinator inference. Log inference
  overhead separately; never add it to measured first-speech latency.
- The first seven-turn probe predates incremental disk recording; retain its exact
  adapter snapshot. Subsequent pilots include independent PCM tracks with sample
  maps as well as the complete stereo WAV.

## Processing

Capture is separate from judgment. Reconstruct spoken exchanges losslessly from
raw output fragments, preserving acknowledgments and actual tools. Save source
hashes and per-fragment event IDs. Block automatic judging on text disagreement,
nonmonotonic transcript intervals or cross-exchange overlap pending audio review.
Version 1 makes no speculative cross-user reassignment. Unresolved grouping remains
an explicit measurement failure rather than being repaired using expected answers.

Opus 5 applies the shared speech-v2-policy-compliance-1 task-quality rubric only
AFTER reconstruction. Backend messages/calls/results remain a separate diagnostic
artifact and never replace spoken text. Transcript judging does not claim audio
turn-taking success: measure overlaps/gaps separately from actual recordings.

Quantitative processing uses canonical Silero, signed speech latencies, measurement
coverage, and waveform alignment. Report first speech separately from substantive
answer timing; the latter needs semantic span annotation and audio verification,
not transcript receipt time. No claim of a listening audit from VAD or ASR alone.

## Pilot acceptance

Check question routing, short continuations, action confirmation, and closure in a
full 30-turn conversation. Verify prompt echoes, raw-to-reconstructed provenance,
coordinator decisions, full/partial recording paths and quantitative metrics. Unit
checks cover progress-only holds, concurrent new speech, and incorrect answers
being accepted as completed responses. Add independent transcription/listening
artifacts for representative exchanges before promoting experimental scores.

The completed standard cohort and its recordings/scores remain unchanged. It scored
707/750 (94.3%); its gate audit passed, but one WAV has a late input-alignment offset
and its canonical eligibility remains on hold. This experiment does not repair or
replace that cohort. The original prompt design remains archived under
../gpt-live-prompt-optimized-design-2026-09-15/.

## Pilot outcome

The seven-turn routing probe scored 7/7. The separate full pilot completed 30/30
exchanges and scored 28/30 after reconstruction. Full-pilot reconstruction, waveform
alignment, prompt echoes, input sequence, gate timing and incremental-recording
equality all passed. Four real-API coordinator boundary probes and 14 regression
tests passed. Pilot captures are excluded from the cohort. Independent transcription
artifacts are retained; substantive word timing remains unvalidated.

Full pilot: `runs/aiwf_medium_context_speech_live_optimized/20260916T064919_gpt-live-1_686b2548`.
Cohort settings and source hashes: [frozen protocol](cohort/protocol.json).

## Cohort blocker discovered after pilot

The completion classifier labeled legitimate clarification questions and substantive
non-numeric answers as progress_only, causing 90-second timeouts. Four narrow
synthetic probes and the successful pilot did not establish adequate reliability.
These are harness-induced failures, not OpenAI API failures or model-quality scores.
The launcher was stopped; already started calls retain their original configuration
and artifacts. A revised coordinator must be validated against these actual failures
before any new scored cohort. No results are promoted to the README table.
