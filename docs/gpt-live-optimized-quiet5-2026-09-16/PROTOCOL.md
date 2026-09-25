# GPT-Live prompt-optimized: five-second quiet interval

Method: **live-playback-exchanges-v2**. Fresh 25 attempts, concurrency five,
750 planned turns, no replacement of failures. The full smoke is excluded.

## Gate

Use the unchanged optimized frontend and backend prompts, GPT-Live frontend,
Terra medium backend, existing questions/audio/tools, and shared Opus 5 judge.
There is no runtime text classifier. Standard model prompts and prior cohorts
remain unchanged.

Wait until input has ended, delegated backend/tool work has completed including
continuations, playback has stopped, and five seconds of quiet have elapsed.
Restart the quiet interval on new backend activity, transcript speech, playback
activity, or completion of user input.

For each backend response that returns text, retain the receipt time of its first
nonempty text delta. Require some nonempty frontend speech fragment afterward.
Do not require a new caption group or speech starting after response.completed:
streamed speech may begin before backend generation finishes. Tool continuations
returning new text create a new observation requirement. Direct replies without
backend text do not require a delegation.

This is evidence of an opportunity to speak the result, not proof of semantic
correctness, complete delivery, or an API-issued end of speech. Receipt times can
reflect network delays. Preserve raw events and waveform alignment for review;
judge content offline. The five-second interval is a conservative timing heuristic.
It increases elapsed session time but is outside user-end-to-first-speech latency.
Unsolicited follow-up speech remains recorded, not deleted.

## Validation

- Replay of the preceding 15 attempts: all 430 committed exchanges had frontend
  speech after backend content availability. The old three-second recordings are
  not relabeled as five-second runs.
- The stopped attempt 13 transcribed the user's name but produced no frontend
  output or delegation for 90 seconds. This was not a gate holding finished speech.
  Preserve it as a no-response failure without asserting an API error.
- 17 unit/regression tests passed, including streaming-before-backend-completion,
  interim-before-result holds, direct replies, five-second timing, and tool loops.
- Before launch: full 30-turn smoke, raw-to-reconstructed transcript checks, Opus 5,
  quantitative V2V, waveform input alignment, prompt echoes, incremental/canonical
  recording equality, and replay of every gate boundary.
- Pilot acceptance depends on measurement integrity, not a quality-score threshold.
- Freeze implementation and prompt hashes before the 25-attempt cohort.

All no-response failures remain in the fixed denominator. A timeout with existing
assistant output pauses new launches for gate review; already active captures
finish and are retained. Empty-response timeouts are retained without aborting
other attempts. Do not silently replace captures.

## Artifacts and interpretation

Preserve stereo recordings, independent raw PCM, API events, backend messages,
prompt and code snapshots, and exact shared judge hash. Reconstruct spoken text
losslessly; never substitute a backend answer. Report first-speech V2V, including
interim acknowledgments, separately from answer quality. Substantive-answer onset
is not yet a validated metric. Automated ASR/alignment does not establish human
listening verification.

See the [documentation review](../gpt-live-optimized-deterministic-2026-09-16/DOCS-REVIEW.md)
for API signal limitations and the official evaluation cookbook's timing caveat.

## Validated smoke and cohort launch

Smoke `20260916T121657_gpt-live-1_4ed48a67` completed 30/30 exchanges and scored
30/30 with the shared Opus 5 judge. Every gate, reconstruction, prompt echo,
recording equality, and waveform alignment check passed. First-speech V2V mean
2132 ms, median 2080 ms, p90 2618 ms; all 30 measurable, zero negative latencies.
The smoke remains excluded from the 25-attempt cohort.

Independent ASR samples cover turns 6, 20, 21, and 29. They preserve the substantive
clarification, speaker answer, short reference answer, and farewell. ASR omitted
one short preamble from turn 20; no transcript was rewritten from ASR. This is not
claimed as a human listening audit.

Five cohort captures started after validation; the controller will capture 25
attempts and run reconstruction, judging, and quantitative audio analysis. Frozen
settings and source hashes are in `cohort/protocol.json`.
