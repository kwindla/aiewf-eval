# GPT-Live corrected-gate standard cohort

25 fresh attempted conversations, concurrency five, 750 planned turns. No imported
pilot attempts and no capture replacements. Previous 25 captures remain diagnostics
under ../speech-v2-live-medium-2026-09-15/.

The only runtime change is resetting the three-second quiet interval at input end,
preventing an earlier backchannel from immediately advancing the script. The exact
shared full speech-v2 prompt is used in both frontend and backend; the optimized
prompt experiment remains inactive. Model gpt-live-1, backend gpt-5.6-terra medium,
voice Marin, unchanged tools, scripted audio, timeouts and recovery disabled.

Every attempted conversation retains its fixed 30-turn accuracy denominator.
Judge: Opus 5 with speech-v2-policy-compliance-1. Quantitative validation includes
canonical Silero speech-end to speech-start V2V, signed overlap/missing measurements,
waveform input alignment, and an audit of input-end to turn-commit intervals.
Explicit API/rate-limit errors and incomplete captures are reported separately.
Judging may retry execution/parse failures only. Source/prompt hashes and dependencies
are frozen in protocol.json; each capture stores its own implementation snapshot.

User approved the fix and this additional 25-attempt cohort on September 15, 2026.
