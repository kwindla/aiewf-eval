# Completed-conversation sampling policy

Authorized by the user September 16, 2026: top up incomplete conversations and
replace the startup-invalid Gemini low capture; report failed conversations as a
separate metric with the turn on which each stopped.

- Score the first 25 complete, harness-valid conversations per model/configuration.
- Preserve every original and additional attempt. Do not choose by quality score.
- Exclude the entire incomplete conversation from completed-conversation accuracy,
  including its observed prefix. Keep that prefix and its judgments in diagnostics.
- Failure rate = incomplete model/service conversations / harness-valid attempts.
  Report the count, denominator, percentage, reason, completed-turn count, and
  one-based failure turn (raw zero-based index retained in JSON).
- Distinguish explicit API errors from no-response timeouts. Exclude harness defects
  from model/service reliability and report their count separately.
- Preserve original attempt-based aggregates. These new conditional accuracy scores
  have a different interpretation and must be labeled accordingly.
- No frontend/backend prompt, KB, scripted audio, tool, or judge-rubric changes.
- Keep Gemini medium's earlier and interaction-aware cohorts separate; do not pool
  distinct harness configurations. Do not pool old invalidated diagnostic cohorts.
- Top-ups use the corresponding cohort's capture/measurement implementation.
  Legacy Gemini adds a startup-only guard preventing commit before scripted input.
- Required initial top-ups: Gemini 3.1 minimal +1, Gemini 3.8 low +1 replacement,
  Gemini 3.8 medium interaction-aware +2, GPT-Live optimized quiet5 +1.
- Capture concurrently below the existing five-per-provider cap. Retry incomplete
  attempts as new logged conversations, never by replacing a log or replaying a turn.
  After six additional attempts in an arm, stop for review if still incomplete.
- A full capture that fails validation is held for review rather than automatically
  replaced. Run Opus 5 and quantitative audio/alignment checks on new captures.
- No replacement is needed for the two reviewed audio holds: waveform-aligned input
  locations resolve the standard Live timestamp issue without latency changes; the
  optimized short answer is present in audio but unmeasured by VAD. Report coverage.

Outputs: attempts.jsonl, started.jsonl, aggregate.json, REPORT.md, individual capture
and processing logs, plus immutable original run artifacts and separate audio reviews.

## Completion

All nine cohorts now have 25 validated complete conversations. Six top-up attempts
produced five accepted completions and one additional GPT-Live no-response failure.
Every added completed capture passed shared judging and quantitative validation.
The startup regression test passed. Reporting checks verified 225 unique selected
conversations across cohorts, 750 scored turns per row, exact failure denominators,
and an unchanged text-model README prefix. See REPORT.md and aggregate.json.
