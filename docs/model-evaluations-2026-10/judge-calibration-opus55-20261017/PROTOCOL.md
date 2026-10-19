# Text judge calibration, October 17, 2026

Frozen before the first paid calibration request. This is a small diagnostic
calibration, not an independent estimate of judge accuracy or model quality.

Compare `claude-opus-5` and `claude-opus-5-5`, both with explicit **medium**
effort, adaptive thinking, no tools, and no ambient SDK setting sources. Use
two repetitions of each of four complete 30-turn conversations with each judge
and each rubric (32 calls total). Shuffle the execution schedule with seed 551017.
The legacy rubric is held fixed to isolate the model change. The candidate
`text-policy-compliance-1` separately tests policy-grounded grading, acceptance
of reasonable attendance-date clarification, and factual alternatives to gold.
It includes the assistant's actual policy and full knowledge base. Neither the
speech rubric nor the assistant's generation prompt changes.

`cases.json` records intended labels for 24 selected turns before querying.
The set includes both original first-slot Gemini conversations, necessary and
reasonable clarifications, venue refusals, successful tools, and synthetic clear
failures (wrong dates/times, fabricated navigation, missed authorized tools, and
out-of-scope weather answers). Only unambiguous dimensions receive labels.
Synthetic cases use copies of full conversations; edits are documented in the
manifest. Labels, source identities, and case annotations never enter judge
prompts. `inputs/` holds the complete blind conversation snapshots. Both rubrics
are snapshotted and hashed in `configuration.json`.

Report every labeled score, disagreement, repetition, full-conversation score,
and operational error. Do not choose a repetition after seeing results. Retries
are allowed only for operational/parse errors, retained in the attempts ledger.
Select the candidate if it correctly grades all clear-failure controls, accepts
the intended legitimate clarifications/refusals, and matches or improves labeled
agreement over the legacy rubric. Otherwise report unresolved disagreements and
retain the previous default. Prefer Opus 5.5 medium if it meets these criteria;
this is an operational choice, not a claim of statistical superiority.

After adoption, freeze the exact prompt/model/effort and rejudge all 30 existing
conversations from each Gemini cohort once, using new output directories. Keep
all original judgments unchanged. Report 900 turns per cohort, independent
conversation-cluster bootstrap uncertainty for the difference, and failures with
reasons. This matched regrade changes grading only: generation dates, dependency
versions, responses, costs, and latency remain those of the original campaigns.
Other leaderboard rows keep their historical judges and are not recalibrated.

[Evidence storage and local audit instructions](../README.md).
