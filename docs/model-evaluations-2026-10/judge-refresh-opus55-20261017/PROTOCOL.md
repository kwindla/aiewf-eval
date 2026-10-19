# Overnight README refresh

Requested October 17, 2026; overnight work began October 18 at approximately
06:24 UTC. Scope is every remaining README configuration after the explicitly
requested removals: 43 text rows and nine voice configurations. No new
generation calls are required or authorized by this refresh.

Search original captures, unjudged captures, nested projects, other worktrees,
and local compressed archives. Deduplicate copies, retain configuration
variants in the inventory, and resolve conflicting versions before selecting.
Select independently of all judge scores. For each text configuration use the
most recent 30 complete, harness-valid conversations; voice uses 25. If fewer
exist, use all eligible complete conversations and disclose the shortfall.
Do not combine different budgets, effort levels, filler arms, quantizations,
or serving stacks to manufacture a 30-conversation cohort. If a retired route
is remapped to a later study, identify the replacement serving configuration.

A complete conversation contains scripted indices 0–29 exactly once, with an
assistant response or tool call at every index. Recovery records are retained
in source snapshots as conversation and latency evidence but are not additional
scored turns. Implementation clarification: the existing judge formatter receives
scripted records only; recovery records are omitted from the judge prompt. This
is the existing behavior used throughout, not a mid-campaign prompt change. A missing
required tool is a quality issue, not a completeness exclusion. Known capture
defects and diagnostics with different prompts/settings are excluded with a
recorded reason, never because of their scores. The same scripted user inputs
and governing prompt must apply; voice reconstructions must preserve actual
spoken content, never substitute backend answers.

Freeze each cohort's candidate decisions, selected membership, source hashes,
and input snapshots before requesting its judgments. Preserve the original
transcripts, logs, judgments, audio evidence, archives, and historical campaign
reports. Work only in separate outputs. Appending other frozen configurations
does not permit revising an already graded cohort based on its results.

All quality judgments use `claude-opus-5-5`, explicit `medium` effort, adaptive
thinking, no tools, and no project settings. Text retains the calibrated
`text-policy-compliance-1` rubric, SHA-256
`70cdc7982a5f2485375e22ff43761ebf74acf21e38bde5d8518a9e95f6cfd3d2`.
Voice retains `speech-v2-policy-compliance-1`, SHA-256
`c5a8a3b3c3240172cc392529a1784b59d2913a32e2ec2ba843a456415b75ef27`.
The judge receives no generating-model identity, source paths, prior scores,
or leaderboard ordering. Audio turn-taking evidence is fixed separately from
the new quality grades; unscored turn-taking remains null.

Text finishes before voice judging begins. Requests run with bounded
concurrency and save every attempt. Credit/balance failures impose a shared
cooldown of at least 90 seconds before retrying, without a credit-retry limit.
Other operational errors have four attempts with backoff and require inspection
if unresolved. Resume completed jobs only when their transcript, model, effort,
rubric, and complete turn coverage match the frozen configuration. Accept the
first structurally valid judgment, regardless of scores.

Recalculate text latency on the selected conversations, excluding scripted
turn zero and including recorded recovery responses. Report median, linear
95th percentile, maximum, sample counts, and missing observations. Preserve
any remaining distinction between legacy first-token and content-aware timing;
never relabel an uncorrectable first-token measurement as TTFAT. Voice latency
is recomputed from the selected per-turn waveform measurements, preserving
alignment decisions, positive-only publication, signed diagnostics, tool versus
non-tool grouping, and null unavailable measurements. Do not rerun captures or
substitute aggregate values from a different cohort.

Publish complete cohort manifests, configuration qualifications, sample sizes,
component errors, strict accuracy, conversation-cluster intervals, actual judge
usage/cost, and recalculated latency. Update README, standalone leaderboard,
chart, and consistency tests together. Retain earlier results as historical
artifacts and clearly identify the new conditional-on-completion analysis.

## Operational clarification, October 18, 2026

One originally ungraded Nano 512 input received four provider
`reasoning_extraction` safeguard rejections. Its recorded assistant output
contains analysis-like text and closing think tags emitted by the tested model.
That visible source output is preserved, not removed or converted into an answer.
A single additive task clarification explains that the request evaluates the
recorded response and asks for brief, concrete score justifications, rather than
reconstructing hidden reasoning. It does not change criteria, source content,
the system rubric, judge model, or medium effort. The exact text, affected job,
and reason are recorded in task-clarifications.json; the actual judge input SHA
and clarification are recorded in successful metadata. No successful prior grade
exists for this case, and no score-based retry or selection is permitted. If the
provider still rejects the clarified task, preserve it unresolved.

## Provider-rejected inputs and publication

The clarification was also rejected. Further `reasoning_extraction` provider
rejections are preserved as explicit unscored dispositions without additional
rephrasing, content removal, judge substitution, or cohort replacement. Ordinary
operational retries and the credit/balance 90-second policy remain unchanged.
Processing proceeds through all remaining text inputs before voice. Mode COMPLETE
markers now mean every selected input has a terminal disposition; their
all_quality_complete field distinguishes valid grades from provider-blocked inputs.

Final metric publication requires a validated grade or an explicit matching
provider-blocked record for every selected input. It does not imply every quality
grade exists. The aggregate records selected, graded, and blocked counts and
all_quality_complete separately. An affected cohort retains its original source N
and all recalculated latency evidence, but its full-cohort pass/error point
estimates and bootstrap intervals are unavailable. Known grades and possible
full-cohort bounds are preserved without imputing unknown turns. No partial
subset is silently promoted to the original cohort's score. Original voice
audio and reliability metrics are independent of quality-judge availability.

Aggregate status fields are explicit: `processing_complete` requires every input
to have a terminal disposition. `complete` and `all_quality_complete` require valid
quality grades for every selected conversation and remain false while any case
is provider-blocked. Publication of available measurements does not mark missing
quality grades complete.

## User-requested retry, October 18, 2026

The user explicitly requested another attempt on all three rejected inputs.
Retry these inputs once with the same frozen source, rubric, model, medium effort,
and existing per-input clarification where present. Preserve earlier rejection
records and completion/verification snapshots under `retries/20261018/`.
Accept the first structurally valid grade; a repeated safeguard rejection remains
unscored. Credit and ordinary operational failures retain the campaign retry
policy. If dispositions change, preserve the original text completion marker and
write a revised marker before republishing metrics. All attempts remain in the
original append-only ledger.

## Final-answer cleanup, October 18, 2026

The user requested stripping leaked reasoning, without investigating the old
parser. Apply `think-sections-v1` to all selected judge inputs before sending
them: remove `<think>...</think>` spans and analysis before an initial `</think>`
when the opening tag was supplied by the chat template. Remove only explicitly
delimited sections; do not classify prose as reasoning. Reject nested/unclosed
or otherwise ambiguous markers. Preserve final text and actual tool actions;
an empty final answer with a valid tool call remains a tool-only turn. Recovery
records were already excluded from grading and do not trigger rejudging alone.

Audit every frozen conversation independently of its previous grade or rejection.
Archive prior grades and blocked dispositions under `reasoning-cleanup/` and
rejudge every changed conversation in full, accepting its first valid grade with
the same rubric, model, and effort. Drop the obsolete analysis clarification for
cleaned inputs. Reuse unchanged grades. Preserve raw captures, source hashes,
membership, and latency observations. Record preprocessing version, changed turn
indices, removed character counts, and original/answer hashes in manifests and
judge metadata; validate these when resuming. Count superseded successful grades
in total request cost. Before/after score changes also include judge variability.

[Evidence storage and local audit instructions](../README.md).
