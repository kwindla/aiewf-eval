# Why fresh Ultra thinking-on scores 91.9%

The canonical result is **827/900 passes, 73 failed turns**, across 30 complete
conversations. No score or transcript was edited for this audit. Every failure
is inventoried in [audit.json (local archive)](../../../README.md) and shown with its complete answer and
judge explanation in [EVIDENCE.md (local archive)](../../../README.md).

| Failure family | Turns | Assessment |
|---|---:|---|
| Session-count policy | 22 | Correct counts, but the system explicitly says not to calculate them. |
| Workshop-pass requirements | 14 | Correct requested sessions, plus an unsupported access restriction. |
| Venue/navigation additions | 29 | Correct refusal, room, or recommendations, plus unsupported venue details or prohibited directions. |
| Wrong date / forgotten attendance correction | 2 | One wrong workshop date; one return to the superseded attendance date. |
| Other grounding additions | 4 | Panel, meetup status, menu, or vegan-catering claims. |
| Extra track count | 1 | Correct official 19, followed by calculated track-instance totals; policy scope is debatable. |
| Early support ticket | 1 | Debatable: the tool schema required no specificity beyond an issue description; early-call realignment is allowed. |

**65 of 73 misses (89%) are session-count policy violations or extra unsupported
claims attached to otherwise useful answers/refusals.** They are concentrated
in a few repeated questions, rather than 73 unrelated breakdowns. Strict pass
requires all three dimensions to pass; a single extra assertion fails the whole
turn. Correctly identifying the requested session does not cancel an added claim.

## What changed in judging

Before the Opus 5.5 migration, the generic v4 rubric said instruction-following
passes if the assistant "directly answers the question OR advances the task" and
grounding fails only for clear factual contradictions. Its conversation prompt
contained golden answers and expected tools, but its system rubric did not include
the complete assistant policy or knowledge base.

The current `text-policy-compliance-1` rubric explicitly enforces the actual
assistant policy and supplies the full policy and KB to the judge. For example,
the generator's policy says not to calculate session totals and not to infer
facts beyond the KB. A correct answer to "how many sessions?" can therefore be
a policy failure even though the arithmetic and listed sessions are accurate.
The no-counting rule was already in the assistant's instructions; it was not
added to Ultra's prompt for this study.

The [paired rubric diagnostic](../rubric-comparison/REPORT.md) judges these exact
30 transcripts with the legacy rubric and the same Opus 5.5 Medium model. It
holds the formatted conversation prompt, transcript, model, and effort fixed;
the rubric/reference package changes. No more Ultra conversations are generated.
The canonical 91.9% result remains the published score.

On these **identical 900 turns**, the pre-upgrade v4 rubric scored
**96.6%**, versus **91.9%**
for the current policy/KB rubric, using the same Opus 5.5 Medium judge.
The measured legacy-minus-current gap is
**4.7 percentage points**.
There were 58 legacy-pass/current-fail
turns and 16 changes in the other
direction. Individual disagreements include judge variability; the controlled
configuration change is the rubric plus supplied reference policy/KB.

The new rubric is not uniformly harsher. Of the 16 improvements, ten accept
policy-compliant venue refusals that the old rubric rejected for failing to echo
the golden answer's support-ticket context. Five accept speaker affiliations
that match the benchmark KB, where the old judge instead substituted outside
knowledge (for example, it rejected the KB's Hamming AI/Taktile affiliations).
One accepts an early support ticket whose required arguments were already known.
The 58 newly failed turns include all 22 session-count violations, all 14 added
pass requirements, and 17 venue/navigation additions.


## Where the judge deserves qualification

- **Unsupported is not the same as proven false.** For 14 workshop-pass answers,
  Ultra correctly listed the June 4 sessions and then added a pass requirement.
  The KB says the expanded pass adds exclusive June 3 workshops; it does not
  explicitly settle access to the June 4 session whose format is Workshop.
  Ultra inferred a requirement, contrary to the no-inference policy. Some judge
  explanations overstate this as a definitive contradiction. The omission of a
  rule from the KB alone does not prove its opposite.
- **Venue embellishments:** the KB lists `Juniper: Expo Sessions` and separately
  `Salons 9-15: Expo Hall`; it never supplies their physical relationship or
  locates an information booth. Ultra often gives the right room/refusal and
  appends those details. These are ungrounded additions, but the canonical
  grounding rubric's "clear factual contradiction" wording versus enforcement
  of the system's no-inference policy leaves dimension assignment debatable.
  They are not evidence that the requested room was wrong.
- **One likely overstrict tool judgment:** slot-041-on, user turn 17, called
  technical support with the already-known name and `trouble with the mobile app`.
  Those satisfy the literal schema. The judge demanded a more specific problem
  before acting and penalized the early ticket, although the rubric allows early
  calls. The current judge accepted the semantically equivalent early call in
  slot-060-on at that same user turn. This is a concrete consistency concern,
  worth treating as a debatable label rather than an obvious failure.
- **One additional track-count judgment is debatable:** the answer gives the
  correct official 19 tracks, then calculates track instances. The policy bans
  totals of sessions/speakers/topics; applying it to an extra track-instance
  summary is an interpretation. Neither debatable singleton explains a large
  score change (each is 0.11 percentage points).

These qualifications do not authorize selective score edits. Any rubric
clarification should be tested consistently across models, not only on Ultra.

## Why this does not establish a model regression

The [old published rows (local archive)](../../../README.md)
showed 100.0% / 98.3%. They are older legacy summaries with a different cohort,
serving/budget profile, and historical judge version. The exact original
near-perfect cohort is not established by those row labels. The freshly measured
91.9% cannot by itself be interpreted as an eight-point capability regression.
Use the same-output rubric diagnostic to measure scoring sensitivity, and keep
the historical-cohort uncertainty separate.

The current README publishes only the fresh thinking-on result. Ultra generation
testing is finished; no smaller-thinking-budget experiment is planned.

[Evidence storage and local audit instructions](../../../README.md).
