# Opus 5.5 medium text judge calibration

The protocol and intended labels were frozen before any calibration grade returned. Four blind 30-turn conversations were graded twice per model and rubric. There are 24 labeled target turns, with only unambiguous dimensions labeled. These deliberately selected cases diagnose the disputed behaviors; they do not estimate general judge accuracy. The archived judge effort was not recorded; the legacy-rubric arm controls effort at medium for both models rather than recreating an undocumented historical default.

| Rubric | Judge (medium) | Labeled dimension agreement | Clear-failure agreement |
|---|---|---:|---:|
| legacy | claude-opus-5 | 115/122 | 20/20 |
| legacy | claude-opus-5-5 | 112/122 | 20/20 |
| candidate | claude-opus-5 | 122/122 | 20/20 |
| candidate | claude-opus-5-5 | 122/122 | 20/20 |

## Repetitions and disagreements

- legacy, claude-opus-5: 2/360 full-conversation dimension scores differ between repetitions. A/8/instruction_following, C/8/instruction_following
- legacy, claude-opus-5-5: 1/360 full-conversation dimension scores differ between repetitions. A/22/instruction_following
- candidate, claude-opus-5: 1/360 full-conversation dimension scores differ between repetitions. C/18/kb_grounding
- candidate, claude-opus-5-5: 1/360 full-conversation dimension scores differ between repetitions. D/25/instruction_following

Every labeled disagreement follows (case / zero-based turn / repetition).

- **legacy/claude-opus-5, B/8/r2, instruction_following**: expected True, got False. The golden immediately supplied two June 5th recommendations since the user had already stated 'all day' attendance and their topic interests. The assistant instead asked again whether she'd be there all day — unnecessary re-confirmation of information it already had, failing to advance the task. No factual error.
- **legacy/claude-opus-5, B/19/r2, instruction_following**: expected True, got False. This is an in-scope event question (venue directions to Salon 2) that the assistant cannot fulfil, so the golden declined while acknowledging the broken app maps and noting the tech support ticket already filed. The assistant instead reused the same generic out-of-scope template from turn 18, omitting both the acknowledgement of the user's stated situation and the reference to the filed ticket.
- **legacy/claude-opus-5, D/19/r1, instruction_following**: expected True, got False. This is an event-related request the assistant cannot fulfil, so the decline needed to do the work of the turn. The golden names Salon 2, acknowledges that the app maps are broken, notes the tech support team has already been notified, and offers session/speaker help. The assistant instead gave a two-sentence bare refusal that omits all of that context, so instruction_following fails. No factual error stated.
- **legacy/claude-opus-5, D/19/r2, instruction_following**: expected True, got False. This is an event-related request the assistant cannot fulfil. The golden declines while naming Salon 2, acknowledging the broken maps, noting the tech support ticket already filed, and offering alternatives. The assistant gave a bare two-sentence refusal that omitted all of that context.
- **legacy/claude-opus-5, B/8/r1, instruction_following**: expected True, got False. The user only corrected the date; the assistant already knew the user attends all day and knew the topic interests. Instead of giving June 5 recommendations as the golden did, it re-asked 'all day, or morning or afternoon' — unnecessary re-confirmation of information it already had, so the turn did not advance the task. No factual error and no function expected.
- **legacy/claude-opus-5, B/19/r1, instruction_following**: expected True, got False. The request is about the event (finding Salon 2 in the venue) but the assistant cannot fulfil it. The golden declines while naming Salon 2, acknowledging the broken app maps, and referencing the tech support ticket already filed. The assistant instead reused a near-verbatim generic scope refusal from the previous turn with none of that context, so the required contextual decline was not delivered.
- **legacy/claude-opus-5, A/8/r1, instruction_following**: expected True, got False. The user already stated at turn 7 they would attend all day, and only changed the date to June 5th. The golden immediately supplied two June 5th recommendations; the assistant instead re-asked for the time of day it already had, failing to advance the task.
- **legacy/claude-opus-5-5, B/8/r2, instruction_following**: expected True, got False. The golden immediately provides June 5 recommendations. The assistant instead re-asks whether the user will be there all day, although the user already said all day and only corrected the date. The deficiency is that it failed to deliver the June 5 session recommendations and asked an unnecessary clarification.
- **legacy/claude-opus-5-5, B/19/r2, instruction_following**: expected True, got False. This is an event-related request (venue directions) the assistant cannot fulfil, so Step 2 applies. The response near-verbatim reuses the turn 18 refusal template. It does not acknowledge the user's broken app maps or mention the tech support ticket already filed, both of which the golden includes.
- **legacy/claude-opus-5-5, D/19/r2, instruction_following**: expected True, got False. Venue directions are an in-scope event topic the assistant cannot fulfil (Step 2). The golden acknowledges the broken app maps, names Salon 2, and notes the tech support ticket already filed. The assistant gave a bare refusal with none of this context.
- **legacy/claude-opus-5-5, A/8/r1, instruction_following**: expected True, got False. The user only corrected the date; the assistant already knew the user's interests and that they would attend all day. The golden immediately gives June 5th recommendations. The assistant instead asks a redundant question about attendance time and provides no recommendations, which is an unnecessary clarification.
- **legacy/claude-opus-5-5, A/19/r1, instruction_following**: expected True, got False. Venue directions are an in-scope request the assistant cannot fulfil (Step 2). The assistant names Salon 2 and suggests asking staff. It fails to acknowledge the user's situation that the app maps are broken and does not mention the tech support ticket already filed, both of which the golden includes.
- **legacy/claude-opus-5-5, A/8/r2, instruction_following**: expected True, got False. The golden immediately provides two June 5th recommendations, since the user's interests and all-day attendance are already known. The assistant omits recommendations and asks a redundant clarifying question about time of day.
- **legacy/claude-opus-5-5, A/19/r2, instruction_following**: expected True, got False. This is an in-scope request the assistant cannot fulfil (Step 2). The golden acknowledges the user's situation (the app maps aren't working) and notes that the tech support team was already notified. The assistant names Salon 2 and suggests on-site staff, but omits both the acknowledgement of the app problem and the reference to the ticket filed at turn 17.
- **legacy/claude-opus-5-5, D/19/r1, instruction_following**: expected True, got False. Venue directions are an in-scope topic the assistant cannot fulfil (Step 2). The golden acknowledges the broken maps, names Salon 2, and notes the tech support ticket already filed. The assistant gave a bare refusal with none of these elements.
- **legacy/claude-opus-5-5, B/8/r1, instruction_following**: expected True, got False. The golden gives two June 5th recommendations ('Serving Voice AI at Scale' and 'Ship it! Building Production-Ready Agents'). The assistant asks another clarifying question and provides no recommendations, so it does not deliver what the turn required.
- **legacy/claude-opus-5-5, B/19/r1, instruction_following**: expected True, got False. Directions to Salon 2 are an event topic the assistant cannot fulfil (Step 2). The golden acknowledges that the user's app maps are broken and notes that tech support has already been notified. The assistant gives a near-template scope refusal echoing turn 18 and omits both of those points.

## Full conversation scores

These totals include unlabeled turns and are reported without selecting a favorable repetition.

| Rubric | Judge | Case | Repetition | Strict passes / 30 |
|---|---|---|---:|---:|
| candidate | claude-opus-5 | A | 1 | 30/30 |
| candidate | claude-opus-5 | A | 2 | 30/30 |
| candidate | claude-opus-5 | B | 1 | 30/30 |
| candidate | claude-opus-5 | B | 2 | 30/30 |
| candidate | claude-opus-5 | C | 1 | 24/30 |
| candidate | claude-opus-5 | C | 2 | 24/30 |
| candidate | claude-opus-5 | D | 1 | 29/30 |
| candidate | claude-opus-5 | D | 2 | 29/30 |
| candidate | claude-opus-5-5 | A | 1 | 30/30 |
| candidate | claude-opus-5-5 | A | 2 | 30/30 |
| candidate | claude-opus-5-5 | B | 1 | 30/30 |
| candidate | claude-opus-5-5 | B | 2 | 30/30 |
| candidate | claude-opus-5-5 | C | 1 | 24/30 |
| candidate | claude-opus-5-5 | C | 2 | 24/30 |
| candidate | claude-opus-5-5 | D | 1 | 29/30 |
| candidate | claude-opus-5-5 | D | 2 | 29/30 |
| legacy | claude-opus-5 | A | 1 | 29/30 |
| legacy | claude-opus-5 | A | 2 | 30/30 |
| legacy | claude-opus-5 | B | 1 | 28/30 |
| legacy | claude-opus-5 | B | 2 | 28/30 |
| legacy | claude-opus-5 | C | 1 | 23/30 |
| legacy | claude-opus-5 | C | 2 | 24/30 |
| legacy | claude-opus-5 | D | 1 | 28/30 |
| legacy | claude-opus-5 | D | 2 | 28/30 |
| legacy | claude-opus-5-5 | A | 1 | 28/30 |
| legacy | claude-opus-5-5 | A | 2 | 27/30 |
| legacy | claude-opus-5-5 | B | 1 | 28/30 |
| legacy | claude-opus-5-5 | B | 2 | 28/30 |
| legacy | claude-opus-5-5 | C | 1 | 23/30 |
| legacy | claude-opus-5-5 | C | 2 | 23/30 |
| legacy | claude-opus-5-5 | D | 1 | 28/30 |
| legacy | claude-opus-5-5 | D | 2 | 28/30 |

## Judge runtime and cost

| Rubric | Judge | Median SDK duration | Cost for 8 calls |
|---|---|---:|---:|
| legacy | claude-opus-5 | 54.3s | $1.4196 |
| legacy | claude-opus-5-5 | 40.5s | $1.1442 |
| candidate | claude-opus-5 | 47.9s | $1.5816 |
| candidate | claude-opus-5-5 | 40.5s | $1.2235 |

## Configuration and operational notes

Successful calibration calls cost $5.3689 according to SDK usage metadata. The first two requests were rejected before grading because SDK 0.1.73 bundled Claude Code 2.1.128; Opus 5.5 requires at least 2.1.280. The SDK was pinned to 0.2.168, bundling 2.1.301, before any successful grade. All errors are retained in `attempts.jsonl`; original attempt directories were preserved.

Candidate revision: `text-policy-compliance-1`; system-prompt SHA256 `70cdc7982a5f2485375e22ff43761ebf74acf21e38bde5d8518a9e95f6cfd3d2`. Explicit medium effort and adaptive thinking were used for both judges; tools and ambient setting sources were disabled. Actual response model IDs, effort, SDK version, user/system prompt hashes, and costs are saved with every result.

The candidate follows the text assistant's actual governing policy. It accepts reasonable clarification after an attendance-date correction and policy-compliant venue refusals; it still rejects invented navigation, wrong dates/times, missed tools, and unsupported weather answers. Empty assistant text with a valid tool call remains accepted, matching the text capture harness. This is a rubric correction informed by the disputed examples, not an independent holdout evaluation. The speech rubric is unchanged.

## Decision

Adopt Opus 5.5 medium with text-policy-compliance-1. Both judge models matched all 122 preregistered dimension labels, including all 20 clear-failure checks, under this rubric. With the legacy rubric, Opus 5 matched 115/122 and Opus 5.5 matched 112/122: the rubric correction, rather than the model upgrade alone, resolves the disputed grading. Opus 5.5 is the requested operational choice and measured 40.5s median SDK duration versus 47.9s for Opus 5 on the candidate rubric, at lower total cost. This small, deliberately selected set does not establish general judge superiority.

[Evidence storage and local audit instructions](../README.md).
