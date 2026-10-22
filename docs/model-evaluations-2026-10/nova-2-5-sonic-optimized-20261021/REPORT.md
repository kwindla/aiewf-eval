# Nova 2.5 Sonic: prompt-optimized benchmark

**97.7% strict pass**, 25 complete 30-turn conversations (750 turns), Opus 5.5 Medium,
unchanged speech-v2 rubric. The 95% conversation-bootstrap interval is 96.5–98.8%
(20,000 draws; seed 20261018). The standard-prompt Nova Sonic row is 86.0% (82.9–88.8%).

| Measure | Prompt-optimized | Standard row |
|---|---:|---:|
| Strict quality | 733/750 | 645/750 |
| Tool use | 739/750 | 676/750 |
| Instruction following | 739/750 | 658/750 |
| KB grounding | 746/750 | 750/750 |
| Audio turn-taking | 750/750 | 750/750 |
| Unsuccessful attempts | 1/26 (3.8%) | 0/25 (0.0%) |
| Conversations with no tool calls | 0/25 | 8/25 |
| First-speech positive P50 / P95 | 1120ms / 1632ms | 1152ms / 1618ms |
| Non-tool median / maximum | 1088ms / 1984ms | 1120ms / 1888ms |
| Tool mean | 1412ms | 1485ms |
| Silence pad mean | 165ms | 181ms |
| Positive / nonpositive / missing timing samples | 750 / 0 / 0 | 750 / 0 / 0 |

## What changed

Only two inputs differ from the standard Nova Sonic row:

- **The SYSTEM policy text**, in [`prompts/system.txt`](../../../benchmarks/aiwf_medium_context_speech_nova_sonic_optimized/prompts/system.txt)
- **The tool and parameter descriptions**, in [`tools.py`](../../../benchmarks/aiwf_medium_context_speech_nova_sonic_optimized/tools.py)

Everything else is inherited from `aiwf_medium_context_speech_v2` unchanged: the 30 scripted
turns and audio, the knowledge-base bytes, tool names, parameters, types and required
arguments, the USER reference layout (`MTE_NOVA_KB_CONTEXT_ROLE=user`), the adapter, and the
judge rubric. The judge still grades against the original speech-v2 policy, so the optimized
policy keeps every speech-v2 rule. There is no change to completion gating or message
structure, so this is a prompt-only comparison with the standard Nova Sonic row. `tests/test_nova_sonic_prompt_integrity.py` checks these properties.

The policy restates the speech-v2 rules in the structure Nova 2.5 Sonic follows best — bracketed
sections, one rule per line — and adds general rules for the failures observed in the standard
row:

| Standard-row failure | Rule added |
|---|---|
| Claims an action succeeded without calling its tool; 8/25 conversations made no tool calls | Five actions happen only through a tool call; call it in the turn the user asks or accepts; confirm only after the result |
| `end_session` never called when the user finishes | Name what counts as finishing and call `end_session` in the same turn as the farewell |
| Vote claimed but `vote_for_session` not called | The session ID is a call argument found on the session's `Session ID` line, used without being spoken |
| Answers "when are the workshops?" with recommendation intake | When/where/who questions get a direct answer even when the user mentions deciding what to attend |
| Only offers to describe a track instead of describing it | Count questions get the session kinds and days in the same reply |
| Points users to staff or desks that aren't listed | Offer a next step only when the tools or knowledge base provide one |

Each tool is called at most once per turn, so the model asks which session the user means
instead of voting for every candidate at once. A short closing checklist repeats the three most
frequent checks. The policy and tool descriptions contain no session names, session IDs, user
names, preferences or scripted turn wording.

## Remaining failures

| Turn | Failures | Behavior |
|---|---:|---|
| 24 (vote for the chosen session) | 8/25 | Claims the vote was recorded without calling `vote_for_session` |
| 4 (session count) | 3/25 | Places MCP talks on an extra day (2), or states a count of workshops (1) |
| 29 (user is finished) | 3/25 | Farewell without `end_session` |
| 0 (workshop dates) | 2/25 | Correct date, but gives an end time the knowledge base doesn't list |
| 7 (recommendations) | 1/25 | Two suggestions without their date |

Four of these 17 turns are KB-grounding errors (turns 0 and 4); the standard row had none.

## Method and limits

The policy was developed on this benchmark's 30-turn script, with no held-out scenario. The
rules are written to be generic, but they were only measured on this task, so the result
doesn't show how the prompt performs on unseen conversations.

Collection followed the standard Nova Sonic protocol: the first 25 complete conversations in attempt
order, at most three concurrent sessions, a 30-minute timeout per attempt, no recovery, and no
replacement by score. One of 26 attempts ended at turn 25 on a Bedrock stream error ("Timed out
waiting for audio bytes or interactive content"); it is counted as a failed conversation.

Latency and turn-taking come from the stereo recordings with the unchanged analyzer. In the
three-landmark input waveform audit (correlation ≥ .95, offset ≤ 250ms, drift ≤ 100ms), 748 of
750 input clips pass. Two clips in two conversations are flagged: one landmark correlation of
0.68 (conversation 3, turn 17) and a 136ms within-turn drift (conversation 16, turn 12). Both
conversations are retained, not replaced. Quality grading does not depend on input alignment,
and excluding the two conversations leaves latency unchanged (non-tool median 1088ms, maximum
1984ms, tool mean 1413ms; 690 samples).

The SYSTEM-prompt rejections in the [filter investigation](../nova-2-5-sonic-20261019/filter-diagnostics/REPORT.md)
come from a service-level safety filter that Nova Sonic applies before a prompt reaches the model,
not from model behavior. Rechecked with the same probe method, the filter gives identical results
for `amazon.nova-2-5-sonic` and `amazon.nova-2-sonic-v1:0`: a short control prompt is accepted,
while the full standard SYSTEM prompt, a short prompt with the session title "Voice Cloning and
Synthesis: Ethics and Technology", and a short prompt with the sponsor name `Replicate` are each
blocked. The README therefore describes it as a Nova Sonic safeguard rather than one specific to
Nova 2.5 Sonic.

As a cross-check, the standard speech-v2 prompt was rerun on the same account and harness:
650/750 (86.7%, 95% CI 83.5–89.6%), consistent with the published 645/750.
Two of those 25 captures fail the input waveform audit, so that rerun supplies no latency figures.

Successful Opus judge requests cost $4.23 for this cohort.

For a single reproduction with AWS credentials configured:

```bash
MTE_NOVA_KB_CONTEXT_ROLE=user MTE_ENABLE_RECOVERY=0 MTE_DEDUPE_TOOL_CALLS=1 MTE_TOOL_RESULT_RUN_LLM=0 \
  uv run multi-turn-eval run aiwf_medium_context_speech_nova_sonic_optimized --model amazon.nova-2-5-sonic --pipeline nova-sonic
```

[Aggregate](summary.json) · [Configuration](../../../ops/nova-2-5-sonic-optimized-20261021/configuration.json) ·
[Standard Nova Sonic report](../nova-2-5-sonic-20261019/user-context-v2/REPORT.md)