# Clever Chatter exact-boundary cohort

Date: 2026-08-05

## Reliability

| Metric | Result |
|---|---:|
| Attempts | 20 |
| Complete 30-turn conversations | 15/20 |
| Completion rate (Wilson 95% CI) | 75.0% (53.1–88.8%) |
| Complete | 15 |
| No Audio Timeout | 5 |

Failed zero-based turns: 0 (1), 4 (2), 10 (1), 12 (1).

## Timing

| Metric | P50 | P90 | P95 | Max | N |
|---|---:|---:|---:|---:|---:|
| First audio | 1107ms | 1508ms | 1785ms | 14504ms | 480 |
| Response completion | 14844ms | 29551ms | 37837ms | 151426ms | 480 |

## Event-flow checks

- Balanced explicit activity boundaries: 20/20 runs.
- Activity starts/ends: 485/485.
- Model audio events during explicit input: 0.
- Harmless terminal/interruption control events during input: 5/5.
- Non-empty transcript turns missing first-audio timing: 0.
- Provider statuses: `{"IN_PROGRESS": 92, "REQUIRES_ACTION": 491}`.
- Runs containing at least one `IN_PROGRESS`: 16/20.
- Tool calls: `{"end_session": 15, "request_tech_support": 8, "submit_dietary_request": 3, "submit_session_suggestion": 31, "vote_for_session": 4}`.

### Tool calls in complete conversations

These are raw call counts; the content judge separately checks timing and arguments.

| Tool | Observed | Expected |
|---|---:|---:|
| `submit_session_suggestion` | 30 | 30 |
| `submit_dietary_request` | 3 | 15 |
| `request_tech_support` | 8 | 15 |
| `vote_for_session` | 4 | 15 |
| `end_session` | 15 | 15 |

## Judging

Judged 15 complete runs (450 turns).

| Metric | Result |
|---|---:|
| Strict pass | 88.9% (87.6–90.2%) |
| Tool use | 415/450 |
| Instruction | 406/450 |
| KB grounding | 448/450 |
| Legacy offline turn-tag match (diagnostic only) | 273/450 |

The legacy offline detector assumes one contiguous bot-audio segment per turn. Clever Chatter often speaks in multiple asynchronous phases, so missing tag matches are not evidence of actual overlap; the raw boundary audit above is authoritative for whether model audio crossed explicit user input.
