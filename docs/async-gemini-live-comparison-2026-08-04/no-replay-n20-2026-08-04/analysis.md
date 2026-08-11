# Clever Chatter no-replay campaign

Date: 2026-08-04

## Result

- Attempts: 20
- Full 30-turn completions: 0
- No-audio timeouts: 19
- Model-ended-early runs: 1
- Partial turns recorded before termination: 88
- No complete runs were available to judge.

The observed completion rate was 0/20. The two-sided 95% Wilson interval is 0.0% to 16.1%.

## Timeout boundary audit

The trace hooks the private SDK immediately after `WebSocket.recv()` and before JSON-to-SDK conversion. Across the 19 timeout runs:

- 19/19 had no raw server event after the final user-audio send completed.
- 19/19 had no raw server event after the last local `UserStoppedSpeaking` signal.
- Raw events captured across all attempts: 7,881.
- Raw receive sequence gaps: 0.
- Raw JSON decode errors: 0.
- Raw socket-close events: 0.
- Server error events: 0.
- GoAway events: 0.
- No unrecognized model-part shape was observed; the raw trace inventory is in `analysis.json`.

Because the receive loop continuously calls the SDK's `session.receive()` and the trace executes before conversion, these data rule out a server message arriving during any observed 15-second wait and then being dropped by the SDK or wrapper. They cannot prove that the server would not have responded just after the watchdog cancelled the run, nor can any client-side logging prove what was never delivered over the socket.

## Other validation

- Runtime records with either a non-timeout result or `replayed=false`: 20/20.
- Replay log lines: 0.
- Runtime warning lines: 0.
- Recordings: 20/20; all stereo: true; all 24 kHz: true.

Timeout zero-based turn counts: 0: 2, 1: 5, 4: 1, 5: 1, 6: 5, 7: 2, 8: 3.

The timeout turn median was 6 (zero-based), with range 0-8.

## Outbound API-boundary validation

A separate diagnostic attempt—not included in the 20-run benchmark sample—traced successful `send_realtime_input()` returns as well as raw receives. It timed out at zero-based turn 5.

- The complete real WAV produced 278 successful SDK sends totaling 266,880 bytes.
- After the WAV ended, 998 further continuous-audio sends totaling 957,942 bytes succeeded before the timeout.
- Maximum SDK send wait: 0.436 ms; send errors: 0; send cancellations: 0.
- Raw receives after the WAV ended: 0.
- The outstanding raw receive was explicitly cancelled after waiting 26,038.1 ms.
- Last successful send: `2026-08-04 21:03:44.080`; timeout: `2026-08-04 21:03:44.081`.

Run: `runs/aiwf_medium_context/20260804T210138_clever-chatter_0b5a6d58`.

## Rebuild

```bash
./.venv/bin/python docs/clever-chatter-comparison-2026-08-04/no-replay-n20-2026-08-04/analyze_campaign.py
```
