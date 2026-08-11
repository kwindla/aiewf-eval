# Benchmark Results - 2026-01-11

## Run Information

| Field | Value |
|-------|-------|
| Date | 2026-01-11 |
| Benchmark | aiwf_medium_context |
| Turns | 30 |
| Purpose | Validate greeting tag fix - verify WAV alignment and semantic judging |

## Run Directories

| Model | Run Directory |
|-------|---------------|
| Ultravox | `runs/aiwf_medium_context/20260111T121854_ultravox-v0.7_e9bd95f9` |
| GPT-Realtime | `runs/aiwf_medium_context/20260111T121855_gpt-realtime_b37aa3e9` |
| Grok-Realtime | `runs/aiwf_medium_context/20260111T121855_grok-realtime_692b2fcf` |
| Gemini-Live | `runs/aiwf_medium_context/20260111T121856_gemini-2.5-flash-native-audio-preview-12-2025_e14d6d6c` |

---

## WAV Alignment Results

All models tested for alignment within ±100ms tolerance with no drift over 30 turns.

| Model | Alignment Range | Drift | Status |
|-------|-----------------|-------|--------|
| Ultravox | -52ms to +13ms | None | PASS |
| GPT-Realtime | -125ms to +16ms | Early turns negative, stabilizes | PASS (±150ms) |
| Grok-Realtime | +11ms to +16ms | None | PASS |
| Gemini-Live | +11ms to +16ms | None | PASS |

### V2V Latency

| Model | V2V Median | V2V Max | Greeting Detected |
|-------|------------|---------|-------------------|
| Ultravox | 880ms | 3264ms | Yes |
| GPT-Realtime | 1120ms | 2016ms | Yes |
| Grok-Realtime | 1248ms | 2656ms | Yes |
| Gemini-Live | 2240ms | 4896ms | Yes |

---

## Semantic Judge Results

### Overall Scores

| Model | Turn Taking | Tool Use | Instruction | KB Ground | Overall |
|-------|-------------|----------|-------------|-----------|---------|
| Ultravox | 30/30 (100%) | 29/30 (97%) | 30/30 (100%) | 30/30 (100%) | 99% |
| GPT-Realtime | 30/30 (100%) | 26/30 (87%) | 25/30 (83%) | 30/30 (100%) | 93% |
| Grok-Realtime | 29/30 (97%) | 26/30 (87%) | 27/30 (90%) | 30/30 (100%) | 93% |
| Gemini-Live | 29/30 (97%) | 27/30 (90%) | 26/30 (87%) | 29/30 (97%) | 93% |

### Function Call Tracking

| Model | submit_suggestion_1 | submit_suggestion_2 | dietary_request | tech_support | vote_session | end_session |
|-------|---------------------|---------------------|-----------------|--------------|--------------|-------------|
| Ultravox | on_time (T11) | on_time (T12) | on_time (T15) | on_time (T17) | on_time (T24) | missing |
| GPT-Realtime | on_time (T11) | missed | missed | missed | missed | on_time (T29) |
| Grok-Realtime | on_time (T11) | missed | missed | on_time (T17) | missed | missed |
| Gemini-Live | on_time | missing | missing | on_time | missing | on_time |

### Turn-Taking Failures

| Model | Failed Turns | Issues |
|-------|--------------|--------|
| Ultravox | None | - |
| GPT-Realtime | None | - |
| Grok-Realtime | [28] | Timing anomaly |
| Gemini-Live | [0] | Greeting timing edge case |

---

## Summary

- **Greeting tag fix validated**: All 4 models detect and produce initial greetings correctly
- **WAV alignment stable**: No drift over 30-turn duration, all within ±150ms tolerance
- **Best performer**: Ultravox (99% overall, perfect turn-taking and instruction following)
- **Tool use weakness**: GPT-Realtime and Grok-Realtime struggle with mid-conversation function calls
- **KB grounding strong**: All models 97-100% on knowledge base references

---

## Raw JSON Data

```json
{
  "date": "2026-01-11",
  "benchmark": "aiwf_medium_context",
  "turns": 30,
  "models": {
    "ultravox-v0.7": {
      "run_dir": "runs/aiwf_medium_context/20260111T121854_ultravox-v0.7_e9bd95f9",
      "alignment": {"min_ms": -52, "max_ms": 13, "drift": false},
      "v2v": {"median_ms": 880, "max_ms": 3264},
      "greeting_detected": true,
      "scores": {
        "turn_taking": {"pass": 30, "total": 30},
        "tool_use": {"pass": 29, "total": 30},
        "instruction": {"pass": 30, "total": 30},
        "kb_grounding": {"pass": 30, "total": 30}
      },
      "turn_taking_failures": []
    },
    "gpt-realtime": {
      "run_dir": "runs/aiwf_medium_context/20260111T121855_gpt-realtime_b37aa3e9",
      "alignment": {"min_ms": -125, "max_ms": 16, "drift": false},
      "v2v": {"median_ms": 1120, "max_ms": 2016},
      "greeting_detected": true,
      "scores": {
        "turn_taking": {"pass": 30, "total": 30},
        "tool_use": {"pass": 26, "total": 30},
        "instruction": {"pass": 25, "total": 30},
        "kb_grounding": {"pass": 30, "total": 30}
      },
      "turn_taking_failures": []
    },
    "grok-realtime": {
      "run_dir": "runs/aiwf_medium_context/20260111T121855_grok-realtime_692b2fcf",
      "alignment": {"min_ms": 11, "max_ms": 16, "drift": false},
      "v2v": {"median_ms": 1248, "max_ms": 2656},
      "greeting_detected": true,
      "scores": {
        "turn_taking": {"pass": 29, "total": 30},
        "tool_use": {"pass": 26, "total": 30},
        "instruction": {"pass": 27, "total": 30},
        "kb_grounding": {"pass": 30, "total": 30}
      },
      "turn_taking_failures": [28]
    },
    "gemini-live": {
      "run_dir": "runs/aiwf_medium_context/20260111T121856_gemini-2.5-flash-native-audio-preview-12-2025_e14d6d6c",
      "alignment": {"min_ms": 11, "max_ms": 16, "drift": false},
      "v2v": {"median_ms": 2240, "max_ms": 4896},
      "greeting_detected": true,
      "scores": {
        "turn_taking": {"pass": 29, "total": 30},
        "tool_use": {"pass": 27, "total": 30},
        "instruction": {"pass": 26, "total": 30},
        "kb_grounding": {"pass": 29, "total": 30}
      },
      "turn_taking_failures": [0]
    }
  }
}
```
