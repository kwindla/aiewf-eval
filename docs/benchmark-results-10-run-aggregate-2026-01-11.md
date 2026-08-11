# 10-Run Aggregate Benchmark Results - 2026-01-11

## Run Information

| Field | Value |
|-------|-------|
| Date | 2026-01-11 |
| Benchmark | aiwf_medium_context |
| Turns per run | 30 |
| Runs per model | 10 |
| Total turns evaluated | 300 per model (1,200 total) |
| Purpose | Establish baseline performance metrics across multiple runs |

---

## Aggregate Results Table

```
-----------------------------------------------------------------------------------------------------------------------------------------
| Model             | Tool    | Instruction | KB       | Turn    | Pass     | Non-Tool V2V  | Non-Tool V2V  | Tool V2V   | Silence Pad  |
|                   | Use     |             | Ground   | Ok      | Rate     | Med           | Max           | Mean       | Mean         |
-----------------------------------------------------------------------------------------------------------------------------------------
| gemini-live       | 258/300 | 261/300     | 293/300  | 278/300 |  86.0% | 2624ms        | 61747ms       | 4082ms     | 90ms         |
-----------------------------------------------------------------------------------------------------------------------------------------
| gpt-realtime      | 271/300 | 260/300     | 300/300  | 296/300 |  86.7% | 1536ms        | 4672ms        | 2199ms     | 341ms        |
-----------------------------------------------------------------------------------------------------------------------------------------
| grok-realtime     | 267/300 | 275/300     | 295/300  | 279/300 |  89.0% | 1184ms        | 2016ms        | 1472ms     | 478ms        |
-----------------------------------------------------------------------------------------------------------------------------------------
| ultravox-v0.7     | 293/300 | 294/300     | 298/300  | 300/300 |  97.7% | 864ms         | 1888ms        | 2406ms     | 82ms         |
-----------------------------------------------------------------------------------------------------------------------------------------
```

---

## Metric Definitions

| Metric | Description |
|--------|-------------|
| **Tool Use** | Correct tool/function calls out of total turns |
| **Instruction** | Instruction following accuracy |
| **KB Ground** | Knowledge base grounding (answers using provided context) |
| **Turn Ok** | Turn-taking success (no timing anomalies) |
| **Pass Rate** | Percentage of turns passing all quality metrics |
| **Non-Tool V2V Med** | Median voice-to-voice latency for non-tool-calling turns |
| **Non-Tool V2V Max** | Maximum voice-to-voice latency for non-tool-calling turns |
| **Tool V2V Mean** | Mean voice-to-voice latency for tool-calling turns |
| **Silence Pad Mean** | Mean silent audio padding before speech begins |

---

## Performance Summary

### Rankings by Category

| Category | 1st | 2nd | 3rd | 4th |
|----------|-----|-----|-----|-----|
| **Overall Pass Rate** | Ultravox (97.7%) | Grok (89.0%) | GPT (86.7%) | Gemini (86.0%) |
| **Tool Use** | Ultravox (97.7%) | GPT (90.3%) | Grok (89.0%) | Gemini (86.0%) |
| **Instruction Following** | Ultravox (98.0%) | Grok (91.7%) | Gemini (87.0%) | GPT (86.7%) |
| **KB Grounding** | GPT (100%) | Ultravox (99.3%) | Grok (98.3%) | Gemini (97.7%) |
| **Turn-Taking** | Ultravox (100%) | GPT (98.7%) | Grok (93.0%) | Gemini (92.7%) |
| **Fastest V2V (median)** | Ultravox (864ms) | Grok (1184ms) | GPT (1536ms) | Gemini (2624ms) |

### Key Observations

#### 1. Ultravox is the clear leader
- **97.7% overall pass rate** - significantly ahead of others
- **100% turn-taking success** - perfect timing across all 300 turns
- **Fastest median V2V latency** at 864ms
- **Low silence padding** (82ms) - minimal delay before speech

#### 2. Grok-Realtime shows strong instruction following
- **91.7% instruction following** - second best after Ultravox
- **Fastest non-tool V2V** among OpenAI-compatible models (1184ms median)
- **Highest silence padding** (478ms) - sends more pre-speech audio
- **Turn-taking issues** (93.0%) - some mid-conversation timing anomalies

#### 3. GPT-Realtime has perfect KB grounding
- **100% KB grounding** - always uses provided context correctly
- **Consistent latency** (max 4672ms vs Gemini's 61747ms)
- **Tool use weakness** (90.3%) - misses some mid-conversation function calls
- **Moderate silence padding** (341ms)

#### 4. Gemini-Live shows high latency variance
- **Extreme max latency** (61747ms) - session instability causing outliers
- **Good KB grounding** (97.7%) despite instability
- **Lowest tool use** (86.0%) - struggles with function calling
- **Low silence padding** (90ms) - quick speech onset when stable

---

## Latency Analysis

### Voice-to-Voice Latency Distribution

| Model | Non-Tool Median | Non-Tool Max | Tool Mean | Silence Pad |
|-------|-----------------|--------------|-----------|-------------|
| Ultravox | 864ms | 1888ms | 2406ms | 82ms |
| Grok-Realtime | 1184ms | 2016ms | 1472ms | 478ms |
| GPT-Realtime | 1536ms | 4672ms | 2199ms | 341ms |
| Gemini-Live | 2624ms | 61747ms | 4082ms | 90ms |

### Observations

1. **Ultravox fastest overall** but has higher tool V2V (2406ms) - likely due to local TTS processing after tool results

2. **Grok has lowest tool V2V** (1472ms) - efficient tool result handling

3. **Gemini's 61747ms max** indicates session reconnection issues - when stable, median is ~2624ms

4. **Silence padding varies significantly**:
   - Ultravox/Gemini: 82-90ms (minimal)
   - GPT: 341ms (moderate)
   - Grok: 478ms (highest) - sends more "thinking" audio

---

## Turn-Taking Analysis

### Success Rates

| Model | Turn-Taking Success | Issues |
|-------|---------------------|--------|
| Ultravox | 300/300 (100%) | None |
| GPT-Realtime | 296/300 (98.7%) | Minor alignment drift |
| Grok-Realtime | 279/300 (93.0%) | VAD timing issues mid-conversation |
| Gemini-Live | 278/300 (92.7%) | Session timeouts, reconnections |

### Common Issues by Model

**Grok-Realtime:**
- Server-side VAD sometimes fails to detect user speech end
- Occasional unprompted responses
- Hit xAI's 30-minute session limit in one run

**Gemini-Live:**
- Session disconnections requiring reconnection (up to 9 retries observed)
- High latency on reconnection turns
- Greeting timing edge cases on turn 0

---

## Tool Use Analysis

### Function Call Success

| Model | Success Rate | Common Issues |
|-------|--------------|---------------|
| Ultravox | 293/300 (97.7%) | Occasional end_session miss |
| GPT-Realtime | 271/300 (90.3%) | Misses mid-conversation functions |
| Grok-Realtime | 267/300 (89.0%) | Similar to GPT pattern |
| Gemini-Live | 258/300 (86.0%) | Multiple function misses |

### Observations

- All models successfully handle first function call
- Mid-conversation functions (dietary_request, vote_session) frequently missed
- Session-ending functions more reliable than mid-session ones

---

## Known Issues Encountered

### 1. Grok 30-Minute Session Limit
- xAI's realtime API has a 30-minute session limit
- One run got stuck in error loop after hitting limit
- Pipeline continued receiving error frames, preventing inactivity timeout
- **Recommendation**: Add websocket error detection to pipeline termination logic

### 2. Gemini Session Instability
- Sessions occasionally disconnect mid-conversation
- Reconnection logic works but causes high latency spikes
- Max V2V of 61747ms (>1 minute) during reconnection attempts

### 3. Turn-Taking Detection Gaps
- Grok's server-side VAD sometimes disagrees with our VAD
- Causes missing_timing_data or negative_ttfb issues
- Not related to our pipeline - appears to be model behavior

---

## Run Directories

### Ultravox (10 runs)
```
runs/aiwf_medium_context/20260111T153608_ultravox-v0.7_f54f3bbe
runs/aiwf_medium_context/20260111T151424_ultravox-v0.7_72b453bd
runs/aiwf_medium_context/20260111T145551_ultravox-v0.7_93d58d81
runs/aiwf_medium_context/20260111T143552_ultravox-v0.7_0d39af7f
runs/aiwf_medium_context/20260111T141935_ultravox-v0.7_62e4e922
runs/aiwf_medium_context/20260111T140139_ultravox-v0.7_05dd5a81
runs/aiwf_medium_context/20260111T134127_ultravox-v0.7_2c4a638b
runs/aiwf_medium_context/20260111T132214_ultravox-v0.7_aa587ca7
runs/aiwf_medium_context/20260111T130334_ultravox-v0.7_5ee56857
runs/aiwf_medium_context/20260111T121854_ultravox-v0.7_e9bd95f9
```

### GPT-Realtime (10 runs)
```
runs/aiwf_medium_context/20260111T144000_gpt-realtime_04c5d708
runs/aiwf_medium_context/20260111T142724_gpt-realtime_684668f4
runs/aiwf_medium_context/20260111T141509_gpt-realtime_34a79d73
runs/aiwf_medium_context/20260111T140337_gpt-realtime_f0e6512b
runs/aiwf_medium_context/20260111T135140_gpt-realtime_12e93c10
runs/aiwf_medium_context/20260111T133842_gpt-realtime_b9345da2
runs/aiwf_medium_context/20260111T132703_gpt-realtime_cd62546b
runs/aiwf_medium_context/20260111T131525_gpt-realtime_5c6961e4
runs/aiwf_medium_context/20260111T130336_gpt-realtime_6dfd3b88
runs/aiwf_medium_context/20260111T121855_gpt-realtime_b37aa3e9
```

### Grok-Realtime (10 runs)
```
runs/aiwf_medium_context/20260111T163051_grok-realtime_37a1d098
runs/aiwf_medium_context/20260111T161951_grok-realtime_5356c69a
runs/aiwf_medium_context/20260111T161620_grok-realtime_4d06b14a
runs/aiwf_medium_context/20260111T160831_grok-realtime_bb82f5d9
runs/aiwf_medium_context/20260111T160657_grok-realtime_f9c4697e
runs/aiwf_medium_context/20260111T155713_grok-realtime_d0dd9189
runs/aiwf_medium_context/20260111T155505_grok-realtime_c337a515
runs/aiwf_medium_context/20260111T154607_grok-realtime_e501d7d7
runs/aiwf_medium_context/20260111T154346_grok-realtime_636853b1
runs/aiwf_medium_context/20260111T153513_grok-realtime_7c834901
```

### Gemini-Live (10 runs)
```
runs/aiwf_medium_context/20260111T142259_gemini-2.5-flash-native-audio-preview-12-2025_e18e168a
runs/aiwf_medium_context/20260111T141316_gemini-2.5-flash-native-audio-preview-12-2025_f19d0358
runs/aiwf_medium_context/20260111T140133_gemini-2.5-flash-native-audio-preview-12-2025_41afeb20
runs/aiwf_medium_context/20260111T135220_gemini-2.5-flash-native-audio-preview-12-2025_041318b2
runs/aiwf_medium_context/20260111T134202_gemini-2.5-flash-native-audio-preview-12-2025_52d3e640
runs/aiwf_medium_context/20260111T133223_gemini-2.5-flash-native-audio-preview-12-2025_ba14f74a
runs/aiwf_medium_context/20260111T132301_gemini-2.5-flash-native-audio-preview-12-2025_ed390791
runs/aiwf_medium_context/20260111T131324_gemini-2.5-flash-native-audio-preview-12-2025_452bec6f
runs/aiwf_medium_context/20260111T130340_gemini-2.5-flash-native-audio-preview-12-2025_ea1adba3
runs/aiwf_medium_context/20260111T121856_gemini-2.5-flash-native-audio-preview-12-2025_e14d6d6c
```

---

## Conclusions

1. **Ultravox is the top performer** with 97.7% pass rate, 100% turn-taking, and fastest latency. It's the most reliable choice for production voice applications.

2. **Grok-Realtime is a strong contender** with 89% pass rate and excellent instruction following (91.7%). Watch for the 30-minute session limit.

3. **GPT-Realtime offers perfect KB grounding** (100%) but has weaker tool use (90.3%). Good for knowledge-heavy applications.

4. **Gemini-Live needs session stability work** - the 61747ms max latency and 86% pass rate suggest production readiness concerns.

5. **All models handle greetings correctly** after the greeting tag fix validation earlier today.

---

## Recommendations

### For Production Use
- **Primary**: Ultravox v0.7 for best overall quality
- **Alternative**: GPT-Realtime for guaranteed KB grounding

### For Development/Testing
- Grok-Realtime provides good performance at potentially lower cost
- Gemini-Live useful for testing edge cases (reconnection handling)

### Pipeline Improvements Needed
1. Add websocket error detection to prevent stuck runs
2. Implement session keepalive for Grok (pre-30-minute refresh)
3. Improve Gemini reconnection handling to reduce latency spikes
