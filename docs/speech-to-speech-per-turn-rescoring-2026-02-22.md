# Speech-to-Speech Per-Turn Rescoring

Date: 2026-02-22

## Background

Commit `3c19c13` switched scoring from per-dimension averaging to strict per-turn pass rate. A turn passes only if all three judged dimensions pass on the same turn (`tool_use_correct && instruction_following && kb_grounding`).

Previously, Pass Rate was the average of the three per-dimension rates. Now it is `turn_pass / total_turns`.

## Runs Used

All data is from the 10-run aggregate documented in `docs/benchmark-results-10-run-aggregate-2026-01-11.md`. Per-turn scores were recomputed from `claude_judged.jsonl` files (not re-judged). Nova-2-sonic is excluded because the runs backing its README numbers are not documented.

### Ultravox (10 runs)
```
runs/aiwf_medium_context/20260111T121854_ultravox-v0.7_e9bd95f9
runs/aiwf_medium_context/20260111T130334_ultravox-v0.7_5ee56857
runs/aiwf_medium_context/20260111T132214_ultravox-v0.7_aa587ca7
runs/aiwf_medium_context/20260111T134127_ultravox-v0.7_2c4a638b
runs/aiwf_medium_context/20260111T140139_ultravox-v0.7_05dd5a81
runs/aiwf_medium_context/20260111T141935_ultravox-v0.7_62e4e922
runs/aiwf_medium_context/20260111T143552_ultravox-v0.7_0d39af7f
runs/aiwf_medium_context/20260111T145551_ultravox-v0.7_93d58d81
runs/aiwf_medium_context/20260111T151424_ultravox-v0.7_72b453bd
runs/aiwf_medium_context/20260111T153608_ultravox-v0.7_f54f3bbe
```

### GPT-Realtime (10 runs)
```
runs/aiwf_medium_context/20260111T121855_gpt-realtime_b37aa3e9
runs/aiwf_medium_context/20260111T130336_gpt-realtime_6dfd3b88
runs/aiwf_medium_context/20260111T131525_gpt-realtime_5c6961e4
runs/aiwf_medium_context/20260111T132703_gpt-realtime_cd62546b
runs/aiwf_medium_context/20260111T133842_gpt-realtime_b9345da2
runs/aiwf_medium_context/20260111T135140_gpt-realtime_12e93c10
runs/aiwf_medium_context/20260111T140337_gpt-realtime_f0e6512b
runs/aiwf_medium_context/20260111T141509_gpt-realtime_34a79d73
runs/aiwf_medium_context/20260111T142724_gpt-realtime_684668f4
runs/aiwf_medium_context/20260111T144000_gpt-realtime_04c5d708
```

### Grok-Realtime (10 runs)
```
runs/aiwf_medium_context/20260111T153513_grok-realtime_7c834901
runs/aiwf_medium_context/20260111T154346_grok-realtime_636853b1
runs/aiwf_medium_context/20260111T154607_grok-realtime_e501d7d7
runs/aiwf_medium_context/20260111T155505_grok-realtime_c337a515
runs/aiwf_medium_context/20260111T155713_grok-realtime_d0dd9189
runs/aiwf_medium_context/20260111T160657_grok-realtime_f9c4697e
runs/aiwf_medium_context/20260111T160831_grok-realtime_bb82f5d9
runs/aiwf_medium_context/20260111T161620_grok-realtime_4d06b14a
runs/aiwf_medium_context/20260111T161951_grok-realtime_5356c69a
runs/aiwf_medium_context/20260111T163051_grok-realtime_37a1d098
```

### Gemini-Live (10 runs)
```
runs/aiwf_medium_context/20260111T121856_gemini-2.5-flash-native-audio-preview-12-2025_e14d6d6c
runs/aiwf_medium_context/20260111T130340_gemini-2.5-flash-native-audio-preview-12-2025_ea1adba3
runs/aiwf_medium_context/20260111T131324_gemini-2.5-flash-native-audio-preview-12-2025_452bec6f
runs/aiwf_medium_context/20260111T132301_gemini-2.5-flash-native-audio-preview-12-2025_ed390791
runs/aiwf_medium_context/20260111T133223_gemini-2.5-flash-native-audio-preview-12-2025_ba14f74a
runs/aiwf_medium_context/20260111T134202_gemini-2.5-flash-native-audio-preview-12-2025_52d3e640
runs/aiwf_medium_context/20260111T135220_gemini-2.5-flash-native-audio-preview-12-2025_041318b2
runs/aiwf_medium_context/20260111T140133_gemini-2.5-flash-native-audio-preview-12-2025_41afeb20
runs/aiwf_medium_context/20260111T141316_gemini-2.5-flash-native-audio-preview-12-2025_f19d0358
runs/aiwf_medium_context/20260111T142259_gemini-2.5-flash-native-audio-preview-12-2025_e18e168a
```

## New Table (strict per-turn pass rate)

| Model | Tool Use | Instruction | KB Ground | Turn Ok | Pass Rate | Non-Tool V2V Med | Non-Tool V2V Max | Tool V2V Mean | Silence Pad Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ultravox-v0.7 | 293/300 | 293/300 | 297/300 | 300/300 | 96.3% | 864ms | 1888ms | 2406ms | 82ms |
| gpt-realtime | 270/300 | 261/300 | 300/300 | 296/300 | 87.0% | 1536ms | 4672ms | 2199ms | 341ms |
| grok-realtime | 267/300 | 275/300 | 295/300 | * | 86.3% | 1184ms | 2016ms | 1472ms | 478ms |
| gemini-live | 258/300 | 258/300 | 292/300 | 278/300 | 82.7% | 2624ms | 30000ms | 4082ms | 90ms |

## Comparison to Old Table (per-dimension averaging)

| Model | Old Pass Rate | New Pass Rate | Old Rank | New Rank | Delta |
|---|---:|---:|---:|---:|---:|
| ultravox-v0.7 | 97.7% | 96.3% | 1 | 1 | -1.4pp |
| gpt-realtime | 86.7% | 87.0% | 2 | 2 | +0.3pp |
| grok-realtime | 86.0% | 86.3% | 4 | 3 | +0.3pp |
| gemini-live | 86.0% | 82.7% | 3 | 4 | -3.3pp |

Rankings are unchanged for ultravox (1st) and gpt-realtime (2nd). Grok and gemini-live swap 3rd/4th. Under per-dimension averaging they were tied at 86.0%, but strict per-turn scoring reveals gemini-live drops to 82.7% (its failures cluster on the same turns) while grok holds at 86.3%.

Per-dimension columns also shifted slightly vs the old table (e.g. ultravox instruction 294->293, KB 298->297). The numbers above are recomputed directly from `claude_judged.jsonl` files.

## Notes

- V2V timing columns are carried over from the old table unchanged.
- Grok-realtime Turn Ok is marked `*` because the API was not production-ready and the metric was not reported.
- Nova-2-sonic is excluded. The aggregate JSON (`benchmark-results-10-run-aggregate-nova-sonic-2026-01-12.json`) does not match the numbers currently in the README (249/170/249 vs 278/265/296). The specific runs backing the README numbers are unknown.
