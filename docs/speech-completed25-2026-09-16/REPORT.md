# Speech benchmark: 25 completed conversations per cohort

Status: complete. Original attempts and old fixed-denominator reports are preserved.

Accuracy below is conditional on completed conversations. Failed conversations—including all their partial turns—are excluded from accuracy and counted separately. Harness-invalid captures are separately excluded, never called provider failures. Stop at the first 25 valid completions without selecting on answer quality.

| Model / cohort | Scored conversations | Accuracy | Failed conversations | Harness excluded | First-speech mean / median | Positive latency coverage |
|---|---:|---:|---:|---:|---:|---:|
| GPT-Realtime-2.1 low | 25 | 708/750 (94.4%) | 0/25 (0.0%) | 0 | 1615 / 1536 ms | 750/750 |
| Grok Voice Think Fast | 25 | 733/750 (97.7%) | 0/25 (0.0%) | 0 | 2655 / 2528 ms | 730/750 |
| GPT-Realtime-1.5 | 25 | 705/750 (94.0%) | 0/25 (0.0%) | 0 | 1279 / 1152 ms | 725/750 |
| Gemini 3.1 minimal | 25 | 656/750 (87.5%) | 1/26 (3.8%) | 0 | 1447 / 1344 ms | 721/750 |
| Gemini 3.8 low | 25 | 697/750 (92.9%) | 0/25 (0.0%) | 1 | 1492 / 1440 ms | 743/750 |
| Gemini 3.8 medium (earlier harness) | 25 | 698/750 (93.1%) | 0/25 (0.0%) | 0 | 1499 / 1472 ms | 743/750 |
| GPT-Live standard | 25 | 707/750 (94.3%) | 0/25 (0.0%) | 0 | 1870 / 1728 ms | 748/750 |
| Gemini 3.8 medium (interaction-aware) | 25 | 698/750 (93.1%) | 2/27 (7.4%) | 0 | 1752 / 1632 ms | 750/750 |
| GPT-Live prompt-optimized | 25 | 739/750 (98.5%) | 2/27 (7.4%) | 0 | 2686 / 2112 ms | 748/750 |

## Failed conversations

Turn numbers below are **one-based**, out of 30; raw zero-based indices are retained in aggregate.json. A no-response timeout is not labeled an explicit API error.

- **Gemini 3.1 minimal**: no_audio_timeout on turn **25**, after 24 completed turns. Run `runs/aiwf_medium_context_speech_v2/20260915T144826_gemini-3.1-flash-live-preview_8a11ff4e`.
- **Gemini 3.8 medium (interaction-aware)**: no_audio_timeout on turn **1**, after 0 completed turns. Run `runs/aiwf_medium_context_speech_v2/20260916T074653_gemini-3.8-live-extended-thinking_c2d7a24b`.
- **Gemini 3.8 medium (interaction-aware)**: no_audio_timeout on turn **1**, after 0 completed turns. Run `runs/aiwf_medium_context_speech_v2/20260916T080944_gemini-3.8-live-extended-thinking_bd55a0b2`.
- **GPT-Live prompt-optimized**: TimeoutError on turn **11**, after 10 completed turns. Run `runs/aiwf_medium_context_speech_live_optimized/20260916T123039_gpt-live-1_5dc5eeb2`.
- **GPT-Live prompt-optimized**: TimeoutError on turn **11**, after 10 completed turns. Run `runs/aiwf_medium_context_speech_live_optimized/20260916T140645_gpt-live-1_fd2c66d8`.

## Measurement and harness notes

- Gemini 3.8 low: original attempt 24 replaced because a startup greeting occupied turn zero. The old capture is preserved as a harness exclusion. Replacement uses a startup guard; model settings and prompt are unchanged.
- GPT-Live standard: one recording has shifted input windows; matching three waveform landmarks per turn recovers input locations without changing any measured V2V latency. See standard-live-alignment-review.json.
- GPT-Live optimized: one short recorded answer was missed by canonical VAD. Keep its latency missing and the completed conversation eligible; see optimized-live-vad-review.json.
- First-speech metrics include interim acknowledgments. Negative and missing observations are retained in coverage, not silently converted to positive latencies. Substantive-answer latency is not established.
- Earlier and interaction-aware Gemini medium cohorts remain separate; they used different harness protocols. Failed diagnostic cohorts and smoke tests are not pooled into current reliability rates.
- Quality uses the same shared Opus 5 rubric and prompts as before. No score-driven recapture or rejudging. Existing empty closing-turn audio limitations remain disclosed in original reports.

