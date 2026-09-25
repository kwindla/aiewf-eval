# Gemini 3.8 audio association audit

The legacy analyzer assigns turn indices by bot audio-marker order. A marker is not guaranteed for every scripted turn, so a missing marker can shift subsequent tool/non-tool labels and omit an otherwise measurable response. It also stops emitting rows at the number of detected markers.

This independent analysis assigns actual recorded speech to logged input windows and verifies those windows by normalized waveform correlation at three speech landmarks per user utterance. It uses the same canonical Silero detector and thresholds. Source recordings, original metrics, model prompts, and semantic judgments remain untouched.

| Setting | Positive V2V observations | Overall mean | Overall median | Overall p90 | Non-tool median | Tool mean | Missing planned turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemini38low | 742/750 | 1496ms | 1440ms | 1984ms | 1440ms | 1504ms | 8 |
| gemini38medium | 743/750 | 1499ms | 1472ms | 1920ms | 1472ms | 1554ms | 7 |

## Coverage and limitations

- **gemini38low:** audit failures none; boundary/detection issues {'missing bot speech': 7}; nonpositive measurements 0. Missing bot speech can be a silent tool-only turn, not necessarily an API error; inspect the corresponding transcript before classifying it.
- **gemini38medium:** audit failures none; boundary/detection issues {'missing bot speech': 7}; nonpositive measurements 0. Missing bot speech can be a silent tool-only turn, not necessarily an API error; inspect the corresponding transcript before classifying it.

The low cohort has seven silent `end_session` turns and one absent logged input window (attempt 24, turn 0). That attempt recorded a startup greeting as turn 0 and sent the next question during greeting handling. This is a harness/startup caveat, not a provider API error. Its source transcript and accuracy scores are retained unchanged; the unmeasurable turn contributes no latency. A read-only check of all 125 original input sequences found no other such anomaly; see [sequence audit](original-input-sequences.json). New medium captures are checked for the same anomaly.

Input-window association fixes the marker-order assumption. It does not change the definition of V2V: first actual assistant speech after user speech, including acknowledgments, rather than time to the substantive answer. Missing measurements are excluded from latency statistics and explicitly counted. All planned turns remain in accuracy denominators.

See [full audit data](aggregate.json) and the preserved original tag-based measurements under the canonical and extension campaign `processed/` directories.
