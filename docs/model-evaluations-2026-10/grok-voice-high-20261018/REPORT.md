# Grok Voice 3.0: reasoning high versus none

| Mode | Complete | Unsuccessful / attempts | Strict pass | 95% conversation bootstrap CI | Tool use | Instructions | Grounding | Turn-taking |
|---|---|---|---|---|---|---|---|---|
| none | 25 | 1/26 | 89.7% | 88.0–91.3% | 737/750 | 680/750 | 719/750 | 748/750 |
| high | 25 | 0/25 | 92.8% | 91.6–94.0% | 750/750 | 697/750 | 729/750 | 748/750 |

High changes strict pass by +3.1 percentage points in these independent cohorts. These are new high conversations compared with the existing none cohort; the baseline was neither regenerated nor rejudged. Each cohort has 25 complete conversations and 750 scripted turns. The speech-v2 tasks, audio inputs and Opus 5.5 Medium rubric are the same.

| Mode | First-speech P50 | P95 | Non-tool median | Tool mean | Positive timing N | Nonpositive | Missing |
|---|---|---|---|---|---|---|---|
| none | 2048ms | 2624ms | 2016ms | 2418ms | 750 | 0 | 0 |
| high | 1984ms | 2336ms | 1984ms | 2056ms | 750 | 0 | 0 |

Latency comes from the actual stereo recording, using the same speech analyzer and three-landmark input waveform correlation as the none study. All completed captures must pass structural and input-alignment checks; timing problems are held for review. All unsuccessful attempts are retained. Quality never determines capture selection. The bootstrap uses the unchanged 20,000-sample conversation method; intervals describe these samples.

[Protocol](PROTOCOL.md) · [High aggregate](summary.json) · [Comparison data (local archive)](../README.md) · [Attempts (local archive)](../README.md) · [Frozen judge inputs (local archive)](../README.md) · [None report](../recent-models-20261018/REPORT.md)

High-only successful Opus judging cost: $4.988293. Judge credit cooldown events: 0; each waits at least 90 seconds. Private reasoning is not grading evidence.

[Evidence storage and local audit instructions](../README.md).
