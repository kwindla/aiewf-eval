# Nemotron 3.5 Lightning thinking-on filler results

Each arm contains 30 full assigned conversations and uses a fixed 900-turn denominator. The no-filler arm is frozen from the preceding binary campaign; dots and dashes were collected fresh and interleaved.

| Arm | Pass rate | 95% CI | Full conversations | TTFAT P50 / P95 | Raw TTFT P50 |
|---|---:|---:|---:|---:|---:|
| nofiller | 842/900 (93.6%) | 92.4–94.6% | 30/30 | 1464/5786 ms | 65 ms |
| dots96 | 178/900 (19.8%) | 15.1–24.7% | 0/30 | 3699/157571 ms | 68 ms |
| dashes96 | 256/900 (28.4%) | 22.3–34.8% | 0/30 | 3371/152954 ms | 69 ms |

## Effects

- `dots96_minus_nofiller`: -73.78 points (whole-conversation bootstrap 95% CI -78.67 to -68.78).
- `dashes96_minus_nofiller`: -65.11 points (whole-conversation bootstrap 95% CI -71.33 to -58.78).
- `dashes96_minus_dots96`: +8.67 points (whole-conversation bootstrap 95% CI +0.67 to +16.56).

## Selected tool-commitment turns

| Turn | nofiller | dots96 | dashes96 |
|---:|---:|---:|---:|
| 11 | 29/30 | 3/30 | 5/30 |
| 12 | 30/30 | 1/30 | 4/30 |
| 15 | 29/30 | 0/30 | 4/30 |
| 17 | 29/30 | 0/30 | 1/30 |
| 24 | 30/30 | 0/30 | 0/30 |
| 25 | 30/30 | 0/30 | 0/30 |
| 29 | 30/30 | 0/30 | 0/30 |

## Failure-mode audit

- All 60 filler conversations ended after an explicit SGLang context-length rejection; none reached all 30 scheduled turns. In every run, the terminal recorded response brought `total_tokens` to at least 65,000 immediately before the next request exceeded the 65,536-token context.
- Dots produced 206 observed scripted responses (mean 6.87, median 5, range 1–16) and 694 missing future turns. Dashes produced 290 observed responses (mean 9.67, median 11, range 1–22) and 610 missing future turns.
- Conditional on a response being observed, strict accuracy was 178/206 (86.4%) for dots and 256/290 (88.3%) for dashes, versus 842/900 (93.6%) for no filler. Thus most of the production-score loss comes from context exhaustion, with a smaller pre-exhaustion accuracy loss.
- The run log's `Recorded turn N: ...` rendering is not a literal ellipsis response. Transcript inspection found zero assistant texts exactly equal to `...`, zero empty assistant responses without a tool call, and 19 empty text fields paired with valid tool calls.
- The suffix is behaviorally visible to the model. Some responses explicitly discuss the long punctuation line, and some expose internal-style deliberation about how to interpret it. This construction therefore does not isolate additional latent computation for Nemotron 3.5 Lightning; it introduces a strong prompt-semantic and runaway-generation intervention.
- Counts are repeated space-separated glyph strings, not tokenizer-normalized equal token budgets. The dots-versus-dashes difference should therefore be interpreted as a comparison of these exact prompt suffixes, not a pure glyph effect at matched model-token count.
