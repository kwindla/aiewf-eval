# Gemma 4 26B A4B — AIEWF medium-context result

This report covers 30 canonical, strictly sequential thinking-off conversations
on the dedicated BaseTen deployment. Every conversation contributes 30
scheduled turns. Missing future turns after an early exit are failures; latency
is summarized only for observed model responses.

| Measure | Result |
|---|---:|
| Strict turn pass | 733/900 (81.4%) |
| Whole-conversation bootstrap 95% CI | 80.6–82.3% |
| Tool error | 13.0% |
| Instruction error | 18.6% |
| KB error | 0.0% |
| Full scheduled coverage | 30/30 |
| Strict protocol completion | 30/30 |
| TTFAT P50 / P95 / max | 597 / 670 / 4583 ms |

## Candidate README row

The analyzer does not edit `README.md`. After reviewing the result, insert this
row at the correct score-sorted position:

```text
| gemma-4-26b-a4b-it (thinking off) | 81.4% | 18.6% | 13.0% | 18.6% | 0.0% | 597ms | 670ms | 4583ms | BaseTen |
```

## Provenance

- Model: `google/gemma-4-26B-A4B-it`
- Provider: dedicated BaseTen vLLM deployment
- Serving: vLLM 0.26.1 development build, automatic prefix caching, one-token MTP
- Sampling: temperature 1.0, top-p 0.95, top-k 64, max tokens 8,192
- Thinking: explicitly disabled
- Filler: none
- Judge: `claude-opus-4-5` / `claude-agent-sdk-v4-turn-taking`
- Fixed denominator: 30 × 30 = 900 scheduled turns
