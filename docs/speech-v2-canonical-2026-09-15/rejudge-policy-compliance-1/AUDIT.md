# Shared speech judge: before/after audit

All 131 conversations received one complete revised Opus 5 judgment. Original transcripts, model prompt provenance, and first-pass judgments passed hash verification. The only rubric change requires policy compliance for instruction following and independent factual assessment across the full KB. Model/frontend prompts and text-model judging are unchanged.

| Arm | Conversations | Original strict | Revised strict | Pass → fail | Fail → pass |
|---|---:|---:|---:|---:|---:|
| gemini38 | 25 | 701/750 (93.5%) | 697/750 (92.9%) | 6 | 2 |
| gemini31 | 25 | 688/750 (91.7%) | 649/750 (86.5%) | 39 | 0 |
| grok | 25 | 741/750 (98.8%) | 733/750 (97.7%) | 8 | 0 |
| openai15 | 25 | 709/750 (94.5%) | 705/750 (94.0%) | 5 | 1 |
| openai21 | 25 | 714/750 (95.2%) | 708/750 (94.4%) | 6 | 0 |
| live | 6 | 171/180 (95.0%) | 169/180 (93.9%) | 2 | 0 |

Execution/parse retries: **3**. These are judge-service failures, separate from model reliability.

Explicit score/explanation conflict flags: **3**. These diagnostic flags are preserved for review, without manual score correction.

The denominator includes every planned turn, including the six missing turns after the Gemini 3.1 timeout. GPT-Live here combines its single validation and five-run pilot; it is not a 25-attempt canonical cohort.

## MCP session-count question

| Arm | Instruction failures before | Instruction failures after | Numeric-pattern answers with instruction pass after |
|---|---:|---:|---:|
| gemini38 | 16/25 | 21/25 | 1 |
| gemini31 | 8/25 | 16/25 | 5 |
| grok | 0/25 | 1/25 | 3 |
| openai15 | 21/25 | 24/25 | 1 |
| openai21 | 16/25 | 21/25 | 2 |
| live | 1/6 | 2/6 | 1 |

Exact-answer disagreements are also recorded in the JSON audit. Numeric patterns are diagnostic, not automatic failures: qualified subsection descriptions require contextual review. See [all 131 answers and every changed judgment](audit.json). No manual score overrides or favorable selective rejudgments were used.

## Limits

Opus judgments are stochastic. Changes outside the motivating question can reflect both the revised rubric and ordinary judge variation; this is not a controlled estimate of rubric effect alone. Audio metrics and all per-turn audio pass/fail assessments were held fixed. Original scores remain in the capture directories and [original aggregate](../aggregate.json). The [revision protocol](protocol.json) fixes the full cohort and prompt hash.

## Explicit score/explanation flags

- `gemini31-01`, turn 4, `instruction_following`: Boolean `True`, explanation says `instruction_following is false`. Full judgment: `gemini31-01/claude_analysis.md` (local-only).
- `openai15-05`, turn 4, `instruction_following`: Boolean `True`, explanation says `instruction_following is false`. Full judgment: `openai15-05/claude_analysis.md` (local-only).
- `gemini31-25`, turn 4, `instruction_following`: Boolean `True`, explanation says `instruction_following is false`. Full judgment: `gemini31-25/claude_analysis.md` (local-only).

Sensitivity check only (not score overrides):
- If the flagged strict passes followed their explicit failure explanations, `gemini31` would be 86.3% instead of 86.5%.
- If the flagged strict passes followed their explicit failure explanations, `openai15` would be 93.9% instead of 94.0%.
