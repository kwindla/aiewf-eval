# Mechanical scorer validation

The frozen scorer agreed with the historical Claude tool-use judgment on
**897/900 target turns (99.667%)**.
The preregistered 99.5% gate **passed**.

The validation set is all turns 12 and 15 from 150 BaseTen BF16, 150 local
FP8-KV, and 150 local BF16-KV completed conversations. The direct replay's
primary outcome is stricter about multiple calls: exactly one correct call is
required. Historical transcripts already contain the benchmark's streamed-call
normalization and deduplication, so the validation tests parity after that seam.

## Categories

```json
{
  "correct_tool_and_arguments": 424,
  "no_tool_false_claim_of_completion": 157,
  "no_tool_other": 13,
  "no_tool_redundant_confirmation_or_question": 306
}
```

## Disagreements

```json
[
  {
    "source": "baseten_bf16",
    "cohort": "n30",
    "slot": 17,
    "run_dir": "runs/aiwf_medium_context/20260806T224411Z_google_gemma-4-31B-it_none_slot017_attempt01_ea4ad44f",
    "turn": 15,
    "mechanical_success": false,
    "mechanical_category": "no_tool_false_claim_of_completion",
    "judge_tool_use_correct": true,
    "agrees": false,
    "assistant_text": "Thank you, Jennifer. I've submitted your request for vegan meals. Is there anything else I can help you with today?",
    "tool_calls": [],
    "judge_reasoning": "Function submit_dietary_request was expected here but was already called at turn 14 (early). Realignment applies - tool_use_correct is TRUE since the function was satisfied. Assistant correctly confirmed the submission."
  },
  {
    "source": "baseten_bf16",
    "cohort": "n120",
    "slot": 37,
    "run_dir": "runs/aiwf_medium_context/20260807T103210Z_google_gemma-4-31B-it_none_slot037_attempt01_55364dce",
    "turn": 15,
    "mechanical_success": false,
    "mechanical_category": "no_tool_redundant_confirmation_or_question",
    "judge_tool_use_correct": true,
    "agrees": false,
    "assistant_text": "Thank you, Jennifer. I've successfully submitted your request for vegan meals. Is there anything else I can help you with today?",
    "tool_calls": [],
    "judge_reasoning": "User said 'Yes' to confirm dietary request. Function was already called at turn 14 (early). Response appropriately confirms the submission was made. No penalty - realignment case."
  },
  {
    "source": "local_bf16",
    "cohort": "n120",
    "slot": 102,
    "run_dir": "runs/aiwf_medium_context/20260807T124540Z_google_gemma-4-31B-it_none_slot102_attempt01_132f5562",
    "turn": 15,
    "mechanical_success": false,
    "mechanical_category": "no_tool_false_claim_of_completion",
    "judge_tool_use_correct": true,
    "agrees": false,
    "assistant_text": "Thank you, Jennifer! I've submitted your request for vegan meals. Is there anything else I can help you with today?",
    "tool_calls": [],
    "judge_reasoning": "Expected submit_dietary_request, but it was already called at turn 14. Realignment applies - the expected function was satisfied by the early call. Response appropriately confirms submission."
  }
]
```
