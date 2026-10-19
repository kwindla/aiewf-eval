"""Policy-grounded text rubric, calibrated separately from the speech rubric."""

from multi_turn_eval.judging.claude_judge import JUDGE_SYSTEM_PROMPT
from .prompts.system import system_instruction

JUDGE_REVISION = "text-policy-compliance-1"

_start = JUDGE_SYSTEM_PROMPT.index("3. **instruction_following** (bool):")
_end = JUDGE_SYSTEM_PROMPT.index("# Critical: Detecting Words-Actions Mismatch")

judge_system_prompt = (
    JUDGE_SYSTEM_PROMPT[:_start]
    + """3. **instruction_following** (bool):
   - TRUE when the assistant answers or advances the user's request, including a reasonable clarification or a refusal required by the governing policy below.
   - FALSE for a concrete policy violation, failure to advance the request, or words that contradict actual actions.
   - Use the actual conversation and governing policy, not literal similarity to the golden response.
   - Missing required tool arguments justify clarification; re-requesting already known tool arguments instead of acting is a failure.

4. **kb_grounding** (bool):
   - Evaluate factual accuracy independently using the full reference knowledge base below.
   - TRUE unless the response makes a clear factual contradiction; additional correct information and valid alternative recommendations are allowed.
   - FALSE for an actual incorrect date, time, room, speaker, or invented total. A policy violation alone is not a factual contradiction.

# Governing policy and fair evaluation

The assistant's supplied system instruction below is authoritative. Golden
responses illustrate acceptable outcomes; they are not mandatory scripts. Do not
require extra details, exact wording, or actions merely because they appear in a
golden response. Verify alternative factual answers against the full knowledge
base. If a golden example conflicts with that knowledge base, explain the conflict
and use the knowledge base.

The policy explicitly excludes venue directions and suggests a generic scope
refusal. Accept such a refusal for navigation requests, including after a maps
problem; mentioning a previous support request is optional. Do not require a
claim that a ticket was filed, especially when no successful action exists.
Quoting a session's listed room as schedule metadata is allowed; inventing a
route through the venue is not. Generic refusals are also appropriate for
unrelated subjects such as weather.

A correction to the attendance date may also change the user's availability.
A brief question about all-day versus morning/afternoon attendance on the new
date is a reasonable clarification, even if the old date had all-day availability.
Directly giving valid recommendations for the new date is also acceptable. This
does not excuse repeatedly asking unchanged information or delaying an authorized
tool action whose required arguments are already known. A clarification with no
required tool call does not itself fail tool_use_correct.

Judge tool execution from actual calls, never from a spoken assertion alone.
This text capture harness can end a tool turn without assistant text; the valid
tool call itself is sufficient and is not an instruction-following failure for
omitting a spoken acknowledgement. Continue to enforce argument correctness,
reuse of known parameters, and the end_session call.

Every false score needs a concrete policy conflict, missing necessary action,
or factual contradiction. Scores must agree with the explanation. A preference
for a more helpful or elegant answer is insufficient to fail a valid response.

"""
    + JUDGE_SYSTEM_PROMPT[_end:]
    + "\n\n# Reference: assistant's governing system instruction and knowledge base\n"
    + system_instruction
)
