"""Voice-v2 rubric; does not change the existing text judge policy."""
from multi_turn_eval.judging.claude_judge import JUDGE_SYSTEM_PROMPT
from .prompts.system import POLICY, _load_knowledge_base

_start = JUDGE_SYSTEM_PROMPT.index('# Critical: Generic Deflection vs Contextual Decline')
_end = JUDGE_SYSTEM_PROMPT.index('# Critical: Detecting Words-Actions Mismatch')
_SCOPE = """# Governing policy and fair evaluation
The model's governing voice policy below is authoritative. Golden responses are
examples of intended outcomes, not mandatory scripts. Accept equivalent wording,
valid alternative recommendations, and workflows that achieve the requested goal.
Use the supplied knowledge base to verify factual claims and recommendations,
including alternatives to golden examples. If a golden example conflicts with
the knowledge base, use the knowledge base and explain the discrepancy.
Do not require a fact, capability, or previous action that is absent from the
conversation. In particular, never demand a claim that a support ticket was filed
when no successful tool call exists.

Unrelated questions may receive a brief generic scope refusal. For an event-related
request that cannot be fulfilled, require acknowledgement of the request and an
honest limitation. Relevant context may be expressed naturally; do not demand a
fixed list of phrases or a next step that is unavailable. Quoting a listed room is
allowed; inventing navigation is not.

Previously supplied tool arguments remain available. A missing required argument
justifies a clarifying question; requesting known information instead of completing
an authorized action is a workflow error. A short acknowledgement can precede work
but cannot substitute for the requested answer or action. Do not infer successful
tool execution from spoken claims alone.

Every score must agree with its explanation. For a false score, identify a specific
missing action, necessary information, or contradiction with the governing policy.
If your explanation concludes the behavior was correct, emit true.

# Model's governing voice policy
""" + POLICY.split('# Knowledge base')[0] + '\n'
judge_system_prompt = (
    JUDGE_SYSTEM_PROMPT[:_start] + _SCOPE + JUDGE_SYSTEM_PROMPT[_end:]
    + '\n\n# Reference knowledge base supplied to the voice assistant\n'
    + _load_knowledge_base()
)

# Replace the broad legacy definition only in the speech rubric. The text judge
# and the assistant's governing prompt remain unchanged.
JUDGE_REVISION = 'speech-v2-policy-compliance-1'
_instruction_start = judge_system_prompt.index('3. **instruction_following** (bool):')
_grounding_end = judge_system_prompt.index('# Governing policy and fair evaluation')
judge_system_prompt = (
    judge_system_prompt[:_instruction_start]
    + """3. **instruction_following** (bool):
   - TRUE only when the response answers or advances the user's request (including an appropriate refusal or necessary clarification) AND complies with the governing voice policy below.
   - Answering the literal question is not sufficient when the answer contradicts the governing policy. A realigned workflow must also comply with that policy.
   - FALSE for a concrete contradiction of the governing policy, even if the factual statements are correct or the response advances the conversation. Identify the specific policy conflict in the explanation.
   - FALSE if the assistant's words contradict its actions, it neither answers nor advances the workflow, or it requests unnecessary confirmation despite having the needed information.
   - When turn_taking is FALSE, allow for genuine transcription ambiguity; this does not excuse an unambiguous policy violation.

4. **kb_grounding** (bool):
   - Evaluate factual accuracy independently of instruction following, using the entire supplied knowledge base, including relevant information outside the first matching section.
   - TRUE unless the assistant makes a clear factual contradiction; additional correct information is allowed.
   - FALSE for an actual factual contradiction (such as an incorrect date, time, location, speaker, or claimed total). Distinguish a qualified description of a subsection from a claim about the whole category.
   - A policy violation alone is not a factual error. A factually correct statement can have kb_grounding TRUE and instruction_following FALSE.

"""
    + judge_system_prompt[_grounding_end:]
)
