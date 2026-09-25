# Shared speech-judge clarification — approved September 15, 2026

The user approved this proposal. The implementation and uniform rejudge are recorded in [the revision protocol](rejudge-policy-compliance-1/protocol.json). The proposal text below is retained for context.

All model/frontend prompts, the knowledge base, questions, tools, and recordings stay byte-identical. The text-model judge also stays unchanged.

The speech judge currently includes both:

- “TRUE if assistant directly answers the question OR advances the task”
- The governing policy is authoritative, including the prohibition on inferred session totals.

Observed problem: some judgments treat any direct numeric answer as instruction-following, while others enforce the governing policy. This occurs across providers. Bare three-session totals received both pass and fail judgments. Qualified descriptions of a subsection require separate factual assessment from claims about an entire track.

## Proposed replacement for the instruction-following definition

An answer must both advance the user's request (or give an appropriate refusal/clarification) AND comply with the governing assistant policy. Answering the literal user question is not sufficient when doing so contradicts that policy. A realigned workflow must also comply with the governing policy.

Mark instruction_following false for a concrete contradiction of the governing policy even if the factual statements happen to be correct. Keep this separate from kb_grounding: mark grounding false only for an actual factual contradiction, verified against the entire supplied knowledge base. Do not excuse a policy violation merely because the answer advances the conversation. State the specific policy conflict in the explanation.

## Uniform application if adopted

- Preserve all original first-pass judge outputs and hashes.
- Apply the same clarified speech rubric to all 125 saved five-model conversations and all six standard-prompt GPT-Live conversations, with model identities hidden as before.
- Rejudge saved conversations; do not repeat voice captures or alter model prompts.
- Publish the new judge version/hash and a before/after score audit. Do not select individual favorable rejudgments.

Implementation changes only the shared speech judge. Model prompts and the text judge remain unchanged.
