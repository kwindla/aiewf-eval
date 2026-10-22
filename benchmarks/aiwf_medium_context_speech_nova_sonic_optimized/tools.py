"""Nova Sonic prompt-optimized tool descriptions.

Names, parameters, types and required arguments are identical to benchmarks/_shared/tools.py;
only the tool and parameter descriptions differ.
"""
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

_NAME = "The user's name as they gave it earlier in the conversation."

# 1. End Session
end_session_function = FunctionSchema(
    name="end_session",
    description="Closes the conversation. Call it with your farewell whenever the user says they are finished or have nothing else.",
    properties={},
    required=[],
)

# 2. Submit Dietary Request
submit_dietary_request_function = FunctionSchema(
    name="submit_dietary_request",
    description="Sends an attendee's dietary need to the event organizers. Call it when the user asks for a dietary request or accepts your offer to submit one.",
    properties={
        "name": {"type": "string", "description": _NAME},
        "dietary_preference": {
            "type": "string",
            "description": "The dietary need the user stated, such as vegetarian or gluten-free.",
        },
    },
    required=["name", "dietary_preference"],
)

# 3. Submit Session Suggestion
submit_session_suggestion_function = FunctionSchema(
    name="submit_session_suggestion",
    description="Sends an attendee's idea for a new session to the organizers. Call it for each suggestion as soon as you know the user's name and the topic they suggest.",
    properties={
        "name": {"type": "string", "description": _NAME},
        "suggestion_text": {
            "type": "string",
            "description": "A short statement of the suggested session, in the user's terms.",
        },
    },
    required=["name", "suggestion_text"],
)

# 4. Vote for a Session
vote_for_session_function = FunctionSchema(
    name="vote_for_session",
    description="Records the user's vote for the one session they have chosen. Call it once the user has picked a single session.",
    properties={
        "name": {"type": "string", "description": _NAME},
        "session_id": {
            "type": "string",
            "description": "The number on the chosen session's Session ID line in the knowledge base. Used in this call only, not spoken.",
        },
    },
    required=["name", "session_id"],
)

# 5. Request for Tech Support
request_tech_support_function = FunctionSchema(
    name="request_tech_support",
    description="Opens a technical support request for an attendee. Call it once the user has described a concrete problem and you know their name.",
    properties={
        "name": {"type": "string", "description": _NAME},
        "issue_description": {
            "type": "string",
            "description": "A concrete description of the problem, including what the user can't do and where.",
        },
    },
    required=["name", "issue_description"],
)

ToolsSchemaNovaSonicOptimized = ToolsSchema(
    standard_tools=[
        end_session_function,
        submit_dietary_request_function,
        submit_session_suggestion_function,
        vote_for_session_function,
        request_tech_support_function,
    ]
)