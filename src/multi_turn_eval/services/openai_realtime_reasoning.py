"""OpenAI Realtime LLM service with configurable reasoning effort.

The stock pipecat `OpenAIRealtimeLLMService` does not expose the
`session.reasoning.effort` field that GPT-5-family realtime models accept.
This subclass adds a `reasoning_effort` constructor kwarg and injects it
into every outgoing `session.update` event.

Server-accepted values (per probe of gpt-realtime-alpha-dolphin-14):
    "minimal", "low", "medium", "high"
The undocumented value "none" is also accepted by the server and is echoed
back in `session.updated`. Any other value triggers an `invalid_value` error.
"""

from typing import Optional

from loguru import logger
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


class OpenAIRealtimeReasoningLLMService(OpenAIRealtimeLLMService):
    """OpenAIRealtimeLLMService that sets `session.reasoning.effort`."""

    def __init__(
        self,
        *args,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._reasoning_effort = reasoning_effort
        if reasoning_effort:
            logger.info(
                f"OpenAIRealtimeReasoningLLMService: reasoning.effort={reasoning_effort!r}"
            )

    async def send_client_event(self, event: events.ClientEvent):
        if isinstance(event, events.SessionUpdateEvent) and self._reasoning_effort:
            payload = event.model_dump(exclude_none=True)
            payload["session"]["reasoning"] = {"effort": self._reasoning_effort}
            await self._ws_send(payload)
            return
        await super().send_client_event(event)
