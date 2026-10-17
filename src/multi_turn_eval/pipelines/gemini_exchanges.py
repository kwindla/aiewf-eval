"""Opt-in Gemini exchange audit; standard prompts and provider-driven completion."""

import asyncio, json, os, time
from multi_turn_eval.pipelines.realtime import (
    RealtimePipeline,
    GeminiLiveLLMServiceWithReconnection,
    GeminiVADParams,
)


class AuditedGeminiService(GeminiLiveLLMServiceWithReconnection):
    def __init__(self, *, audit, **kwargs):
        self.audit = audit
        super().__init__(**kwargs)

    async def _connection_task_handler(self, config):
        # Extended Thinking requires explicit NON_BLOCKING declarations. Pipecat's
        # Gemini-3 default omits behavior, so set it on this opt-in route only.
        for tool in config.tools or []:
            declarations = (
                tool.get("function_declarations", [])
                if isinstance(tool, dict)
                else (tool.function_declarations or [])
            )
            for declaration in declarations:
                if isinstance(declaration, dict):
                    declaration["behavior"] = "NON_BLOCKING"
                else:
                    declaration.behavior = "NON_BLOCKING"
        self.audit("setup", config=config.model_dump(mode="json", exclude_none=True))
        await super()._connection_task_handler(config)

    async def _handle_session_ready(self, session):
        original = session.receive

        async def receive():
            async for message in original():
                data = message.model_dump(mode="json", exclude_none=True)
                for part in (
                    data.get("server_content", {})
                    .get("model_turn", {})
                    .get("parts", [])
                ):
                    blob = part.get("inline_data")
                    if blob and "data" in blob:
                        blob["base64_length"] = len(blob.pop("data"))
                self.audit("server", message=data)
                status = self._get_interaction_status(message)
                if status == "IN_PROGRESS":
                    self._interaction_in_progress = True
                yield message
                # Pipecat dispatches completion only when turn_complete is set. An IDLE
                # status-only message still needs to close the asynchronous interaction.
                sc = message.server_content
                if sc and not sc.turn_complete and status == "IDLE":
                    await self._handle_msg_turn_complete(message)

        session.receive = receive
        await super()._handle_session_ready(session)

    async def _handle_msg_turn_complete(self, message):
        status = self._get_interaction_status(message)
        self.audit(
            "phase_boundary",
            status=status,
            input_open=self._explicit_input_activity_open,
        )
        await super()._handle_msg_turn_complete(message)
        if status == "IDLE" and not self._explicit_input_activity_open:
            self.audit("interaction_idle")


class AuditedGeminiPipeline(RealtimePipeline):
    def __init__(self, benchmark, event_path, **kwargs):
        super().__init__(benchmark, **kwargs)
        self.event_file = event_path.open("w", buffering=1)
        self._end_session_requested = False
        self._closing_task = None

    def audit(self, kind, **data):
        self.event_file.write(
            json.dumps(
                dict(monotonic=time.monotonic(), turn=self.turn_idx, type=kind, **data),
                default=str,
            )
            + "\n"
        )
        if kind == "interaction_idle" and self._end_session_requested and not self.done:
            if self._closing_task is None or self._closing_task.done():
                self._closing_task = asyncio.create_task(self._finish_silent_closure())

    def _create_llm(self, service_class, model):
        return AuditedGeminiService(
            audit=self.audit,
            api_key=os.environ["GOOGLE_API_KEY"],
            tools=self.benchmark.tools_schema,
            inference_on_context_initialization=True,
            on_reconnecting=self._on_gemini_reconnecting,
            on_reconnected=self._on_gemini_reconnected,
            gemini_3_protocol=True,
            require_interaction_status=True,
            settings=AuditedGeminiService.Settings(
                model=model,
                system_instruction=self.benchmark.system_instruction,
                thinking=self._build_gemini_live_thinking_config(model),
                vad=GeminiVADParams(disabled=True),
            ),
        )

    async def _function_catchall(self, params):
        if params.function_name != "end_session":
            return await super()._function_catchall(params)
        key = (params.function_name, str(params.arguments or {}))
        if self._enable_tool_call_dedupe and key in self._seen_tool_calls:
            self._duplicate_tool_call_ids.add(params.tool_call_id)
            return await self._emit_tool_result_callback(
                params, {"status": "duplicate_skipped"}
            )
        self._seen_tool_calls.add(key)
        self._end_session_requested = True
        self.audit("end_session_requested")
        await self._emit_tool_result_callback(params, {"status": "success"})
        # Do not cancel: allow the provider to produce its farewell and terminal IDLE.

    async def _finish_silent_closure(self):
        # Normal spoken closure goes through TurnGate. This fallback handles a truly
        # silent tool-only close after provider IDLE and drained playback.
        await asyncio.sleep(1)
        while not self.done and (
            self.llm.is_interaction_in_progress() or self.turn_gate._bot_speaking
        ):
            await asyncio.sleep(0.05)
        if not self.done and not self.turn_gate._pending_transcript:
            await self._on_turn_end("")

    async def _on_turn_end(self, text):
        if self.done:
            return
        self.audit("exchange_commit", text=text)
        if (
            self._end_session_requested
            and self.turn_idx < len(self.effective_turns) - 1
        ):
            self.recorder.write_turn(
                user_text=self.effective_turns[self.turn_idx]["input"],
                assistant_text=text,
            )
            await self._terminate_failed_run("model_end_session_early")
        else:
            await super()._on_turn_end(text)
