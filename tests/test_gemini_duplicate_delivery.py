import asyncio, importlib.util, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from multi_turn_eval.pipelines import gemini_regular as m


def test_duplicate_receives_response_without_executing_action_again():
    async def check():
        llm = SimpleNamespace(_tool_result=AsyncMock())
        pipeline = SimpleNamespace(
            _duplicate_tool_call_ids={"duplicate-id"},
            llm=llm,
            audit=lambda *a, **k: None,
        )
        params = SimpleNamespace(
            tool_call_id="duplicate-id", function_name="submit_session_suggestion"
        )
        # Use a real instance type so the inherited handler's super() binds correctly.
        pipeline = object.__new__(m.RegularPipeline)
        pipeline._duplicate_tool_call_ids = {"duplicate-id"}
        pipeline.llm = llm
        pipeline.audit = lambda *a, **k: None
        with patch.object(
            m.AuditedGeminiPipeline, "_function_catchall", new=AsyncMock()
        ) as original:
            await pipeline._function_catchall(params)
            original.assert_awaited_once_with(params)
            llm._tool_result.assert_awaited_once_with(
                "duplicate-id",
                "submit_session_suggestion",
                {"status": "duplicate_skipped"},
            )
            llm._tool_result.reset_mock()
            params.tool_call_id = "new-id"
            await pipeline._function_catchall(params)
            llm._tool_result.assert_not_awaited()

    asyncio.run(check())
