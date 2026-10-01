import asyncio, importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, patch
from multi_turn_eval.pipelines import startup_guard as m


def test_startup_greeting_cannot_commit_as_first_user_answer():
    async def run():
        pipeline = m.StartupGuard.__new__(m.StartupGuard)
        pipeline._first_user_end_time = None
        with patch.object(
            m.RealtimePipeline, "_on_turn_end", new_callable=AsyncMock
        ) as commit:
            await pipeline._on_turn_end("Welcome to the fair.")
            commit.assert_not_awaited()
            pipeline._first_user_end_time = 123
            await pipeline._on_turn_end("The requested answer.")
            commit.assert_awaited_once_with("The requested answer.")

    asyncio.run(run())
