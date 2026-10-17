"""Exclude startup greetings from scripted conversation turns."""

from loguru import logger
from multi_turn_eval.pipelines.realtime import RealtimePipeline


class StartupGuard(RealtimePipeline):
    async def _on_turn_end(self, text):
        if self._first_user_end_time is None:
            logger.info(
                "[STARTUP_GUARD] Ignoring transcript before scripted input: {}", text
            )
            return
        await super()._on_turn_end(text)
