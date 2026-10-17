"""Nemotron 3.5 Lightning request shaping for OpenAI-compatible routes."""

from multi_turn_eval.pipelines.base import BasePipeline


class Pipeline(BasePipeline):
    def _setup_context(self):
        pass

    def _setup_llm(self):
        pass

    def _build_task(self):
        pass

    async def _queue_first_turn(self):
        pass

    async def _queue_next_turn(self):
        pass


class Capture:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def stub() -> BasePipeline:
    pipeline = object.__new__(Pipeline)
    pipeline.service_name = "vllm-openai"
    return pipeline


def test_lightning_thinking_off_enables_nonempty_content_guard(monkeypatch):
    monkeypatch.setenv("MTE_VLLM_THINKING", "false")
    service = stub()._create_llm(
        Capture, "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    )
    assert service.kwargs["params"].extra["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "force_nonempty_content": True,
    }


def test_lightning_nonempty_content_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MTE_VLLM_FORCE_NONEMPTY_CONTENT", "false")
    service = stub()._create_llm(
        Capture, "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    )
    assert service.kwargs["params"].extra["extra_body"]["chat_template_kwargs"][
        "force_nonempty_content"
    ] is False
