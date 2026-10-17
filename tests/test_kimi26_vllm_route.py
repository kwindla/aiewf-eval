"""Kimi K2.6 request shaping for OpenAI-compatible vLLM/SGLang routes."""

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


def stub():
    pipeline = object.__new__(Pipeline)
    pipeline.service_name = "vllm-openai"
    return pipeline


def test_kimi26_thinking_off_sends_both_native_template_guards(monkeypatch):
    monkeypatch.setenv("MTE_VLLM_THINKING", "false")
    service = stub()._create_llm(Capture, "nvidia/Kimi-K2.6-NVFP4")
    assert service.kwargs["params"].extra["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "thinking": False,
    }


def test_kimi26_thinking_on_keeps_guards_consistent(monkeypatch):
    monkeypatch.setenv("MTE_VLLM_THINKING", "true")
    service = stub()._create_llm(Capture, "moonshotai/Kimi-K2.6")
    assert service.kwargs["params"].extra["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "thinking": True,
    }


def test_non_kimi_request_shape_is_unchanged(monkeypatch):
    monkeypatch.setenv("MTE_VLLM_THINKING", "false")
    service = stub()._create_llm(Capture, "deepseek-ai/DeepSeek-V4-Flash-0731")
    assert service.kwargs["params"].extra["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
    }
