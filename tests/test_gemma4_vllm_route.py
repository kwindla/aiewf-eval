"""Gemma 4 request shaping for OpenAI-compatible vLLM/SGLang routes."""

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


def test_gemma4_thinking_off_sends_both_native_template_guards(monkeypatch):
    monkeypatch.setenv("MTE_VLLM_THINKING", "false")
    service = stub()._create_llm(Capture, "google/gemma-4-31B-it")
    assert service.kwargs["params"].extra["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "thinking": False,
    }


def test_non_gemma_route_does_not_receive_gemma_thinking_guard(monkeypatch):
    monkeypatch.setenv("MTE_VLLM_THINKING", "false")
    service = stub()._create_llm(Capture, "deepseek-ai/DeepSeek-V4-Pro-0813")
    assert service.kwargs["params"].extra["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
    }
