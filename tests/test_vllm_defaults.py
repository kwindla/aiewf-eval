"""vLLM sampling defaults: Qwen3.8 model-card profile, established defaults elsewhere."""

import os

import pytest

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


@pytest.fixture
def pipeline(monkeypatch):
    for name in os.environ:
        if name.startswith("MTE_VLLM_"):
            monkeypatch.delenv(name)
    instance = object.__new__(Pipeline)
    instance.service_name = "vllm-openai"
    return instance


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize(
    "model",
    ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.6-27B"],
)
def test_non_qwen38_models_keep_established_defaults(
    pipeline, monkeypatch, model, thinking
):
    monkeypatch.setenv("MTE_VLLM_THINKING", str(thinking).lower())
    params = pipeline._create_llm(Capture, model).kwargs["params"]
    assert params.temperature == 0.6
    assert params.top_p == 0.95
    template = params.extra["extra_body"]["chat_template_kwargs"]
    expected = {"enable_thinking": thinking}
    if model.startswith("Qwen/"):
        expected["preserve_thinking"] = thinking
    assert template == expected


@pytest.mark.parametrize("thinking", [False, True])
def test_explicit_sampling_overrides_defaults(pipeline, monkeypatch, thinking):
    monkeypatch.setenv("MTE_VLLM_THINKING", str(thinking).lower())
    monkeypatch.setenv("MTE_VLLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("MTE_VLLM_TOP_P", "0.9")
    params = pipeline._create_llm(Capture, "Qwen/Qwen3.6-27B").kwargs["params"]
    assert params.temperature == 0.4
    assert params.top_p == 0.9
