"""Qwen3.8 request shaping for the AIEWF vllm-openai route."""

import pytest
from openai import NOT_GIVEN

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.services.vllm_openai import is_qwen_reasoning_history_model


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


def test_qwen38_registered_seed_is_top_level_and_optional(monkeypatch):
    monkeypatch.setenv("MTE_VLLM_THINKING", "false")
    monkeypatch.setenv("MTE_VLLM_REASONING_EFFORT", "")
    monkeypatch.setenv("MTE_VLLM_MAX_TOKENS", "")
    monkeypatch.setenv("MTE_VLLM_SEED", "310017")
    seeded = stub()._create_llm(Capture, "qwen3.8-27b-nvfp4-unsloth")
    assert seeded.kwargs["params"].extra["seed"] == 310017
    assert "seed" not in seeded.kwargs["params"].extra["extra_body"]

    monkeypatch.setenv("MTE_VLLM_SEED", "")
    unseeded = stub()._create_llm(Capture, "qwen3.8-27b-nvfp4-unsloth")
    assert "seed" not in unseeded.kwargs["params"].extra


def test_qwen38_effort_and_toggle_must_agree(monkeypatch):
    monkeypatch.setenv("MTE_VLLM_THINKING", "true")
    monkeypatch.setenv("MTE_VLLM_REASONING_EFFORT", "")
    with pytest.raises(ValueError, match="require explicit"):
        stub()._create_llm(Capture, "qwen3.8-27b-nvfp4-unsloth")

    monkeypatch.setenv("MTE_VLLM_THINKING", "false")
    monkeypatch.setenv("MTE_VLLM_REASONING_EFFORT", "low")
    with pytest.raises(ValueError, match="must omit"):
        stub()._create_llm(Capture, "qwen3.8-27b-nvfp4-unsloth")


def test_qwen38_reasoning_history_recognition():
    assert is_qwen_reasoning_history_model("Qwen3.8-27B-NVFP4")
    assert is_qwen_reasoning_history_model("Qwen/Qwen3.8-27B-FP8")
