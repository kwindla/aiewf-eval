"""Protect the paid campaign from silently using provider-default reasoning."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.realtime import events

from multi_turn_eval.cli import infer_pipeline, load_service_class
from multi_turn_eval.pipelines.text import TextPipeline
from multi_turn_eval.pipelines.grok_realtime import XAIRealtimeLLMService


def service(monkeypatch, provider, model, **env):
    for key in ('MTE_ANTHROPIC_THINKING', 'MTE_ANTHROPIC_EFFORT'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(provider.upper() + '_API_KEY', 'test-key')
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    pipeline = object.__new__(TextPipeline)
    pipeline.service_name = provider
    return pipeline._create_llm(load_service_class(provider), model)


@pytest.mark.parametrize('effort', ['none', 'low'])
def test_luna_uses_responses_reasoning_with_tools(monkeypatch, effort):
    llm = service(monkeypatch, 'openai', 'gpt-6-luna',
                  MTE_OPENAI_RESPONSES_REASONING_EFFORT=effort)
    request = llm._responses_request_params(LLMContext(messages=[{'role':'user','content':'Hello'}]))
    assert request['model'] == 'gpt-6-luna'
    assert request['reasoning'] == {'effort': effort}


@pytest.mark.parametrize('model,mode', [
    ('claude-sonnet-5-5', 'between_tools'), ('claude-sonnet-5-5', 'adaptive'),
    ('claude-haiku-5-5', 'disabled'), ('claude-haiku-5-5', 'adaptive'),
])
def test_anthropic_low_does_not_override_disabled_mode(monkeypatch, model, mode):
    llm = service(monkeypatch, 'anthropic', model,
                  MTE_ANTHROPIC_THINKING=mode, MTE_ANTHROPIC_EFFORT='low')
    assert llm._settings.extra['thinking']['type'] == mode
    assert llm._settings.extra['output_config'] == {'effort':'low'}
    if mode == 'between_tools':
        assert llm._settings.extra['thinking'] == {'type':'between_tools'}


@pytest.mark.parametrize('model,mode,effort', [
    ('claude-sonnet-5-5','disabled','low'),
    ('claude-haiku-5-5','between_tools','low'),
    ('claude-sonnet-5-5','between_tools','max'),
    ('claude-haiku-5-5','disabled','xhigh'),
])
def test_anthropic_rejects_invalid_combinations(monkeypatch, model, mode, effort):
    with pytest.raises(ValueError):
        service(monkeypatch, 'anthropic', model,
                MTE_ANTHROPIC_THINKING=mode, MTE_ANTHROPIC_EFFORT=effort)


def test_grok_none_reaches_wire_after_openai_reasoning_filter():
    async def scenario():
        llm = object.__new__(XAIRealtimeLLMService)
        llm._xai_reasoning_effort = 'none'
        llm._ws_send = AsyncMock()
        event = events.SessionUpdateEvent(session=events.SessionProperties())
        await llm.send_client_event(event)
        assert llm._ws_send.await_args.args[0]['session']['reasoning'] == {'effort':'none'}
    asyncio.run(scenario())
    assert infer_pipeline('grok-voice-3.0') == 'grok-realtime'


def test_credit_reload_waits_90_seconds_before_retry(monkeypatch):
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from multi_turn_eval.services import anthropic_logged
    monkeypatch.setenv('MTE_ANTHROPIC_CREDIT_RETRY_SECONDS', '90')
    call=AsyncMock(side_effect=[RuntimeError('Your credit balance is too low'), 'stream'])
    sleep=AsyncMock()
    monkeypatch.setattr(AnthropicLLMService, '_create_message_stream', call)
    monkeypatch.setattr(anthropic_logged.asyncio, 'sleep', sleep)
    llm=object.__new__(anthropic_logged.LoggedAnthropicLLMService)
    result=asyncio.run(llm._create_message_stream(None, {'stream':False}))
    assert result=='stream' and call.await_count==2
    sleep.assert_awaited_once_with(90)


def test_sonnet55_unset_mode_uses_supported_low_latency_default(monkeypatch):
    from pathlib import Path
    llm=service(monkeypatch,'anthropic','claude-sonnet-5-5')
    assert llm._settings.extra['thinking']=={'type':'between_tools'}
