"""Protect Nova point releases against wrong-model calls and stale wire settings."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from multi_turn_eval.cli import infer_pipeline
from multi_turn_eval.pipelines.nova_sonic import NovaSonicLLMServiceWithCompletionSignal


@pytest.mark.parametrize('model', [
    'amazon.nova-sonic-v1:0', 'amazon.nova-2-sonic-v1:0',
    'amazon.nova-2-5-sonic', 'nova_sonic',
])
def test_nova_pipeline_inference(model):
    assert infer_pipeline(model) == 'nova-sonic'


def test_point_release_model_and_endpointing_reach_wire():
    llm = NovaSonicLLMServiceWithCompletionSignal(
        access_key_id='test', secret_access_key='test', region='us-east-1',
        model='amazon.nova-2-5-sonic', endpointing_sensitivity='HIGH',
    )
    assert llm._settings.model == 'amazon.nova-2-5-sonic'
    event = json.loads(llm.build_session_start_json())['event']['sessionStart']
    assert event['turnDetectionConfiguration']['endpointingSensitivity'] == 'HIGH'
    assert 'endpointingSensitivity' not in event['inferenceConfiguration']
    # Inherit Pipecat's stream-bound receive loop and single transcript emission.
    assert '_receive_task_handler' not in type(llm).__dict__


def test_recovery_disabled_never_truncates_context(monkeypatch):
    monkeypatch.setenv('MTE_ENABLE_RECOVERY', '0')
    llm = object.__new__(NovaSonicLLMServiceWithCompletionSignal)
    llm._wants_connection = True
    llm._on_max_reconnects_exceeded = AsyncMock()
    llm._truncate_context_for_reconnection = lambda: pytest.fail('Lost context')
    asyncio.run(llm.reset_conversation())
    assert not llm._wants_connection
    llm._on_max_reconnects_exceeded.assert_awaited_once()


def test_pipeline_passes_requested_point_release_to_service(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from multi_turn_eval.pipelines import nova_sonic
    from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig

    monkeypatch.setenv('AWS_ACCESS_KEY_ID','test')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY','test')
    captured={}

    class Constructed(Exception): pass
    def capture(**kwargs):
        captured.update(kwargs)
        raise Constructed
    monkeypatch.setattr(nova_sonic,'NovaSonicLLMServiceWithCompletionSignal',capture)
    pipeline=nova_sonic.NovaSonicPipeline(BenchmarkConfig())
    with pytest.raises(Constructed):
        asyncio.run(pipeline.run(SimpleNamespace(run_dir=tmp_path),'amazon.nova-2-5-sonic'))
    assert captured['settings'].model=='amazon.nova-2-5-sonic'
    assert captured['settings'].system_instruction==BenchmarkConfig.system_instruction
    assert captured['settings'].voice=='tiffany'


def test_user_reference_layout_preserves_every_prompt_byte(monkeypatch):
    from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
    from multi_turn_eval.pipelines.nova_sonic import NovaSonicPipeline
    monkeypatch.setenv('MTE_NOVA_KB_CONTEXT_ROLE','user')
    pipeline=NovaSonicPipeline(BenchmarkConfig())
    policy,messages=pipeline._prompt_configuration()
    assert messages==[{'role':'user','content':BenchmarkConfig.reference_context}]
    assert policy+messages[0]['content']==BenchmarkConfig.system_instruction
    assert 'Voice Cloning and Synthesis' in messages[0]['content']
    assert '- Replicate - Run models' in messages[0]['content']


def test_user_reference_layout_rejects_missing_or_inexact_parts(monkeypatch):
    from types import SimpleNamespace
    from multi_turn_eval.pipelines.nova_sonic import NovaSonicPipeline
    monkeypatch.setenv('MTE_NOVA_KB_CONTEXT_ROLE','user')
    for fields in [{},{'system_policy':'rules','reference_context':'truncated'}]:
        pipeline=NovaSonicPipeline(SimpleNamespace(turns=[],system_instruction='complete',**fields))
        with pytest.raises(ValueError,match='exact policy/reference'):
            pipeline._prompt_configuration()


def test_local_vad_and_timing_logger_are_wired_before_input_transport(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from multi_turn_eval.pipelines import nova_sonic
    from multi_turn_eval.transports import paced_input
    from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
    monkeypatch.setenv('AWS_ACCESS_KEY_ID','test');monkeypatch.setenv('AWS_SECRET_ACCESS_KEY','test')
    pipeline=nova_sonic.NovaSonicPipeline(BenchmarkConfig())
    class Constructed(Exception):pass
    def capture(params,**kwargs):
        assert isinstance(pipeline.context_aggregator,LLMContextAggregatorPair)
        vad=pipeline.context_aggregator.user()._params.vad_analyzer
        assert isinstance(vad,nova_sonic.ResamplingSileroVAD)
        assert pipeline.llm_logger._vad_params.start_secs==.2
        assert pipeline.llm_logger._vad_params.stop_secs==.8
        assert not hasattr(params,'vad_analyzer')
        raise Constructed
    monkeypatch.setattr(paced_input,'PacedInputTransport',capture)
    with pytest.raises(Constructed):
        asyncio.run(pipeline.run(SimpleNamespace(run_dir=tmp_path),'amazon.nova-2-5-sonic'))


def test_pacer_is_released_at_recording_baseline_before_connection_wait(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from multi_turn_eval.pipelines import nova_sonic
    pipeline=object.__new__(nova_sonic.NovaSonicPipeline)
    pipeline.audio_buffer=SimpleNamespace(start_recording=AsyncMock(),_init_sample_rate=24000)
    events=[]
    pipeline.paced_input=SimpleNamespace(set_recording_baseline=lambda:events.append('baseline'),
                                        signal_ready=lambda:events.append('ready'))
    pipeline.output_transport=SimpleNamespace(reset_recording_baseline=Mock(),enable_greeting_tag=Mock())
    pipeline.llm_logger=SimpleNamespace(set_recording_start_time=Mock())
    pipeline.task=SimpleNamespace(queue_frames=AsyncMock(),cancel=AsyncMock())
    pipeline.turn_gate=SimpleNamespace(clear_pending=Mock())
    pipeline._get_current_turn=lambda:{}
    pipeline._get_audio_path_for_turn=lambda index:None
    pipeline.turn_idx=0
    async def scenario():
        pipeline._greeting_started=asyncio.Event();pipeline._greeting_started.set()
        pipeline._greeting_done=asyncio.Event();pipeline._greeting_done.set()
        async def sleep(seconds):
            if seconds>0:assert events==['baseline','ready']
        monkeypatch.setattr(nova_sonic.asyncio,'sleep',sleep)
        await pipeline._queue_first_turn(delay=0)
    asyncio.run(scenario())
    assert events==['baseline','ready']


def test_native_tool_result_delivery_is_immediate_and_idempotent():
    from pipecat.processors.aggregators.llm_context import LLMContext
    llm=object.__new__(NovaSonicLLMServiceWithCompletionSignal)
    llm._completed_tool_calls=set();llm._send_tool_result=AsyncMock()
    llm._context=LLMContext([{'role':'tool','tool_call_id':'call-one','content':'{"status":"success"}'}])
    async def scenario():
        await llm.complete_tool_call('call-one',{'status':'success'})
        await llm.complete_tool_call('call-one',{'status':'success'})
        await llm._process_completed_function_calls(send_new_results=True)
    asyncio.run(scenario())
    llm._send_tool_result.assert_awaited_once_with('call-one',{'status':'success'})
    assert llm._completed_tool_calls=={'call-one'}


def test_native_tool_result_delivery_does_not_cache_a_send_error():
    llm=object.__new__(NovaSonicLLMServiceWithCompletionSignal)
    llm._completed_tool_calls=set();llm._send_tool_result=AsyncMock(side_effect=RuntimeError('closed'))
    with pytest.raises(RuntimeError,match='closed'):
        asyncio.run(llm.complete_tool_call('call-one',{'status':'success'}))
    assert not llm._completed_tool_calls


def test_scripted_turn_tag_survives_early_speech_and_late_vad(monkeypatch):
    from pipecat.frames.frames import UserStoppedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import TransportParams
    from multi_turn_eval.transports.null_audio_output import NullAudioOutputTransport
    monkeypatch.setattr(BaseOutputTransport,'process_frame',AsyncMock())
    transport=NullAudioOutputTransport(TransportParams())
    async def scenario():
        transport.mark_scripted_user_turn()
        assert transport._tag_next_bot_audio
        # First bot audio consumes the marker before local VAD-stop arrives.
        transport._tag_next_bot_audio=False
        await transport.process_frame(UserStoppedSpeakingFrame(),FrameDirection.DOWNSTREAM)
        assert not transport._tag_next_bot_audio
        # A genuinely new scripted WAV arms the next response independently.
        transport.mark_scripted_user_turn()
        assert transport._tag_next_bot_audio
    asyncio.run(scenario())
