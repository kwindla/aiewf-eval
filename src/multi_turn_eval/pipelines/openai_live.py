"""GPT-Live benchmark adapter: continuous audio, delegated tools, paced playback.

Live has no authoritative speech-completed event. This adapter uses drained
playback plus a conservative quiet interval, while holding turns open throughout
backend work and tool continuation. Raw events make that heuristic auditable.
"""
import asyncio
import base64
import json
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pipecat.frames.frames import (
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, BotSpeakingFrame,
    EndFrame, ErrorFrame, InputAudioRawFrame, LLMContextFrame, MetricsFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.live.llm import OpenAILiveLLMService, ResponsesDelegation
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMSettings, OpenAIResponsesReasoningConfig
from pipecat.transports.base_transport import TransportParams
from multi_turn_eval.processors.audio_buffer import WallClockAlignedAudioBufferProcessor
from multi_turn_eval.processors.tool_call_recorder import ToolCallRecorder
from multi_turn_eval.transports.null_audio_output import NullAudioOutputTransport


@dataclass
class LiveTurnState:
    quiet_seconds: float = 3.0
    active: set = field(default_factory=set)
    had_calls: set = field(default_factory=set)
    speaking: bool = False
    last_activity: float = 0.0
    transcript: str = ''
    input_finished: bool = False

    def event(self, e, now):
        kind = e['type']
        if kind == 'session.output_transcript.delta':
            self.transcript += e['delta']
            self.last_activity = now
        elif kind == 'session.delegation.created':
            self.active.add(e['delegation']['id'])
            self.last_activity = now
        elif kind == 'response.event':
            key = e.get('delegation_id', 'uncorrelated')
            inner = e['event']; subtype = inner['type']
            if subtype == 'response.created':
                self.active.add(key)
                self.had_calls.discard(key)
            elif subtype == 'response.output_item.done' and inner.get('item', {}).get('type') == 'function_call':
                self.had_calls.add(key)
            elif subtype == 'response.completed':
                # A completed tool-calling response is not the end of its
                # delegation: Pipecat sends results and starts a continuation.
                if key not in self.had_calls:
                    self.active.discard(key)
            self.last_activity = now

    def ready(self, now):
        return (self.input_finished and bool(self.transcript.strip()) and
                not self.active and not self.speaking and
                now - self.last_activity >= self.quiet_seconds)

    def finish_input(self, now):
        # A backchannel can finish while the user is still speaking. Start a
        # fresh quiet interval at input end so it cannot commit that turn early.
        self.input_finished = True
        self.last_activity = max(self.last_activity, now)

    def begin(self, now):
        if self.active or self.speaking:
            raise RuntimeError('Cannot advance while backend or playback is active')
        self.transcript = ''
        self.input_finished = False
        self.last_activity = now


class LoggedLiveService(OpenAILiveLLMService):
    def __init__(self, owner, **kwargs):
        self.owner = owner
        super().__init__(**kwargs)

    async def _handle_server_event(self, evt):
        e = evt.model_dump(mode='json', exclude_none=True)
        self.owner.state.event(e, time.monotonic())
        if e['type'] == 'session.output_audio.delta':
            e['audio_bytes'] = len(base64.b64decode(e.pop('delta')))
        self.owner.log('server', event=e)
        if e['type'] == 'session.started':
            self.owner.ready.set()
        elif e['type'] == 'session.closed':
            self.owner.closed = e
        elif e['type'] == 'error':
            self.owner.error = e
        elif e['type'] == 'response.event' and e['event']['type'] in ('response.failed', 'response.incomplete'):
            self.owner.error = e
        await super()._handle_server_event(evt)


class PlaybackObserver(FrameProcessor):
    def __init__(self, owner):
        self.owner = owner
        super().__init__()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        state = self.owner.state
        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, (BotStartedSpeakingFrame, BotSpeakingFrame)):
                state.speaking = True
                state.last_activity = time.monotonic()
            elif isinstance(frame, BotStoppedSpeakingFrame):
                state.speaking = False
                state.last_activity = time.monotonic()
            if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
                self.owner.log(type(frame).__name__)
        if isinstance(frame, MetricsFrame) and direction == FrameDirection.DOWNSTREAM:
            for metric in frame.data:
                if isinstance(metric, LLMUsageMetricsData):
                    self.owner.recorder.record_usage_metrics(metric.value, getattr(metric, 'model', None))
        if isinstance(frame, ErrorFrame):
            self.owner.error = str(frame)
            self.owner.log('pipeline_error', error=str(frame))
        await self.push_frame(frame, direction)


class OpenAILivePipeline:
    """Standalone adapter using the shared benchmark tools, recorder and audio transport."""
    def __init__(self, benchmark, recorder, *, backend='gpt-5.6-terra', effort='medium', quiet_seconds=3.0, exchange_coordinator=None, recording_observer=None):
        self.benchmark = benchmark
        self.recorder = recorder
        self.backend = backend
        self.effort = effort
        self.exchange_coordinator = exchange_coordinator
        self.recording_observer = recording_observer
        self.state = LiveTurnState(quiet_seconds=quiet_seconds)
        self.ready = asyncio.Event()
        self.closed = None
        self.error = None
        self.end_requested = False
        self.turn = None
        self.started = time.monotonic()
        self.events = (recorder.run_dir/'live-events.jsonl').open('w')
        self.duplicate_ids = set()
        self.seen_calls = set()
        self.audio_queue = asyncio.Queue()

    def log(self, kind, **data):
        self.events.write(json.dumps({'t': time.monotonic()-self.started, 'turn': self.turn, 'type': kind, **data}, default=str)+'\n')
        self.events.flush()

    async def tool(self, params):
        # Same simulated result and duplicate suppression as the shared harness.
        key = (params.function_name, str(params.arguments or {}))
        if key in self.seen_calls:
            self.duplicate_ids.add(params.tool_call_id)
            result = {'status': 'duplicate_skipped'}
        else:
            self.seen_calls.add(key)
            result = {'status': 'success'}
        self.log('tool_result', name=params.function_name, arguments=params.arguments, result=result)
        await params.result_callback(result)
        if params.function_name == 'end_session':
            self.end_requested = True

    async def feed(self):
        """A continuous microphone stream; WAVs replace silence in the stream."""
        deadline = time.monotonic()
        pcm = b''; position = 0; complete = None
        while True:
            if position >= len(pcm):
                if complete is not None:
                    self.state.finish_input(time.monotonic())
                    self.log('input_end')
                    complete.set(); complete = None
                if not self.audio_queue.empty():
                    pcm, complete = self.audio_queue.get_nowait(); position = 0
                    self.log('input_start')
                else:
                    pcm = b''
            chunk = pcm[position:position+960] if pcm else bytes(960)
            position += len(chunk)
            chunk = chunk.ljust(960, b'\0')
            await self.task.queue_frame(InputAudioRawFrame(chunk, 24000, 1))
            deadline += .02
            await asyncio.sleep(max(0, deadline-time.monotonic()))

    async def run(self, indices=None):
        indices = list(range(len(self.benchmark.turns))) if indices is None else list(indices)
        llm = LoggedLiveService(self, api_key=os.environ['OPENAI_API_KEY'],
            settings=OpenAILiveLLMService.Settings(model='gpt-live-1', voice='marin', system_instruction=getattr(self.benchmark, 'frontend_system_instruction', self.benchmark.system_instruction)),
            delegation=ResponsesDelegation(settings=OpenAIResponsesLLMSettings(
                model=self.backend, system_instruction=getattr(self.benchmark, 'backend_system_instruction', self.benchmark.system_instruction),
                reasoning=OpenAIResponsesReasoningConfig(effort=self.effort))))
        for schema in self.benchmark.tools_schema.standard_tools:
            llm.register_function(schema.name, self.tool)
        context = LLMContext([], tools=self.benchmark.tools_schema)
        pair = LLMContextAggregatorPair(context)
        output = NullAudioOutputTransport(TransportParams(audio_out_enabled=True, audio_out_sample_rate=24000))
        buffer = WallClockAlignedAudioBufferProcessor(sample_rate=24000, num_channels=2)
        @buffer.event_handler('on_track_audio_data')
        async def save_audio(processor, user, bot, sample_rate, channels):
            user = np.frombuffer(user, dtype='<i2'); bot = np.frombuffer(bot, dtype='<i2')
            size = max(len(user), len(bot))
            stereo = np.zeros((size, 2), dtype='<i2'); stereo[:len(user),0] = user; stereo[:len(bot),1] = bot
            with wave.open(str(self.recorder.run_dir/'conversation.wav'), 'wb') as f:
                f.setnchannels(2); f.setsampwidth(2); f.setframerate(sample_rate); f.writeframes(stereo.tobytes())
        self.task = PipelineTask(Pipeline([pair.user(), llm,
            ToolCallRecorder(lambda: self.recorder, lambda: self.duplicate_ids),
            pair.assistant(), PlaybackObserver(self), output,
            *([self.recording_observer] if self.recording_observer else []), buffer]),
            params=PipelineParams(audio_in_sample_rate=24000, audio_out_sample_rate=24000, enable_usage_metrics=True))
        @self.task.event_handler('on_pipeline_started')
        async def started(task, frame):
            output.reset_recording_baseline(recording_sample_rate=24000)
            await buffer.start_recording()
            self.log('recording_start')
        runner = asyncio.create_task(PipelineRunner(handle_sigint=False).run(self.task))
        feeder = None
        failure = None
        try:
            await self.task.queue_frame(LLMContextFrame(context))
            await asyncio.wait_for(self.ready.wait(), 30)
            feeder = asyncio.create_task(self.feed())
            # Allow an optional startup greeting to finish before turn zero.
            self.state.last_activity = time.monotonic()
            greeting_deadline = time.monotonic() + 35
            while (self.state.speaking or self.state.active or
                   time.monotonic()-self.state.last_activity < self.state.quiet_seconds):
                if self.error: raise RuntimeError(str(self.error))
                if time.monotonic() > greeting_deadline: raise TimeoutError('Startup audio did not settle')
                await asyncio.sleep(.05)
            for idx in indices:
                self.turn = idx
                self.state.begin(time.monotonic())
                self.recorder.start_turn(idx)
                self.seen_calls.clear(); self.duplicate_ids.clear()
                data, sr = sf.read(self.benchmark.get_audio_path(idx), dtype='int16')
                if data.ndim != 1: raise ValueError('Expected mono input audio')
                if sr != 24000: data = resample_poly(data.astype(float), 24000, sr).astype('<i2')
                complete = asyncio.Event()
                await self.audio_queue.put((data.tobytes(), complete))
                await asyncio.wait_for(complete.wait(), 30)
                until = time.monotonic()+90
                while True:
                    if self.error: raise RuntimeError(str(self.error))
                    if runner.done(): raise RuntimeError('Pipeline stopped before turn completion')
                    if time.monotonic()>until: raise TimeoutError('No completed Live answer within 90 seconds')
                    if self.state.ready(time.monotonic()):
                        if self.exchange_coordinator is None or await self.exchange_coordinator(self):
                            # Speech or delegation may have started during the check.
                            if self.state.ready(time.monotonic()):
                                break
                    await asyncio.sleep(.05)
                self.log('turn_complete', transcript=self.state.transcript, active=list(self.state.active), speaking=self.state.speaking)
                self.recorder.write_turn(user_text=self.benchmark.turns[idx]['input'], assistant_text=self.state.transcript.strip())
                if self.end_requested:
                    if idx != indices[-1]: raise RuntimeError('Model called end_session before the final scripted turn')
                    break
        except (Exception, asyncio.CancelledError) as exc:
            failure = {'reason': type(exc).__name__, 'detail': str(exc), 'turn': self.turn}
            self.log('failure', **failure)
        finally:
            if feeder:
                feeder.cancel(); await asyncio.gather(feeder, return_exceptions=True)
            if not runner.done():
                await self.task.queue_frame(EndFrame())
                try: await asyncio.wait_for(asyncio.shield(runner), 20)
                except asyncio.TimeoutError:
                    await self.task.cancel(); await runner
            if failure is None and self.recorder.total_turns_scored != len(indices):
                failure = {'reason': 'incomplete_conversation', 'turn': self.turn}
            if self.closed is None and failure is None:
                failure = {'reason': 'missing_session_closed', 'turn': self.turn}
            self.recorder.write_summary(status='failed' if failure else 'completed', failure=failure)
            self.events.close()
        if failure: raise RuntimeError(str(failure))
