"""Bounded Pipecat Live probe; excluded from canonical benchmarks."""
import asyncio, base64, json, os, time, wave, argparse
from pathlib import Path
from dotenv import load_dotenv
from pipecat.frames.frames import InputAudioRawFrame, LLMContextFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.live.llm import OpenAILiveLLMService, ResponsesDelegation
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMSettings
from benchmarks.aiwf_medium_context.config import BenchmarkConfig as LegacyBenchmark
class B(LegacyBenchmark):
    from benchmarks.aiwf_medium_context_speech_v2.prompts.system import system_instruction
import numpy as np
from scipy.signal import resample_poly

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='smoke')
args=parser.parse_args()
OUT=Path(__file__).parent/args.output
OUT.mkdir(exist_ok=False)
load_dotenv()
start=time.monotonic(); events=[]; audio=[]; calls=[]; turns=[]; ready=asyncio.Event()
def log(kind, **data):
    row=dict(t=time.monotonic()-start,type=kind,**data);events.append(row)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
class Live(OpenAILiveLLMService):
    async def _handle_server_event(self,evt):
        d=evt.model_dump(mode='json',exclude_none=True)
        if d['type']=='session.output_audio.delta':
            pcm=base64.b64decode(d.pop('delta'));audio.append((time.monotonic()-start,pcm));d['bytes']=len(pcm)
        log('server',event=d)
        await super()._handle_server_event(evt)
class Sink(FrameProcessor):
    async def process_frame(self,frame,direction):
        await super().process_frame(frame,direction)
        if type(frame).__name__=='ErrorFrame':log('pipeline_error',error=str(frame))
        await self.push_frame(frame,direction)
async def main():
    llm=Live(api_key=os.environ['OPENAI_API_KEY'],settings=Live.Settings(model='gpt-live-1',voice='marin',system_instruction='You are the AI Engineer World\'s Fair voice assistant. Be concise and natural. Delegate all conference facts, recommendations, and requested actions to the backend. Ask only for missing information. Relay its answers accurately. Confirm actions only after the backend reports success.'),delegation=ResponsesDelegation(settings=OpenAIResponsesLLMSettings(model='gpt-5.6-terra',system_instruction=B.system_instruction)))
    @llm.event_handler('on_session_started')
    async def on_ready(service,session):ready.set()
    async def tool(params):
        calls.append(dict(name=params.function_name,arguments=params.arguments));log('tool',**calls[-1])
        await params.result_callback({'status':'success','message':'Benchmark simulation: request recorded.'})
    for schema in B.tools_schema.standard_tools:llm.register_function(schema.name,tool)
    context=LLMContext([],tools=B.tools_schema); pair=LLMContextAggregatorPair(context)
    task=PipelineTask(Pipeline([pair.user(),llm,Sink(),pair.assistant()]),params=PipelineParams(audio_in_sample_rate=24000,audio_out_sample_rate=24000))
    async def send(pcm):
        for pos in range(0,len(pcm),960):
            await task.queue_frame(InputAudioRawFrame(pcm[pos:pos+960],24000,1));await asyncio.sleep(.02)
    async def drive():
        await task.queue_frame(LLMContextFrame(context));await asyncio.wait_for(ready.wait(),30)
        await send(bytes(48000))
        for idx in [0,9,10,11]:
            with wave.open(str(B.get_audio_path(idx))) as w:
                assert w.getnchannels()==1 and w.getsampwidth()==2
                pcm=w.readframes(w.getnframes());sr=w.getframerate()
            if sr!=24000:pcm=resample_poly(np.frombuffer(pcm,dtype='<i2').astype(float),24000,sr).astype('<i2').tobytes()
            log('input_start',turn=idx);await send(pcm);end=time.monotonic()-start;log('input_end',turn=idx)
            # Fixed generous listening window; silence continues as a microphone would.
            await send(bytes(48000*25))
            turns.append(dict(turn=idx,input_end=end))
        await task.queue_frame(EndFrame())
    runner=asyncio.create_task(PipelineRunner(handle_sigint=False).run(task));driver=asyncio.create_task(drive())
    try:await asyncio.wait_for(asyncio.gather(runner,driver),180)
    finally:
        if not runner.done():await task.cancel()
        driver.cancel()
        with wave.open(str(OUT/'assistant.wav'),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000)
            cursor=0
            for t,pcm in audio:
                offset=int(t*24000);w.writeframes(bytes(max(0,offset-cursor)*2));w.writeframes(pcm);cursor=max(offset,cursor)+len(pcm)//2
        (OUT/'summary.json').write_text(json.dumps(dict(model='gpt-live-1',backend='gpt-5.6-terra',openai=__import__('openai').__version__,tools=calls,turns=turns,audio_chunks=len(audio),duration=time.monotonic()-start),indent=2))
asyncio.run(main())
