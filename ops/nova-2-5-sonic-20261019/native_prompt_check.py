"""Excluded diagnostic: submit the unchanged full benchmark via native helpers."""
import asyncio,json,uuid,hashlib
from common import HERE,MODEL,env,dump,now
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
from multi_turn_eval.pipelines.nova_sonic import NovaSonicLLMServiceWithCompletionSignal

async def main():
    private=env();benchmark=BenchmarkConfig()
    llm=NovaSonicLLMServiceWithCompletionSignal(
        access_key_id=private['AWS_ACCESS_KEY_ID'],secret_access_key=private['AWS_SECRET_ACCESS_KEY'],
        session_token=private.get('AWS_SESSION_TOKEN'),region='us-east-1',
        settings=NovaSonicLLMServiceWithCompletionSignal.Settings(model=MODEL,voice='tiffany'),
        endpointing_sensitivity='HIGH')
    stream=await llm.open_stream(llm.create_client());prompt=str(uuid.uuid4());audio=str(uuid.uuid4())
    tools=llm.get_llm_adapter().from_standard_tools(benchmark.tools_schema)
    request={'at':now(),'requested_model':MODEL,'invoked_model':llm._settings.model,
        'system_prompt_sha256':hashlib.sha256(benchmark.system_instruction.encode()).hexdigest(),
        'system_prompt_characters':len(benchmark.system_instruction),'tools':tools,
        'interactive_input':'Hello!','events':[],'outcome':None}
    await llm.send_event(llm.build_session_start_json(),stream)
    await llm.send_prompt_start(tools,prompt,stream)
    await llm.send_text(benchmark.system_instruction,'SYSTEM',prompt,stream,False)
    await llm.send_audio_input_start(prompt,audio,stream)
    await llm.send_text('Hello!','USER',prompt,stream,True)
    async def silence():
        while True:
            await llm.send_audio(bytes(640),prompt,audio,stream);await asyncio.sleep(.02)
    keepalive=asyncio.create_task(silence())
    try:
        async with asyncio.timeout(45):
            while True:
                output=await stream.await_output();r=await output[1].receive()
                if r is None:raise RuntimeError('Stream ended')
                event=json.loads(r.value.bytes_)['event'];request['events'].extend(event)
                if 'audioOutput' in event:
                    request['outcome']='generated_audio';break
    except Exception as exc:
        request.update(outcome='provider_content_filter' if 'content filters' in str(exc) else 'operational_error',error_type=type(exc).__name__,error=str(exc))
    finally:
        keepalive.cancel();await asyncio.gather(keepalive,return_exceptions=True)
        await stream.input_stream.close()
        dump(HERE/'smoke/native-full-prompt.json',request)
        print(json.dumps(request),flush=True)

if __name__=='__main__':asyncio.run(main())
