"""Versioned full-response readback diagnostics; unchanged probe.py remains frozen."""
import asyncio,hashlib,json,sys,time,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import HERE as STUDY,ROOT,MODEL,env,dump,now,sha
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
from multi_turn_eval.pipelines.nova_sonic import NovaSonicLLMServiceWithCompletionSignal

HERE=Path(__file__).resolve().parent

async def probe(case,phase):
    target=HERE/'results'/phase/(case['id']+'.json')
    if target.exists():
        result=json.loads(target.read_text());assert result['request']==case
        return result
    start=time.monotonic();events=[];texts=[];usage=[];private=env()
    llm=NovaSonicLLMServiceWithCompletionSignal(
        access_key_id=private['AWS_ACCESS_KEY_ID'],secret_access_key=private['AWS_SECRET_ACCESS_KEY'],
        session_token=private.get('AWS_SESSION_TOKEN'),region='us-east-1',
        settings=NovaSonicLLMServiceWithCompletionSignal.Settings(model=MODEL,voice='tiffany'),
        endpointing_sensitivity='HIGH')
    result={'at':now(),'request':case,'invoked_model':llm._settings.model,'region':'us-east-1',
            'session_start':json.loads(llm.build_session_start_json()),'outcome':None}
    stream=None;keepalive=None;audio_seen=False
    try:
        async with asyncio.timeout(45):
            stream=await llm.open_stream(llm.create_client());prompt=str(uuid.uuid4());audio=str(uuid.uuid4())
            tools=llm.get_llm_adapter().from_standard_tools(BenchmarkConfig.tools_schema) if case.get('tools',True) else []
            result['tools']=tools
            await llm.send_event(llm.build_session_start_json(),stream)
            await llm.send_prompt_start(tools,prompt,stream)
            for message in case['messages']:
                await llm.send_text(message['text'],message['role'],prompt,stream,False)
            await llm.send_audio_input_start(prompt,audio,stream)
            await llm.send_text(case.get('interactive_input','Hello!'),'USER',prompt,stream,True)
            async def silence():
                while True:
                    await llm.send_audio(bytes(640),prompt,audio,stream);await asyncio.sleep(.02)
            keepalive=asyncio.create_task(silence())
            while True:
                output=await stream.await_output();r=await output[1].receive()
                if r is None:raise RuntimeError('Response stream ended')
                event=json.loads(r.value.bytes_)['event'];events.extend(event)
                if 'textOutput' in event:texts.append(event['textOutput'])
                if 'usageEvent' in event:usage.append(event['usageEvent'])
                if 'audioOutput' in event:audio_seen=True
                if audio_seen and event.get('contentEnd',{}).get('stopReason')=='END_TURN':
                    result['outcome']='generated_audio';break
    except TimeoutError as exc:
        result.update(outcome='timeout',error_type=type(exc).__name__,error=str(exc))
    except Exception as exc:
        result.update(outcome='provider_content_filter' if 'content filters' in str(exc) else 'operational_error',error_type=type(exc).__name__,error=str(exc))
    finally:
        if keepalive:
            keepalive.cancel();await asyncio.gather(keepalive,return_exceptions=True)
        if stream:
            try:await asyncio.wait_for(stream.input_stream.close(),5)
            except Exception as exc:result['cleanup_error']=str(exc)
    result.update(events=events,public_text=texts,usage_events=usage,elapsed_seconds=time.monotonic()-start)
    dump(target,result)
    print(json.dumps({'phase':phase,'id':case['id'],'characters':sum(len(m['text']) for m in case['messages']),
                      'outcome':result['outcome'],'seconds':round(result['elapsed_seconds'],2)}),flush=True)
    return result

async def main(phase):
    requests=HERE/'requests'/(phase+'.json');cases=json.loads(requests.read_text())
    identity={'requests_sha256':sha(requests),'sources_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),STUDY/'common.py',ROOT/'src/multi_turn_eval/pipelines/nova_sonic.py',HERE/'PROTOCOL.md']}}
    path=HERE/'requests'/(phase+'-freeze.json')
    if path.exists():assert json.loads(path.read_text())==identity
    else:dump(path,identity)
    semaphore=asyncio.Semaphore(2)
    async def one(case):
        async with semaphore:return await probe(case,phase)
    results=await asyncio.gather(*(one(c) for c in cases))
    dump(HERE/(phase+'-summary.json'),[{'id':r['request']['id'],'outcome':r['outcome'],
         'characters':sum(len(m['text']) for m in r['request']['messages']),
         'error':r.get('error'),'elapsed_seconds':r['elapsed_seconds']} for r in results])

if __name__=='__main__':asyncio.run(main(sys.argv[1]))
