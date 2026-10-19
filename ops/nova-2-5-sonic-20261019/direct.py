"""Excluded native Bedrock 2.5 invocation and tool-result continuation probe."""
import asyncio, json, uuid
from common import HERE,MODEL,env,dump,now
from multi_turn_eval.pipelines.nova_sonic import NovaSonicLLMServiceWithCompletionSignal

async def main():
    path=HERE/'smoke/direct.json'
    if path.exists():return
    private=env()
    llm=NovaSonicLLMServiceWithCompletionSignal(
        access_key_id=private['AWS_ACCESS_KEY_ID'],secret_access_key=private['AWS_SECRET_ACCESS_KEY'],
        session_token=private.get('AWS_SESSION_TOKEN'),region='us-east-1',
        settings=NovaSonicLLMServiceWithCompletionSignal.Settings(model=MODEL,voice='tiffany'),
        endpointing_sensitivity='HIGH')
    client=llm.create_client();stream=await llm.open_stream(client)
    prompt=str(uuid.uuid4());llm._stream=stream;llm._prompt_name=prompt
    events=[];text=[];calls=[];audio_bytes=0;answered=False
    await llm.send_event(llm.build_session_start_json(),stream)
    tool={'toolSpec':{'name':'probe','description':'Return a test number.',
        'inputSchema':{'json':json.dumps({'type':'object','properties':{'value':{'type':'integer'}},'required':['value']})}}}
    await llm.send_prompt_start([tool],prompt,stream)
    await llm.send_text('Call probe with value 323. When the tool succeeds, say OK.', 'SYSTEM',prompt,stream,False)
    audio_content=str(uuid.uuid4())
    await llm.send_audio_input_start(prompt,audio_content,stream)
    await llm.send_text('Please run the test now.', 'USER',prompt,stream,True)
    async def silence():
        while True:
            await llm.send_audio(bytes(640),prompt,audio_content,stream)
            await asyncio.sleep(.02)
    keepalive=asyncio.create_task(silence())
    try:
        async with asyncio.timeout(120):
            while True:
                output=await stream.await_output();r=await output[1].receive()
                if r is None:raise RuntimeError('Stream ended before tool continuation')
                if not getattr(r,'value',None):raise RuntimeError(str(r))
                event=json.loads(r.value.bytes_)['event'];events.extend(event)
                if 'audioOutput' not in event: print(json.dumps(event),flush=True)
                if 'textOutput' in event:text.append(event['textOutput']['content'])
                if 'audioOutput' in event:audio_bytes+=len(event['audioOutput'].get('content',''))
                if 'toolUse' in event:
                    call=event['toolUse'];calls.append(call)
                    assert call['toolName']=='probe' and json.loads(call['content'])=={'value':323},call
                if 'contentEnd' in event and event['contentEnd'].get('stopReason')=='TOOL_USE' and calls and not answered:
                    await llm._send_tool_result(calls[-1]['toolUseId'],{'status':'success'})
                    answered=True
                if answered and audio_bytes and 'OK' in ''.join(text).upper() and 'contentEnd' in event and event['contentEnd'].get('stopReason')=='END_TURN':break
        dump(path,{'at':now(),'requested_model':MODEL,'invoked_model':llm._settings.model,'region':'us-east-1',
                   'session_start':json.loads(llm.build_session_start_json()),'events':events,'calls':calls,
                   'public_text':text,'audio_base64_bytes':audio_bytes,'tool_result_sent':answered,'verified':True})
        print('Native 2.5 model and tool round-trip verified',flush=True)
    finally:
        keepalive.cancel()
        await asyncio.gather(keepalive,return_exceptions=True)
        await stream.input_stream.close()

if __name__=='__main__':asyncio.run(main())
