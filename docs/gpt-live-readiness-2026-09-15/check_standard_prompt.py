"""Check exact standard speech prompt acceptance at both Live layers."""
import asyncio,json,os,hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from benchmarks.aiwf_medium_context_speech_v2.prompts.system import system_instruction
async def main():
 load_dotenv();events=[]
 async with AsyncOpenAI() as client:
  async with client.live.connect() as conn:
   await conn.session.start(session={'model':'gpt-live-1','instructions':system_instruction,'audio':{'format':{'type':'audio/pcm','rate':24000},'output':{'voice':'marin'}},'delegation':{'type':'responses','responses':{'model':'gpt-5.6-terra','instructions':system_instruction,'reasoning':{'effort':'medium'}}}})
   async with asyncio.timeout(30):
    async for event in conn:
     data=event.model_dump(mode='json',exclude_none=True);events.append(data)
     if event.type=='session.started':
      assert data['session']['instructions']==system_instruction
      assert data['session']['delegation']['responses']['instructions']==system_instruction
      await conn.session.close()
     if event.type in ['session.closed','error']:break
 out=Path(__file__).parent/'standard-prompt-acceptance.json'
 out.write_text(json.dumps({'prompt_sha256':hashlib.sha256(system_instruction.encode()).hexdigest(),'events':events},indent=2))
 print([e['type'] for e in events])
 for e in events:
  if e['type']=='error':print(e)
 assert any(e['type']=='session.started' for e in events) and any(e['type']=='session.closed' for e in events)
asyncio.run(main())
