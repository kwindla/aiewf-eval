"""Configuration-only probe of draft prompts. No audio, tools, or benchmark score."""
import asyncio,hashlib,json
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
HERE=Path(__file__).resolve().parent
async def main():
 load_dotenv();frontend=(HERE/'frontend.txt').read_text();backend=(HERE/'backend.txt').read_text();events=[]
 async with AsyncOpenAI() as client:
  async with client.live.connect() as connection:
   await connection.session.start(session={'model':'gpt-live-1','instructions':frontend,'audio':{'format':{'type':'audio/pcm','rate':24000},'output':{'voice':'marin'}},'delegation':{'type':'responses','responses':{'model':'gpt-5.6-terra','instructions':backend,'reasoning':{'effort':'medium'}}}})
   async with asyncio.timeout(30):
    async for event in connection:
     data=event.model_dump(mode='json',exclude_none=True);events.append(data)
     if event.type=='session.started':
      assert data['session']['instructions']==frontend
      assert data['session']['delegation']['responses']['instructions']==backend
      await connection.session.close()
     if event.type in ['session.closed','error']:break
 result=dict(kind='configuration acceptance only; excluded from all benchmarks',frontend_sha256=hashlib.sha256(frontend.encode()).hexdigest(),backend_sha256=hashlib.sha256(backend.encode()).hexdigest(),events=events)
 (HERE/'acceptance-probe.json').write_text(json.dumps(result,indent=2)+'\n')
 assert any(e['type']=='session.started' for e in events) and any(e['type']=='session.closed' for e in events)
 assert not any(e['type']=='error' for e in events)
 print('Distinct frontend/backend prompts accepted byte-for-byte; session started and closed. No behavioral inference.')
asyncio.run(main())
