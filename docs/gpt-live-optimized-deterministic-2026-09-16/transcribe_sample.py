"""Independent per-channel timestamped ASR; no KB/expected-answer hints."""
import argparse,asyncio,hashlib,io,json,wave
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
async def main():
 p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--turns',default='0,4,6');a=p.parse_args();load_dotenv();run=Path(a.run);out=run/'audio-review';out.mkdir(exist_ok=True)
 events=[json.loads(l) for l in (run/'live-events.jsonl').read_text().splitlines()];base=next(e['t'] for e in events if e['type']=='recording_start')
 with wave.open(str(run/'conversation.wav'),'rb') as f:rate=f.getframerate();audio=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').reshape(-1,2)
 sem=asyncio.Semaphore(2)
 async with AsyncOpenAI() as client:
  async def transcribe(turn,channel):
   start=next(e['t'] for e in events if e['type']=='input_start' and e['turn']==turn)-base;end=next(e['t'] for e in events if e['type']=='turn_complete' and e['turn']==turn)-base
   first=max(0,int((start-.25)*rate));last=min(len(audio),int(end*rate));key=f'turn-{turn:02d}-{channel}'
   wav=out/(key+'.wav')
   with wave.open(str(wav),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes(audio[first:last,channel].tobytes())
   target=out/(key+'.json')
   if target.exists():return
   async with sem:
    with wav.open('rb') as f:r=await client.audio.transcriptions.create(model='whisper-1',file=f,response_format='verbose_json',timestamp_granularities=['word','segment'],language='en',temperature=0)
   target.write_text(json.dumps(dict(model='whisper-1',channel='user' if channel==0 else 'assistant',turn=turn,clip_start_sample=first,sample_rate=rate,audio_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),parameters=dict(response_format='verbose_json',timestamp_granularities=['word','segment'],language='en',temperature=0,prompt=None),transcription=r.model_dump()),indent=2)+'\n');print(key,r.text,flush=True)
  await asyncio.gather(*(transcribe(int(t),c) for t in a.turns.split(',') for c in [0,1]))
if __name__=='__main__':asyncio.run(main())
