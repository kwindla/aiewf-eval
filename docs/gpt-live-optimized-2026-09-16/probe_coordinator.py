"""Live API boundary probes; expected labels never sent to the coordinator."""
import asyncio,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv
from coordinator import ExchangeCoordinator
CASES=[('progress','When are the workshops?','Let me check the program for you.',False),('incorrect_answer','When are the workshops?','The workshops are on February 30.',True),('clarification','Can you recommend a session?','What topics interest you?',True),('social','Thanks!','You are welcome.',True)]
async def main():
 load_dotenv();results=[]
 for name,user,assistant,expected in CASES:
  c=ExchangeCoordinator();events=[]
  with tempfile.TemporaryDirectory() as d:
   owner=SimpleNamespace(turn=0,state=SimpleNamespace(transcript=assistant),recorder=SimpleNamespace(run_dir=Path(d)),benchmark=SimpleNamespace(turns=[{'input':user}]),log=lambda kind,**data:events.append(dict(type=kind,**data)))
   try:actual=await c(owner)
   finally:await c.close()
  results.append(dict(case=name,user=user,assistant=assistant,expected_ready=expected,actual_ready=actual,events=events));print(name,actual,flush=True)
 path=Path(__file__).with_name('coordinator-probes.json');path.write_text(json.dumps(results,indent=2)+'\n')
 assert all(r['expected_ready']==r['actual_ready'] for r in results)
if __name__=='__main__':asyncio.run(main())
