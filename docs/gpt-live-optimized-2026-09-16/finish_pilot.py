"""Finish pilot validation after capture cleanup, retaining all intermediate outputs."""
import json,re,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
log=HERE/'pilot-full-v1.log';run=None
for _ in range(150):
 m=re.search(r'^Output directory: (.+)$',log.read_text(),re.M)
 if m:
  run=ROOT/m[1]
  if (run/'runtime.json').exists() and (run/'incremental-conversation.wav').exists():break
 time.sleep(10)
else:raise RuntimeError('Pilot capture has not finalized within 25 minutes')
for name,args in [('process',[HERE/'process_run.py',run]),('asr',[HERE/'transcribe_sample.py',run,'--turns','0,14,25,29']),('summary',[HERE/'summarize_run.py',run])]:
 with (HERE/f'pilot-full-{name}.log').open('w') as f:
  code=subprocess.call([sys.executable,*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if code:print(name,'failed',code,flush=True)
print('Pilot validation pipeline finished:',run,flush=True)
