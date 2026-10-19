"""Start the prospective refresh only after final validation has completed."""
import json,os,subprocess,sys,time
from common import HERE,ROOT,dump
while not (HERE/'SMOKE-COMPLETE.json').exists():time.sleep(5)
pids={}
env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)
for name,script in [('collection','collect_campaign.py'),('judging','judge.py'),('analysis','finish_analysis.py')]:
 with (HERE/(name+'.log')).open('a') as out:
  proc=subprocess.Popen([sys.executable,str(HERE/script)],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 pids[name]=proc.pid
 dump(HERE/'processes.json',pids)
print(json.dumps(pids),flush=True)
