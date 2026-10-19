import subprocess,sys,time
from common import HERE,ROOT
while not (HERE/'judging/COMPLETE.json').exists():time.sleep(15)
subprocess.run([sys.executable,str(HERE/'analyze.py')],cwd=ROOT,check=True)
print('Fresh Responses cohorts analyzed; publication awaits main-worktree integration after frozen Grok high study.',flush=True)
