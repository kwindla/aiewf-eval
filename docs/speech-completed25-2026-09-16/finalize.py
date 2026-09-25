"""Wait for authorized top-ups, then aggregate and publish the new policy."""
import subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for _ in range(720):
 if (HERE/'topups-complete.json').exists():break
 time.sleep(10)
else:raise RuntimeError('Top-ups have not completed; no publication')
for name in ['report.py','publish.py']:
 subprocess.run([sys.executable,str(HERE/name)],cwd=ROOT,check=True)
