"""Schedule diagnostics only after frozen references and development checks."""
import subprocess
import sys
import time
from common import *

def main():
    while not (HERE/'references/FROZEN.json').exists() or not (HERE/'development-B-COMPLETE.json').exists():
        time.sleep(5)
    assert read(HERE/'references/FROZEN.json')['minimum_gate_passed']
    dump(HERE/'validation-request-registration.json',{'reference_freeze_sha256':sha(HERE/'references/FROZEN.json'),
         'selection_sha256':sha(HERE/'validation-selection.json'),'driver_sha256':sha(HERE/'driver-sha256.json'),
         'repeats':['validation-A','validation-B'],'ownership':'diagnostic only; canonical bulk is separately first-valid'})
    for phase in ['validation-A','validation-B']:
        subprocess.run([sys.executable,str(HERE/'driver.py'),phase],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(HERE/'validate.py')],cwd=ROOT,check=True)

if __name__=='__main__':main()
